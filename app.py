"""
NeuroMPR PACS — Flask backend
=============================
A small, complete PACS around the NeuroMPR MPR/3D viewer, built for BraTS-format
brain-tumour MRI (NIfTI) and plain DICOM:

  Archive      SQLite index of patients / studies / series; your BraTS dataset
               folders are indexed in place (files are never modified); uploads
               and DICOM arrivals are stored under pacs_data/
  Worklist     search, filters, priority, reading status, assignment, priors
  Viewer       the NeuroMPR viewer (triplanar MPR, measurements, segmentation,
               niivue 3D with atlas-fitted scalp & skull), opened per study
  Reports      versioned draft -> preliminary -> final (signed) + addenda
  Annotations  measurements saved per study
  DICOM        import (files / zip), export (zip, optional anonymisation),
               Storage SCP (C-STORE in), C-ECHO, C-FIND, C-GET, C-MOVE,
               Storage SCU (send to another PACS)        [needs pydicom/pynetdicom]
  Security     user accounts with roles (admin / radiologist / viewer), audit log

Run:
    pip install flask nibabel numpy scipy opencv-python-headless
    pip install pydicom pynetdicom          # DICOM import/export/network (optional)
    pip install torch monai                 # model segmentation (optional)
    python app.py        ->  http://127.0.0.1:5000  (first visit creates the admin account)

Files: app.py (this), headmodel.py, dicom_io.py, static/{pacs,viewer,login}.html,
static/vendor/niivue, assets/head_atlas.npz. Runtime data goes to pacs_data/.
"""

import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import hashlib
import io
import json
import queue
import re
import secrets
import shutil
import sqlite3
import threading
import time
import traceback
import uuid
import zipfile
from collections import OrderedDict
from functools import wraps
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from flask import (Flask, Response, g, has_request_context, jsonify, redirect, request,
                   send_file, send_from_directory, session)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import headmodel
import dicom_io

# --------------------------------------------------------------------------------------
# CONFIG — edit these for your machine
# --------------------------------------------------------------------------------------
DATASET_ROOTS = [r"C:\Users\USER\Documents\BRATS_samplefiles"]   # folders scanned for BraTS case folders (more can be added in Admin)
DATA_DIR = Path(__file__).parent / "pacs_data"                   # database, archive, caches, reports
REQUIRE_LOGIN = True                                             # False = single-user, no login screen
HOST, PORT = "127.0.0.1", 5000                                   # use "0.0.0.0" to serve other PCs on your network
DICOM_SCP_ENABLED = True                                         # receive / answer DICOM network requests
DICOM_AE_TITLE = "NEUROMPR"
DICOM_PORT = 11112
CASE_CACHE_MAX = 3                                               # studies kept in RAM (~150 MB each for BraTS)
SCAN_ON_STARTUP = True
OPEN_BROWSER = True                                              # open the worklist in your browser on start
MODEL_PATH = r""                                                 # optional MONAI SegResNet checkpoint

# optional overrides (handy for testing without editing this file)
DATA_DIR = Path(os.environ.get("NEUROMPR_DATA_DIR", DATA_DIR))
if os.environ.get("NEUROMPR_DATASET_ROOTS"):
    DATASET_ROOTS = [p for p in os.environ["NEUROMPR_DATASET_ROOTS"].split(os.pathsep) if p]
PORT = int(os.environ.get("NEUROMPR_PORT", PORT))
DICOM_PORT = int(os.environ.get("NEUROMPR_DICOM_PORT", DICOM_PORT))
REQUIRE_LOGIN = os.environ.get("NEUROMPR_REQUIRE_LOGIN", str(REQUIRE_LOGIN)).lower() in ("1", "true", "yes")

DB_PATH = DATA_DIR / "pacs.db"
ARCHIVE_DIR = DATA_DIR / "archive"          # NIfTI study folders for uploaded / DICOM-derived studies
DICOM_STORE = DATA_DIR / "dicom_store"      # original DICOM instances (received or imported)
CACHE_DIR = DATA_DIR / "cache"
DERIVED_DIR = DATA_DIR / "derived"          # saved auto/model segmentations
REPORTS_DIR = DATA_DIR / "reports"
TMP_DIR = DATA_DIR / "tmp"
for _d in (DATA_DIR, ARCHIVE_DIR, DICOM_STORE, CACHE_DIR / "thumbs", CACHE_DIR / "headmodel",
           DERIVED_DIR, REPORTS_DIR, TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 2048 * 1024 * 1024
_key_file = DATA_DIR / "secret.key"
if not _key_file.exists():
    _key_file.write_text(secrets.token_hex(32))
app.secret_key = _key_file.read_text().strip()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=12 * 3600)

ROLES = ("admin", "radiologist", "viewer")
STATUS_ORDER = ("unread", "opened", "draft", "preliminary", "final")
LABEL_NAMES = {1: "Necrotic / non-enhancing core (NCR/NET)", 2: "Peritumoral edema (ED)",
               4: "Enhancing tumor (ET)", 5: "Resection cavity (RC)"}
TUMOR_LABELS = (1, 2, 4)                      # internal label scheme (BraTS 2021); RC is not tumour
TUMOR_OVERLAY_COLOR_BGR = (0, 0, 255)

# ======================================================================================
# DATABASE
# ======================================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS patients(
  id INTEGER PRIMARY KEY, patient_id TEXT UNIQUE NOT NULL, name TEXT DEFAULT '', sex TEXT DEFAULT '',
  birth_date TEXT DEFAULT '', notes TEXT DEFAULT '', created_at REAL);
CREATE TABLE IF NOT EXISTS studies(
  id INTEGER PRIMARY KEY, study_uid TEXT UNIQUE NOT NULL,
  patient_pk INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  accession TEXT DEFAULT '', study_date TEXT DEFAULT '', description TEXT DEFAULT '', modality TEXT DEFAULT 'MR',
  source TEXT, folder TEXT NOT NULL, dataset_root TEXT DEFAULT '', collection TEXT DEFAULT '',
  label_scheme TEXT DEFAULT 'brats2021', timepoint TEXT DEFAULT '', signature TEXT DEFAULT '',
  status TEXT DEFAULT 'unread', priority TEXT DEFAULT 'routine', assigned_to TEXT DEFAULT '',
  flagged INTEGER DEFAULT 0, hidden INTEGER DEFAULT 0, missing INTEGER DEFAULT 0,
  has_seg INTEGER DEFAULT 0, seg_source TEXT DEFAULT '', tumor_ml REAL,
  num_series INTEGER DEFAULT 0, size_bytes INTEGER DEFAULT 0,
  indexed_at REAL, last_opened REAL, opened_by TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS series(
  id INTEGER PRIMARY KEY, study_pk INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
  series_uid TEXT NOT NULL, key TEXT NOT NULL, description TEXT DEFAULT '', file TEXT NOT NULL,
  number INTEGER DEFAULT 0, dims TEXT DEFAULT '', spacing TEXT DEFAULT '', dtype TEXT DEFAULT '',
  size_bytes INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS reports(
  id INTEGER PRIMARY KEY, study_pk INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
  version INTEGER NOT NULL, status TEXT NOT NULL, fields TEXT DEFAULT '{}', html_file TEXT DEFAULT '',
  author TEXT DEFAULT '', created_at REAL, updated_at REAL, signed_by TEXT DEFAULT '', signed_at REAL,
  parent_id INTEGER);
CREATE TABLE IF NOT EXISTS annotations(
  study_pk INTEGER PRIMARY KEY REFERENCES studies(id) ON DELETE CASCADE,
  data TEXT DEFAULT '[]', updated_by TEXT DEFAULT '', updated_at REAL);
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL, role TEXT NOT NULL,
  full_name TEXT DEFAULT '', active INTEGER DEFAULT 1, created_at REAL, last_login REAL);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY, ts REAL, username TEXT, action TEXT, study_pk INTEGER, detail TEXT DEFAULT '', ip TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS dicom_nodes(
  id INTEGER PRIMARY KEY, name TEXT, ae_title TEXT NOT NULL, host TEXT NOT NULL, port INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS ix_studies_patient ON studies(patient_pk);
CREATE INDEX IF NOT EXISTS ix_studies_status ON studies(status);
CREATE INDEX IF NOT EXISTS ix_studies_folder ON studies(folder);
CREATE INDEX IF NOT EXISTS ix_series_study ON series(study_pk);
CREATE INDEX IF NOT EXISTS ix_reports_study ON reports(study_pk);
CREATE INDEX IF NOT EXISTS ix_audit_study ON audit(study_pk);
CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit(ts);
"""
_tls = threading.local()


def _connect():
    c = sqlite3.connect(str(DB_PATH), timeout=30, isolation_level=None, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def db():
    if has_request_context():
        if "db" not in g:
            g.db = _connect()
        return g.db
    c = getattr(_tls, "conn", None)
    if c is None:
        c = _tls.conn = _connect()
    return c


@app.teardown_appcontext
def _close_db(exc):
    c = g.pop("db", None)
    if c is not None:
        c.close()


class tx:
    """with tx() as c: ...  — one write transaction."""
    def __enter__(self):
        self.c = db()
        self.c.execute("BEGIN IMMEDIATE")
        return self.c

    def __exit__(self, et, ev, tb):
        self.c.execute("ROLLBACK" if et else "COMMIT")
        return False


def q1(sql, args=()):
    r = db().execute(sql, args).fetchone()
    return dict(r) if r else None


def qa(sql, args=()):
    return [dict(r) for r in db().execute(sql, args).fetchall()]


def init_db():
    c = _connect()
    c.executescript(SCHEMA)
    c.close()


def get_setting(key, default=None):
    r = q1("SELECT value FROM settings WHERE key=?", (key,))
    return json.loads(r["value"]) if r else default


def set_setting(key, value):
    db().execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, json.dumps(value)))


def jerr(msg, code=400, **extra):
    return jsonify({"error": msg, **extra}), code


# ======================================================================================
# AUTH, ROLES, AUDIT
# ======================================================================================
LOCAL_USER = {"id": 0, "username": "local", "role": "admin", "full_name": "Local user"}
PUBLIC_PATHS = {"/login", "/setup", "/api/login", "/api/setup", "/api/me", "/favicon.ico"}


def users_exist():
    return q1("SELECT 1 AS x FROM users LIMIT 1") is not None


def current_user():
    if not REQUIRE_LOGIN:
        return LOCAL_USER
    if "user" in g:
        return g.user
    uid = session.get("uid")
    u = q1("SELECT id, username, role, full_name, active FROM users WHERE id=?", (uid,)) if uid else None
    g.user = u if (u and u["active"]) else None
    return g.user


@app.before_request
def _gate():
    p = request.path
    if p in PUBLIC_PATHS or p.startswith("/static/"):
        return None
    if not REQUIRE_LOGIN:
        return None
    if not users_exist():
        return jerr("setup_required", 401) if p.startswith("/api/") else redirect("/setup")
    if not current_user():
        if p.startswith("/api/") or p.startswith("/reports/"):
            return jerr("login_required", 401)
        return redirect("/login?next=" + request.full_path.rstrip("?"))
    return None


def require_role(*roles):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            u = current_user()
            if not u or (u["role"] != "admin" and u["role"] not in roles):
                return jerr(f"permission denied (needs {' or '.join(roles)})", 403)
            return fn(*a, **kw)
        return wrapper
    return deco


def audit(action, study_pk=None, detail="", username=None):
    try:
        if username is None:
            u = current_user() if has_request_context() else None
            username = u["username"] if u else "system"
        ip = request.remote_addr if has_request_context() else ""
        db().execute("INSERT INTO audit(ts, username, action, study_pk, detail, ip) VALUES(?,?,?,?,?,?)",
                     (time.time(), username, action, study_pk, str(detail)[:1000], ip or ""))
    except Exception:
        traceback.print_exc()


# ======================================================================================
# BraTS NAMING, DATASET INDEXING
# ======================================================================================
# matched against lower-case file names; t1ce is tested before t1
MODALITY_KEYS = [
    ("t1ce",  ["_t1ce.", "-t1ce.", "t1ce.nii", "-t1c.", "_t1c.", "t1c.nii", "t1gd"]),
    ("t1",    ["_t1.", "-t1.", "t1.nii", "-t1n.", "_t1n.", "t1n.nii", "t1w"]),
    ("t2",    ["_t2.", "-t2.", "t2.nii", "-t2w.", "_t2w.", "t2w.nii"]),
    ("flair", ["_flair.", "-flair.", "flair.nii", "-t2f.", "_t2f.", "t2f.nii"]),
    ("seg",   ["_seg.", "-seg.", "seg.nii", "label"]),
]
SERIES_TITLES = {"flair": "FLAIR", "t1": "T1", "t1ce": "T1 post-contrast (T1CE)", "t2": "T2",
                 "seg": "Tumor segmentation (dataset ground truth)"}
BRATS_SUBTYPES = {"GLI": "glioma", "MEN": "meningioma", "MET": "brain metastases", "PED": "pediatric tumours",
                  "SSA": "glioma, Sub-Saharan Africa", "MEN-RT": "meningioma radiotherapy"}
MOD_ORDER = {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3, "seg": 9}


def is_nifti(name):
    n = name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def classify_file(name):
    low = name.lower()
    for mod, keys in MODALITY_KEYS:
        if any(k in low for k in keys):
            return mod
    return None


def stable_uid(*parts):
    return "2.25." + str(uuid.uuid5(uuid.NAMESPACE_URL, "neurompr:" + ":".join(str(p) for p in parts)).int)


def describe_case_folder(folder: Path, files: dict):
    """Patient / timepoint / collection / label scheme from BraTS naming conventions."""
    name = folder.name
    new_style = any(re.search(r"[-_](t1n|t1c|t2w|t2f)\.nii", f.name.lower()) for f in files.values())
    patient, timepoint, collection = name, "", ""
    m = re.match(r"^(BraTS-([A-Z]+(?:-RT)?)-\d+)-(\d+)$", name)
    if m:
        patient, timepoint = m.group(1), m.group(3)
        collection = f"BraTS-{m.group(2)} ({BRATS_SUBTYPES.get(m.group(2), m.group(2).lower())})"
    elif re.match(r"^BraTS20\d\d", name):
        collection = "BraTS " + name[5:9]
    elif new_style:
        collection = "BraTS 2023+"
    elif any(k in files for k in ("flair", "t1ce", "t2")):
        collection = "BraTS-format MRI"
    else:
        collection = "NIfTI"
    mods = [k.upper() for k in sorted(files, key=lambda k: MOD_ORDER.get(k, 5)) if k != "seg"]
    desc = f"MRI brain — {collection} ({' / '.join(mods)})"
    return {"patient_id": patient, "timepoint": timepoint, "collection": collection,
            "label_scheme": "brats2023" if new_style else "brats2021", "description": desc}


def folder_case_files(folder: Path):
    """{modality: Path} for a BraTS-style folder (first match per modality)."""
    out = {}
    try:
        entries = sorted(p for p in folder.iterdir() if p.is_file() and is_nifti(p.name))
    except OSError:
        return out
    for f in entries:
        mod = classify_file(f.name)
        if mod and mod not in out:
            out[mod] = f
    return out


def _signature(paths):
    parts = []
    for p in sorted(paths):
        try:
            st = Path(p).stat()
            parts.append(f"{Path(p).name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            parts.append(f"{Path(p).name}:missing")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _header_info(path: Path):
    img = nib.load(str(path))
    return ("x".join(str(int(s)) for s in img.shape[:3]),
            "x".join(f"{float(z):.3g}" for z in img.header.get_zooms()[:3]),
            str(img.get_data_dtype()))


def upsert_patient(c, patient_id, name="", sex="", birth_date=""):
    r = c.execute("SELECT id, name, sex, birth_date FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
    if r:
        # fill blanks only — never overwrite edits made in the PACS
        upd = {k: v for k, v in (("name", name), ("sex", sex), ("birth_date", birth_date)) if v and not r[k]}
        if upd:
            c.execute(f"UPDATE patients SET {', '.join(k + '=?' for k in upd)} WHERE id=?", (*upd.values(), r["id"]))
        return r["id"]
    cur = c.execute("INSERT INTO patients(patient_id, name, sex, birth_date, created_at) VALUES(?,?,?,?,?)",
                    (patient_id, name, sex, birth_date, time.time()))
    return cur.lastrowid


def index_study(folder: Path, *, source, root="", manifest=None):
    """Create/refresh one study (and its patient + series) from a folder of NIfTI files.
    Dataset folders follow BraTS naming; archive folders carry a study.json manifest."""
    folder = Path(folder)
    if manifest is None and (folder / "study.json").exists():
        manifest = json.loads((folder / "study.json").read_text())
    if manifest:
        files = {s["key"]: folder / s["file"] for s in manifest["series"]}
        info = {"patient_id": manifest.get("patient_id") or folder.name, "timepoint": "",
                "collection": manifest.get("collection") or ("DICOM import" if manifest.get("source") == "dicom" else "Upload"),
                "label_scheme": manifest.get("label_scheme", "brats2021"),
                "description": manifest.get("description") or ""}
        study_uid = manifest.get("study_uid") or stable_uid("folder", str(folder))
        titles = {s["key"]: s.get("description") or SERIES_TITLES.get(s["key"], s["key"]) for s in manifest["series"]}
        suids = {s["key"]: s.get("series_uid") or stable_uid(study_uid, s["key"]) for s in manifest["series"]}
        numbers = {s["key"]: s.get("number") or i + 1 for i, s in enumerate(manifest["series"])}
    else:
        files = folder_case_files(folder)
        info = describe_case_folder(folder, files)
        study_uid = stable_uid("study", folder.name)
        titles = {k: SERIES_TITLES.get(k, k) for k in files}
        suids = {k: stable_uid(study_uid, k) for k in files}
        numbers = {k: MOD_ORDER.get(k, 5) + 1 for k in files}
    if not any(k != "seg" for k in files):
        return None
    if not info["description"]:
        info["description"] = f"{(manifest or {}).get('modality') or 'MR'} — " + ", ".join(
            titles.get(k, k) for k in sorted(files, key=lambda k: numbers.get(k, 9)))
    sig = _signature(files.values())
    with tx() as c:
        st = c.execute("SELECT id, folder, signature FROM studies WHERE study_uid=?", (study_uid,)).fetchone()
        if st and st["folder"] != str(folder) and Path(st["folder"]).exists() and source == "dataset":
            # same case name in two dataset roots: keep both, key the second by path
            study_uid = stable_uid("study-path", str(folder))
            st = c.execute("SELECT id, folder, signature FROM studies WHERE study_uid=?", (study_uid,)).fetchone()
        if st and st["signature"] == sig and st["folder"] == str(folder):
            c.execute("UPDATE studies SET missing=0 WHERE id=?", (st["id"],))
            return st["id"], False
        ppk = upsert_patient(c, info["patient_id"],
                             (manifest or {}).get("patient_name", ""), (manifest or {}).get("sex", ""),
                             (manifest or {}).get("birth_date", ""))
        size = sum(p.stat().st_size for p in files.values() if p.exists())
        vals = dict(patient_pk=ppk, source=source, folder=str(folder), dataset_root=str(root),
                    collection=info["collection"], label_scheme=info["label_scheme"], timepoint=info["timepoint"],
                    signature=sig, has_seg=int("seg" in files), num_series=len(files), size_bytes=size, missing=0,
                    modality=(manifest or {}).get("modality") or "MR")
        if st:
            c.execute(f"UPDATE studies SET {', '.join(k + '=?' for k in vals)} WHERE id=?", (*vals.values(), st["id"]))
            pk, new = st["id"], False
        else:
            vals.update(study_uid=study_uid, description=info["description"], indexed_at=time.time(),
                        study_date=(manifest or {}).get("study_date", ""), accession=(manifest or {}).get("accession", ""),
                        seg_source="ground_truth" if "seg" in files else "")
            cur = c.execute(f"INSERT INTO studies({', '.join(vals)}) VALUES({', '.join('?' * len(vals))})", tuple(vals.values()))
            pk, new = cur.lastrowid, True
        c.execute("DELETE FROM series WHERE study_pk=?", (pk,))
        for key, p in sorted(files.items(), key=lambda kv: numbers.get(kv[0], 9)):
            try:
                dims, spacing, dtype = _header_info(p)
            except Exception:
                dims, spacing, dtype = "", "", "unreadable"
            c.execute("INSERT INTO series(study_pk, series_uid, key, description, file, number, dims, spacing, dtype, size_bytes)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (pk, suids[key], key, titles.get(key, key), str(p), numbers.get(key, 0), dims, spacing, dtype,
                       p.stat().st_size if p.exists() else 0))
    return pk, new


SCAN = {"running": False, "done": 0, "total": 0, "new": 0, "updated": 0, "missing": 0, "errors": [],
        "started": None, "finished": None}
_scan_lock = threading.Lock()


def dataset_roots():
    extra = get_setting("extra_dataset_roots", []) or []
    seen, out = set(), []
    for r in list(DATASET_ROOTS) + list(extra):
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _scan_worker():
    try:
        roots = dataset_roots()
        cands = []
        for root in roots:
            rp = Path(root)
            if not rp.is_dir():
                SCAN["errors"].append(f"dataset folder not found: {root}")
                continue
            for dirpath, dirnames, filenames in os.walk(rp):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                mods = {classify_file(f) for f in filenames if is_nifti(f)}
                if mods & {"t1", "t1ce", "t2", "flair"}:
                    cands.append((Path(dirpath), root))
        SCAN["total"] = len(cands)
        seen = set()
        for folder, root in cands:
            try:
                r = index_study(folder, source="dataset", root=root)
                if r:
                    seen.add(str(folder))
                    if r[1]:
                        SCAN["new"] += 1
            except Exception as e:
                SCAN["errors"].append(f"{folder.name}: {e}")
            SCAN["done"] += 1
        # dataset studies whose folder disappeared stay in the archive (reports!) but are flagged
        for st in qa("SELECT id, folder FROM studies WHERE source='dataset' AND missing=0"):
            if st["folder"] not in seen and not Path(st["folder"]).exists():
                db().execute("UPDATE studies SET missing=1 WHERE id=?", (st["id"],))
                SCAN["missing"] += 1
        audit("dataset_scan", detail=f"{SCAN['total']} case folders, {SCAN['new']} new, {SCAN['missing']} missing",
              username="system")
        threading.Thread(target=_enrich_tumor_volumes, daemon=True, name="tumor-volumes").start()
    except Exception as e:
        SCAN["errors"].append(str(e))
        traceback.print_exc()
    finally:
        SCAN["running"] = False
        SCAN["finished"] = time.time()


def _enrich_tumor_volumes():
    """Fill the worklist's tumour-volume column from dataset ground-truth segmentations.
    Runs after a scan, one study at a time with short pauses, so it never competes
    with the viewer for the CPU."""
    for st in qa("SELECT s.id, s.label_scheme, x.file FROM studies s JOIN series x ON x.study_pk=s.id AND x.key='seg'"
                 " WHERE s.tumor_ml IS NULL AND s.hidden=0 AND s.missing=0"):
        t0 = time.time()
        try:
            img = nib.load(st["file"])
            seg = remap_labels(np.asarray(img.dataobj), st["label_scheme"])
            vv = float(np.prod(img.header.get_zooms()[:3]))
            ml = round(float(np.isin(seg, TUMOR_LABELS).sum()) * vv / 1000, 2)
            db().execute("UPDATE studies SET tumor_ml=? WHERE id=? AND tumor_ml IS NULL", (ml, st["id"]))
        except Exception as e:
            print(f"[enrich] study {st['id']}: {e}")
        time.sleep(max(0.02, time.time() - t0))           # use at most ~half of one CPU core


def start_scan():
    with _scan_lock:
        if SCAN["running"]:
            return False
        SCAN.update(running=True, done=0, total=0, new=0, updated=0, missing=0, errors=[],
                    started=time.time(), finished=None)
    threading.Thread(target=_scan_worker, daemon=True, name="dataset-scan").start()
    return True


# ======================================================================================
# CASE LOADING (in-memory LRU of opened studies)
# ======================================================================================
CASES = OrderedDict()            # case_id ("s<study pk>") -> case dict
_case_lock = threading.RLock()
_loading = {}                    # case_id -> Event while a load is in progress


def case_id_for(pk):
    return f"s{int(pk)}"


def remap_labels(seg: np.ndarray, scheme: str) -> np.ndarray:
    """Normalise to the internal BraTS-2021 scheme (1 NCR, 2 ED, 4 ET, 5 RC).
    BraTS 2023+ files use 1 NETC, 2 SNFH, 3 ET (and 4 = resection cavity in 2024)."""
    seg = seg.astype(np.uint8, copy=True)
    if scheme == "brats2023" or bool((seg == 3).any()):
        rc, et = seg == 4, seg == 3
        if scheme == "brats2023":
            seg[rc] = 5
        seg[et] = 4
    return seg


def _load_nifti(path, ref_shape=None, ref_affine=None, order=1, dtype=np.float32):
    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(dtype=np.float32)) if dtype == np.float32 else np.asarray(img.dataobj)
    if arr.ndim > 3:
        arr = arr[..., 0]
    if ref_shape is not None and (tuple(arr.shape) != tuple(ref_shape) or not np.allclose(img.affine, ref_affine, atol=1e-3)):
        from scipy import ndimage as ndi
        M = np.linalg.inv(img.affine) @ ref_affine
        arr = ndi.affine_transform(arr.astype(np.float32), M[:3, :3], offset=M[:3, 3], output_shape=tuple(ref_shape),
                                   order=order, mode="constant", cval=0)
    return arr.astype(dtype) if arr.dtype != dtype else arr, img


def load_study_case(pk):
    st = q1("SELECT * FROM studies WHERE id=?", (pk,))
    if not st:
        raise KeyError(f"unknown study {pk}")
    series = qa("SELECT * FROM series WHERE study_pk=? ORDER BY number", (pk,))
    imgs = [s for s in series if s["key"] != "seg"]
    if not imgs:
        raise FileNotFoundError("study has no image series")
    vols, src_dtypes, affine, zooms, shape = {}, {}, None, None, None
    for s in imgs:
        p = Path(s["file"])
        if not p.exists():
            raise FileNotFoundError(f"series file missing: {p}")
        arr, img = _load_nifti(p, shape, affine)
        if affine is None:
            affine, zooms, shape = img.affine, [float(z) for z in img.header.get_zooms()[:3]], list(arr.shape)
        vols[s["key"]] = arr
        src_dtypes[s["key"]] = str(img.get_data_dtype())
    gt = None
    sseg = next((s for s in series if s["key"] == "seg"), None)
    if sseg and Path(sseg["file"]).exists():
        g_arr, _ = _load_nifti(sseg["file"], shape, affine, order=0, dtype=np.float32)
        gt = remap_labels(np.rint(g_arr), st["label_scheme"])
    seg, seg_source = gt, ("ground_truth" if gt is not None else None)
    if st["seg_source"] and st["seg_source"] != "ground_truth":
        dp = DERIVED_DIR / dicom_io._safe(st["study_uid"]) / "seg_active.nii.gz"
        if dp.exists():
            d_arr, _ = _load_nifti(dp, shape, affine, order=0)
            seg, seg_source = np.rint(d_arr).astype(np.uint8), st["seg_source"]
    case = {"path": st["folder"], "vols": vols, "src_dtypes": src_dtypes, "affine": affine, "zooms": zooms,
            "shape": shape, "seg": seg, "seg_ground_truth": gt, "seg_source": seg_source, "restored": {},
            "loaded_at": time.time(), "study_pk": pk, "study_uid": st["study_uid"]}
    if gt is not None and st["tumor_ml"] is None:
        stats = compute_tumor_stats(case)
        db().execute("UPDATE studies SET tumor_ml=? WHERE id=?", (stats["total_volume_ml"], pk))
    return case


def get_case(case_id):
    """Return a loaded case; evicted / never-loaded studies are (re)loaded on demand."""
    with _case_lock:
        if case_id in CASES:
            CASES.move_to_end(case_id)
            return CASES[case_id]
        ev = _loading.get(case_id)
        owner = ev is None
        if owner:
            ev = _loading[case_id] = threading.Event()
    if not owner:
        ev.wait(120)
        with _case_lock:
            if case_id in CASES:
                return CASES[case_id]
        raise KeyError(f"case {case_id} failed to load")
    try:
        m = re.fullmatch(r"s(\d+)", str(case_id))
        if not m:
            raise KeyError(f"Unknown case_id: {case_id}")
        case = load_study_case(int(m.group(1)))
        with _case_lock:
            CASES[case_id] = case
            CASES.move_to_end(case_id)
            while len(CASES) > CASE_CACHE_MAX:
                CASES.popitem(last=False)
        return case
    finally:
        with _case_lock:
            _loading.pop(case_id, None)
        ev.set()


# ======================================================================================
# 2D SLICING, WINDOWING, OVERLAYS
# ======================================================================================
PLANE_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}


def extract_slice(vol, plane, index):
    axis = PLANE_AXIS[plane]
    index = int(max(0, min(vol.shape[axis] - 1, index)))
    if axis == 2:
        sl = vol[:, :, index]
    elif axis == 1:
        sl = vol[:, index, :]
    else:
        sl = vol[index, :, :]
    return np.rot90(sl)


def plane_extent(shape, zooms, plane):
    sx, sy, sz = shape
    zx, zy, zz = zooms
    if plane == "axial":
        return sy, sx, zy, zx
    if plane == "coronal":
        return sz, sx, zz, zx
    return sz, sy, zz, zy


def window_to_uint8(sl, wc, ww):
    lo = wc - ww / 2.0
    return (np.clip((sl - lo) / max(ww, 1e-6), 0, 1) * 255).astype(np.uint8)


def default_window(vol):
    nz = vol[vol > 0] if (vol > 0).any() else vol[vol != 0]
    if nz.size == 0:
        return 0.0, 1.0
    if nz.size > 400_000:
        nz = nz[:: nz.size // 400_000]
    lo, hi = np.percentile(nz, [1, 99])
    return float((lo + hi) / 2.0), float(max(hi - lo, 1.0))


def physically_scaled(img, row_spacing, col_spacing, target=512):
    h, w = img.shape[:2]
    aspect = (h * row_spacing) / max(w * col_spacing, 1e-6)
    if aspect >= 1:
        out_h, out_w = target, max(1, int(round(target / aspect)))
    else:
        out_w, out_h = target, max(1, int(round(target * aspect)))
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def seg_overlay_rgba(mask_slice, opacity=0.45):
    """Tumour labels in one uniform red (BGR order, matching the base image)."""
    h, w = mask_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    m = np.isin(mask_slice, TUMOR_LABELS)
    if m.any():
        rgba[m, 0], rgba[m, 1], rgba[m, 2] = TUMOR_OVERLAY_COLOR_BGR
        rgba[m, 3] = int(255 * opacity)
    rc = mask_slice == 5
    if rc.any():                                                   # resection cavity: blue
        rgba[rc, 0], rgba[rc, 1], rgba[rc, 2], rgba[rc, 3] = 255, 140, 60, int(255 * opacity)
    return rgba


def encode_png(img):
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


# ======================================================================================
# SEGMENTATION, VOLUMETRICS, RESTORATION
# ======================================================================================
_MODEL, _MODEL_TRIED = None, False


def try_load_model():
    global _MODEL, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    try:
        import torch
        from monai.networks.nets import SegResNet
        if not MODEL_PATH or not Path(MODEL_PATH).exists():
            print(f"[model] checkpoint not found ({MODEL_PATH!r}) — using auto-threshold segmentation.")
            return None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        model = SegResNet(spatial_dims=3, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16,
                          in_channels=4, out_channels=3, dropout_prob=0.2)
        sd = None
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model", "network", "module"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    sd = ckpt[key]
                    break
            if sd is None and any(hasattr(v, "shape") for v in ckpt.values()):
                sd = ckpt
        if sd is None:
            return None
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        _MODEL = {"model": model, "device": device, "torch": torch}
        print("[model] MONAI SegResNet loaded.")
    except Exception as e:
        print(f"[model] load failed ({e}) — using auto-threshold segmentation.")
        _MODEL = None
    return _MODEL


def largest_component_3d(mask_u8):
    from scipy import ndimage
    labeled, n = ndimage.label(mask_u8)
    if n == 0:
        return mask_u8
    sizes = ndimage.sum(mask_u8, labeled, range(1, n + 1))
    return (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)


def auto_segment(case):
    """Heuristic (Otsu on FLAIR / T1CE inside the brain) — NOT a clinical algorithm."""
    vols = case["vols"]
    flair = vols.get("flair", next(iter(vols.values())))
    t1ce = vols.get("t1ce")
    brain = (flair > np.percentile(flair[flair > 0], 5)) if (flair > 0).any() else flair > 0
    brain = brain.astype(np.uint8)
    seg = np.zeros(flair.shape, dtype=np.uint8)

    def otsu_mask(vol, base):
        vals = vol[base > 0]
        if vals.size < 100:
            return np.zeros(vol.shape, dtype=np.uint8)
        vmin, vmax = np.percentile(vals, [1, 99.5])
        norm = (np.clip((vol - vmin) / max(vmax - vmin, 1e-6), 0, 1) * 255).astype(np.uint8)
        thr, _ = cv2.threshold(norm[base > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ((norm > thr) & (base > 0)).astype(np.uint8)

    edema = otsu_mask(flair, brain)
    enhancing = (otsu_mask(t1ce, brain) & edema) if t1ce is not None else np.zeros_like(edema)
    kernel = np.ones((3, 3), np.uint8)
    ec, nc = np.zeros_like(edema), np.zeros_like(enhancing)
    for z in range(edema.shape[2]):
        e = cv2.morphologyEx(edema[:, :, z] * 255, cv2.MORPH_OPEN, kernel)
        ec[:, :, z] = cv2.morphologyEx(e, cv2.MORPH_CLOSE, kernel) > 0
        if enhancing.any():
            nc[:, :, z] = cv2.morphologyEx(enhancing[:, :, z] * 255, cv2.MORPH_OPEN, kernel) > 0
    ec = largest_component_3d(ec)
    nc = (nc.astype(bool) & ec.astype(bool)).astype(np.uint8)
    seg[ec > 0] = 2
    seg[nc > 0] = 4
    if t1ce is not None and nc.any():
        vals = t1ce[nc > 0]
        if vals.size > 50:
            seg[(t1ce <= np.percentile(vals, 30)) & (nc > 0)] = 1
    return seg, "auto-threshold (heuristic, non-clinical)"


def run_model_segment(case):
    m = try_load_model()
    if m is None:
        return None
    torch = m["torch"]
    order = ["flair", "t1", "t1ce", "t2"]
    if not all(k in case["vols"] for k in order):
        return None
    stacked = np.stack([case["vols"][k] for k in order], axis=0).astype(np.float32)
    for ch in range(stacked.shape[0]):
        a = stacked[ch]
        mu, sd = a.mean(), a.std()
        stacked[ch] = (a - mu) / (sd + 1e-8) if sd > 1e-6 else (a - mu)
    inp = torch.from_numpy(np.expand_dims(np.transpose(stacked, (0, 3, 1, 2)), 0)).float().to(m["device"])
    from monai.inferers import sliding_window_inference
    with torch.no_grad():
        roi = (min(64, inp.shape[2]), min(128, inp.shape[3]), min(128, inp.shape[4]))
        pred = sliding_window_inference(inp, roi, sw_batch_size=1, predictor=m["model"])
    lab = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    return np.transpose(lab, (1, 2, 0)).astype(np.uint8)


def compute_tumor_stats(case):
    seg = case.get("seg")
    if seg is None:
        return {"labels": {}, "total_voxels": 0, "total_volume_ml": 0.0}
    zooms = case["zooms"]
    vv = float(np.prod(zooms))
    labels = {}
    counts = np.bincount(seg.ravel(), minlength=6)
    for lbl in range(1, len(counts)):
        if counts[lbl] == 0:
            continue
        coords = np.argwhere(seg == lbl)
        phys = coords * np.array(zooms)
        diag = float(np.linalg.norm(phys.max(0) - phys.min(0))) if len(coords) > 1 else 0.0
        labels[int(lbl)] = {"name": LABEL_NAMES.get(int(lbl), f"Label {lbl}"), "voxels": int(counts[lbl]),
                            "volume_mm3": round(counts[lbl] * vv, 1), "volume_ml": round(counts[lbl] * vv / 1000, 2),
                            "bbox_diag_mm": round(diag, 1)}
    tumor = np.isin(seg, TUMOR_LABELS)
    total = int(tumor.sum())
    out = {"labels": labels, "total_voxels": total, "total_volume_ml": round(total * vv / 1000, 2)}
    if total:
        coords = np.argwhere(tumor)
        out["bbox_min_vox"] = [int(v) for v in coords.min(0)]
        out["bbox_max_vox"] = [int(v) for v in coords.max(0)]
        out["centroid_vox"] = [round(float(v), 1) for v in coords.mean(0)]
    return out


def build_restored_volume(case, modality):
    vol = case["vols"][modality]
    seg = case.get("seg")
    if seg is None:
        raise ValueError("Run segmentation before requesting brain restoration.")
    mask3d = np.isin(seg, TUMOR_LABELS).astype(np.uint8)
    vmin, vmax = np.percentile(vol[vol > 0], [0.5, 99.5]) if (vol > 0).any() else (0, 1)
    rng = max(vmax - vmin, 1e-6)
    restored = vol.copy()
    for z in range(vol.shape[2]):
        m = mask3d[:, :, z]
        if not m.any():
            continue
        u8 = (np.clip((vol[:, :, z] - vmin) / rng, 0, 1) * 255).astype(np.uint8)
        restored[:, :, z] = cv2.inpaint(u8, m * 255, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0 * rng + vmin
    case["restored"][modality] = restored
    return restored


def array_to_nii_bytes(arr, affine, dtype=np.float32):
    img = nib.Nifti1Image(np.ascontiguousarray(arr.astype(dtype)), affine)
    img.header.set_data_dtype(dtype)
    return img.to_bytes()


def half_res(vol, affine, is_label=False):
    """2x2x2 block reduction for the 3D view on PCs without a graphics card: the ray caster
    takes one sample per voxel, so half the voxels per axis is about half the work per pixel.
    Intensities are block-averaged; labels keep the highest label covering >= 1/4 of a block."""
    sx, sy, sz = (s // 2 * 2 for s in vol.shape)
    b = vol[:sx, :sy, :sz].reshape(sx // 2, 2, sy // 2, 2, sz // 2, 2)
    if is_label:
        frac = (b > 0).mean(axis=(1, 3, 5))
        out = np.where(frac >= 0.25, b.max(axis=(1, 3, 5)), 0).astype(np.uint8)
    else:
        out = b.mean(axis=(1, 3, 5), dtype=np.float32)
    A = affine.copy()
    A[:3, 3] = affine[:3, 3] + affine[:3, :3] @ np.array([0.5, 0.5, 0.5])
    A[:3, :3] = affine[:3, :3] * 2
    return out, A


def volume_send_dtype(case, key, vol):
    """int16 when the source was integer (BraTS is int16): half the bytes of float32,
    a quarter less GPU memory, identical values."""
    src = case.get("src_dtypes", {}).get(key, "float32")
    if src.startswith(("int", "uint")) and vol.min() >= -32768 and vol.max() <= 32767:
        return np.int16
    return np.float32


# ======================================================================================
# THUMBNAILS
# ======================================================================================
_thumb_sem = threading.Semaphore(2)
PLACEHOLDER_PNG = encode_png(np.full((8, 8, 3), 24, np.uint8))


def thumb_file(study_uid):
    return CACHE_DIR / "thumbs" / f"{dicom_io._safe(study_uid)}.jpg"


def make_thumbnail(pk):
    st = q1("SELECT * FROM studies WHERE id=?", (pk,))
    series = {s["key"]: s for s in qa("SELECT * FROM series WHERE study_pk=?", (pk,))}
    key = next((k for k in ("flair", "t2", "t1ce", "t1", "ct") if k in series), None) or \
        next((k for k in series if k != "seg"), None)
    if not key:
        raise FileNotFoundError("no image series")
    img = nib.load(series[key]["file"])
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    seg = None
    if "seg" in series and Path(series["seg"]["file"]).exists():
        s_arr = np.asarray(nib.load(series["seg"]["file"]).dataobj)
        if s_arr.shape == vol.shape:
            seg = remap_labels(s_arr, st["label_scheme"])
            if st["tumor_ml"] is None:
                vv = float(np.prod(img.header.get_zooms()[:3]))
                db().execute("UPDATE studies SET tumor_ml=? WHERE id=?",
                             (round(float(np.isin(seg, TUMOR_LABELS).sum()) * vv / 1000, 2), pk))
    if seg is not None and np.isin(seg, TUMOR_LABELS).any():
        k = int(np.argmax(np.isin(seg, TUMOR_LABELS).sum(axis=(0, 1))))
    else:
        k = int(np.argmax((vol != 0).sum(axis=(0, 1))))
    sl = np.rot90(vol[:, :, k])
    wc, ww = default_window(vol[:, :, max(0, k - 3):k + 4])
    u8 = window_to_uint8(sl, wc, ww)
    zx, zy = [float(z) for z in img.header.get_zooms()[:2]]
    u8 = physically_scaled(u8, zy, zx, target=160)
    bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    if seg is not None:
        m = np.rot90(np.isin(seg[:, :, k], TUMOR_LABELS)).astype(np.uint8) * 255
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, cnts, -1, (40, 40, 255), 1, cv2.LINE_AA)
    canvas = np.zeros((160, 160, 3), np.uint8)
    h, w = bgr.shape[:2]
    y0, x0 = (160 - h) // 2, (160 - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = bgr
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    thumb_file(st["study_uid"]).write_bytes(buf.tobytes())


# ======================================================================================
# HEAD MODEL (atlas-fitted scalp + skull) — background jobs, disk cache
# ======================================================================================
HEAD = {}                        # study pk -> {"state": "running"|"ready"|"error"|"n/a", ...}
_head_lock = threading.Lock()


def head_file(study_uid):
    return CACHE_DIR / "headmodel" / f"{dicom_io._safe(study_uid)}_v1.npz"


def head_status(case):
    """Current state without starting any work (cached file counts as ready)."""
    pk = case["study_pk"]
    cur = HEAD.get(pk)
    if cur:
        return cur
    if head_file(case["study_uid"]).exists():
        return ensure_headmodel(case)
    return {"state": "idle"}


def ensure_headmodel(case):
    pk, uid = case["study_pk"], case["study_uid"]
    with _head_lock:
        cur = HEAD.get(pk)
        if cur and cur["state"] in ("running", "ready", "n/a"):
            return cur
        f = head_file(uid)
        if f.exists():
            try:
                info = json.loads(str(np.load(f)["info"]))
                HEAD[pk] = {"state": "ready", "info": info}
                return HEAD[pk]
            except Exception:
                f.unlink(missing_ok=True)
        if not headmodel.atlas_available():
            HEAD[pk] = {"state": "n/a", "message": "head atlas file assets/head_atlas.npz is missing"}
            return HEAD[pk]
        HEAD[pk] = {"state": "running", "started": time.time()}

    def job():
        try:
            out = headmodel.build_head_meshes(case["vols"], case["affine"])
            (sv, sf), (cv, cf) = out["skull"], out["scalp"]
            tmp = head_file(uid).with_suffix(".tmp.npz")
            np.savez(tmp, skull_v=sv, skull_f=sf, scalp_v=cv, scalp_f=cf, info=json.dumps(out["info"]))
            os.replace(tmp, head_file(uid))
            HEAD[pk] = {"state": "ready", "info": out["info"]}
        except ValueError as e:
            HEAD[pk] = {"state": "n/a", "message": str(e)}
        except Exception as e:
            traceback.print_exc()
            HEAD[pk] = {"state": "error", "message": str(e)}

    threading.Thread(target=job, daemon=True, name=f"headmodel-{pk}").start()
    return HEAD[pk]


# ======================================================================================
# PAGES & AUTH ROUTES
# ======================================================================================
@app.errorhandler(413)
def _too_large(e):
    return jerr(f"Upload too large (limit {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB per request).", 413)


@app.route("/")
def page_worklist():
    return send_from_directory(app.static_folder, "pacs.html")


@app.route("/viewer")
def page_viewer():
    return send_from_directory(app.static_folder, "viewer.html")


@app.route("/login")
@app.route("/setup")
def page_login():
    if REQUIRE_LOGIN and request.path == "/login" and not users_exist():
        return redirect("/setup")
    return send_from_directory(app.static_folder, "login.html")


@app.route("/favicon.ico")
def favicon():
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><circle cx='8' cy='8' r='6' "
           "fill='#ffb000'/></svg>")
    return Response(svg, mimetype="image/svg+xml")


@app.route("/api/me")
def api_me():
    u = current_user() if (not REQUIRE_LOGIN or users_exist()) else None
    return jsonify({"user": u, "require_login": REQUIRE_LOGIN, "setup_required": REQUIRE_LOGIN and not users_exist(),
                    "dicom": dicom_io.HAVE_PYDICOM, "dicom_network": dicom_io.HAVE_PYNETDICOM,
                    "headmodel": headmodel.atlas_available(), "ae_title": DICOM_AE_TITLE})


@app.route("/api/setup", methods=["POST"])
def api_setup():
    if users_exist():
        return jerr("Setup already done — sign in instead.", 409)
    d = request.get_json(force=True)
    username, pw = (d.get("username") or "").strip(), d.get("password") or ""
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,32}", username):
        return jerr("Username: 3-32 letters, digits, dot, dash or underscore.")
    if len(pw) < 8:
        return jerr("Password must be at least 8 characters.")
    with tx() as c:
        cur = c.execute("INSERT INTO users(username, pw_hash, role, full_name, created_at) VALUES(?,?,?,?,?)",
                        (username, generate_password_hash(pw), "admin", d.get("full_name") or username, time.time()))
    session.clear()
    session["uid"] = cur.lastrowid
    session.permanent = True
    audit("setup_admin_created", username=username)
    return jsonify({"ok": True})


@app.route("/api/login", methods=["POST"])
def api_login():
    d = request.get_json(force=True)
    u = q1("SELECT * FROM users WHERE username=?", ((d.get("username") or "").strip(),))
    if not u or not u["active"] or not check_password_hash(u["pw_hash"], d.get("password") or ""):
        time.sleep(0.4)
        audit("login_failed", detail=(d.get("username") or "")[:40], username="-")
        return jerr("Wrong username or password.", 401)
    session.clear()
    session["uid"] = u["id"]
    session.permanent = True
    db().execute("UPDATE users SET last_login=? WHERE id=?", (time.time(), u["id"]))
    audit("login", username=u["username"])
    return jsonify({"ok": True, "user": {k: u[k] for k in ("id", "username", "role", "full_name")}})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    u = current_user()
    if u:
        audit("logout")
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/me/password", methods=["POST"])
def api_my_password():
    u = current_user()
    if not REQUIRE_LOGIN or not u or not u.get("id"):
        return jerr("Not available without login.")
    d = request.get_json(force=True)
    row = q1("SELECT pw_hash FROM users WHERE id=?", (u["id"],))
    if not check_password_hash(row["pw_hash"], d.get("old_password") or ""):
        return jerr("Current password is wrong.", 403)
    if len(d.get("new_password") or "") < 8:
        return jerr("New password must be at least 8 characters.")
    db().execute("UPDATE users SET pw_hash=? WHERE id=?", (generate_password_hash(d["new_password"]), u["id"]))
    audit("password_changed")
    return jsonify({"ok": True})


@app.route("/api/users")
@require_role("admin")
def api_users():
    return jsonify(qa("SELECT id, username, role, full_name, active, created_at, last_login FROM users ORDER BY username"))


@app.route("/api/users/names")
def api_user_names():
    """Who studies can be assigned to (any signed-in user may read this)."""
    return jsonify(qa("SELECT username, full_name, role FROM users WHERE active=1 AND role IN ('admin','radiologist') ORDER BY username"))


@app.route("/api/users", methods=["POST"])
@require_role("admin")
def api_user_create():
    d = request.get_json(force=True)
    username, pw, role = (d.get("username") or "").strip(), d.get("password") or "", d.get("role") or "viewer"
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,32}", username):
        return jerr("Username: 3-32 letters, digits, dot, dash or underscore.")
    if len(pw) < 8:
        return jerr("Password must be at least 8 characters.")
    if role not in ROLES:
        return jerr("Unknown role.")
    try:
        db().execute("INSERT INTO users(username, pw_hash, role, full_name, created_at) VALUES(?,?,?,?,?)",
                     (username, generate_password_hash(pw), role, d.get("full_name") or "", time.time()))
    except sqlite3.IntegrityError:
        return jerr("That username already exists.", 409)
    audit("user_created", detail=f"{username} ({role})")
    return jsonify({"ok": True})


@app.route("/api/users/<int:uid>", methods=["PATCH"])
@require_role("admin")
def api_user_update(uid):
    d = request.get_json(force=True)
    u = q1("SELECT * FROM users WHERE id=?", (uid,))
    if not u:
        return jerr("No such user.", 404)
    upd = {}
    if "role" in d:
        if d["role"] not in ROLES:
            return jerr("Unknown role.")
        upd["role"] = d["role"]
    if "active" in d:
        upd["active"] = 1 if d["active"] else 0
    if "full_name" in d:
        upd["full_name"] = str(d["full_name"])[:80]
    if d.get("password"):
        if len(d["password"]) < 8:
            return jerr("Password must be at least 8 characters.")
        upd["pw_hash"] = generate_password_hash(d["password"])
    me = current_user()
    if uid == me.get("id") and (upd.get("role", "admin") != "admin" or upd.get("active", 1) == 0):
        return jerr("You can't demote or deactivate your own account.")
    if upd:
        db().execute(f"UPDATE users SET {', '.join(k + '=?' for k in upd)} WHERE id=?", (*upd.values(), uid))
    audit("user_updated", detail=f"{u['username']}: {', '.join(k for k in upd if k != 'pw_hash') or ''}"
                                 f"{' password reset' if 'pw_hash' in upd else ''}")
    return jsonify({"ok": True})


# ======================================================================================
# WORKLIST / ARCHIVE API
# ======================================================================================
SORTS = {"imported": "s.indexed_at", "opened": "s.last_opened", "patient": "p.patient_id", "date": "s.study_date",
         "tumor": "s.tumor_ml", "status": "CASE s.status WHEN 'unread' THEN 0 WHEN 'opened' THEN 1 WHEN 'draft' THEN 2 "
                                          "WHEN 'preliminary' THEN 3 ELSE 4 END",
         "priority": "CASE s.priority WHEN 'stat' THEN 0 WHEN 'urgent' THEN 1 ELSE 2 END", "collection": "s.collection"}


def study_row_sql():
    return ("SELECT s.id AS pk, s.study_uid, s.accession, s.study_date, s.description, s.modality, s.source, s.collection,"
            " s.timepoint, s.status, s.priority, s.assigned_to, s.flagged, s.hidden, s.missing, s.has_seg, s.seg_source,"
            " s.tumor_ml, s.num_series, s.size_bytes, s.indexed_at, s.last_opened, s.opened_by,"
            " p.id AS patient_pk, p.patient_id, p.name AS patient_name, p.sex, p.birth_date,"
            " (SELECT group_concat(key, ',') FROM (SELECT key FROM series WHERE study_pk=s.id ORDER BY number)) AS series_keys,"
            " (SELECT status FROM reports r WHERE r.study_pk=s.id ORDER BY version DESC LIMIT 1) AS report_status,"
            " (SELECT count(*) FROM studies s2 WHERE s2.patient_pk=s.patient_pk AND s2.id<>s.id AND s2.hidden=0) AS priors"
            " FROM studies s JOIN patients p ON p.id=s.patient_pk")


@app.route("/api/studies")
def api_studies():
    a = request.args
    where, args = [], []
    u = current_user()
    if not (a.get("hidden") == "1" and u["role"] == "admin"):
        where.append("s.hidden=0")
    if a.get("q"):
        like = f"%{a['q'].strip()}%"
        where.append("(p.patient_id LIKE ? OR p.name LIKE ? OR s.accession LIKE ? OR s.description LIKE ?"
                     " OR s.collection LIKE ? OR s.study_uid LIKE ?)")
        args += [like] * 6
    for f in ("status", "priority", "collection", "source"):
        if a.get(f):
            vals = a[f].split(",")
            where.append(f"s.{f} IN ({','.join('?' * len(vals))})")
            args += vals
    if a.get("has_seg") in ("0", "1"):
        where.append("s.has_seg=?")
        args.append(int(a["has_seg"]))
    if a.get("flagged") == "1":
        where.append("s.flagged=1")
    if a.get("mine") == "1":
        where.append("s.assigned_to=?")
        args.append(u["username"])
    if a.get("patient_pk"):
        where.append("s.patient_pk=?")
        args.append(int(a["patient_pk"]))
    if a.get("date_from"):
        where.append("s.study_date>=?")
        args.append(a["date_from"].replace("-", ""))
    if a.get("date_to"):
        where.append("s.study_date<=?")
        args.append(a["date_to"].replace("-", ""))
    wsql = (" WHERE " + " AND ".join(where)) if where else ""
    sort = SORTS.get(a.get("sort", "imported"), "s.indexed_at")
    direction = "ASC" if a.get("dir") == "asc" else "DESC"
    per = max(10, min(500, int(a.get("per_page", 50))))
    page = max(1, int(a.get("page", 1)))
    total = db().execute(f"SELECT count(*) FROM studies s JOIN patients p ON p.id=s.patient_pk{wsql}", args).fetchone()[0]
    rows = qa(f"{study_row_sql()}{wsql} ORDER BY {sort} {direction} NULLS LAST, p.patient_id ASC LIMIT ? OFFSET ?",
              args + [per, (page - 1) * per])
    facets = {}
    for f in ("status", "priority", "collection"):
        facets[f] = {r[f]: r["n"] for r in qa(f"SELECT {f}, count(*) AS n FROM studies WHERE hidden=0 GROUP BY {f}")}
    facets["flagged"] = db().execute("SELECT count(*) FROM studies WHERE hidden=0 AND flagged=1").fetchone()[0]
    facets["mine"] = db().execute("SELECT count(*) FROM studies WHERE hidden=0 AND assigned_to=?", (u["username"],)).fetchone()[0]
    facets["all"] = db().execute("SELECT count(*) FROM studies WHERE hidden=0").fetchone()[0]
    return jsonify({"rows": rows, "total": total, "page": page, "per_page": per, "facets": facets,
                    "scan": {k: SCAN[k] for k in ("running", "done", "total", "new")}})


def _study_detail(pk):
    st = q1(f"{study_row_sql()} WHERE s.id=?", (pk,))
    if not st:
        return None
    st["folder"] = q1("SELECT folder FROM studies WHERE id=?", (pk,))["folder"]
    st["series"] = qa("SELECT id, series_uid, key, description, file, number, dims, spacing, dtype, size_bytes"
                      " FROM series WHERE study_pk=? ORDER BY number", (pk,))
    for s in st["series"]:
        s["file"] = Path(s["file"]).name
    st["priors_list"] = qa(f"{study_row_sql()} WHERE s.patient_pk=? AND s.id<>? AND s.hidden=0 ORDER BY s.study_date DESC, s.timepoint DESC",
                           (st["patient_pk"], pk))
    st["reports"] = qa("SELECT id, version, status, author, created_at, updated_at, signed_by, signed_at, html_file, parent_id"
                       " FROM reports WHERE study_pk=? ORDER BY version DESC", (pk,))
    st["audit"] = qa("SELECT ts, username, action, detail FROM audit WHERE study_pk=? ORDER BY ts DESC LIMIT 25", (pk,))
    ann = q1("SELECT data, updated_by, updated_at FROM annotations WHERE study_pk=?", (pk,))
    st["annotation_count"] = len(json.loads(ann["data"])) if ann else 0
    st["headmodel"] = HEAD.get(pk, {}).get("state") or ("ready" if head_file(st["study_uid"]).exists() else "")
    return st


@app.route("/api/studies/<int:pk>")
def api_study(pk):
    st = _study_detail(pk)
    return jsonify(st) if st else jerr("No such study.", 404)


@app.route("/api/studies/<int:pk>", methods=["PATCH"])
@require_role("radiologist")
def api_study_update(pk):
    d = request.get_json(force=True)
    st = q1("SELECT * FROM studies WHERE id=?", (pk,))
    if not st:
        return jerr("No such study.", 404)
    upd = {}
    for k in ("priority", "assigned_to", "description", "study_date", "accession"):
        if k in d:
            upd[k] = str(d[k] or "").strip()[:200]
    if "priority" in upd and upd["priority"] not in ("routine", "urgent", "stat"):
        return jerr("priority must be routine, urgent or stat")
    if "study_date" in upd:
        upd["study_date"] = upd["study_date"].replace("-", "")
    if "flagged" in d:
        upd["flagged"] = 1 if d["flagged"] else 0
    if "status" in d:
        if d["status"] not in STATUS_ORDER:
            return jerr("unknown status")
        upd["status"] = d["status"]
    if "hidden" in d:
        if current_user()["role"] != "admin":
            return jerr("only admins can hide studies", 403)
        upd["hidden"] = 1 if d["hidden"] else 0
    if upd:
        db().execute(f"UPDATE studies SET {', '.join(k + '=?' for k in upd)} WHERE id=?", (*upd.values(), pk))
        audit("study_updated", pk, ", ".join(f"{k}={v}" for k, v in upd.items()))
    return jsonify({"ok": True})


@app.route("/api/patients/<int:ppk>", methods=["PATCH"])
@require_role("radiologist")
def api_patient_update(ppk):
    d = request.get_json(force=True)
    upd = {k: str(d[k] or "").strip()[:120] for k in ("name", "sex", "birth_date", "notes") if k in d}
    if "sex" in upd and upd["sex"] not in ("", "M", "F", "O"):
        return jerr("sex must be M, F, O or blank")
    if "birth_date" in upd:
        upd["birth_date"] = upd["birth_date"].replace("-", "")
    if upd:
        db().execute(f"UPDATE patients SET {', '.join(k + '=?' for k in upd)} WHERE id=?", (*upd.values(), ppk))
        audit("patient_updated", None, f"patient #{ppk}: {', '.join(upd)}")
    return jsonify({"ok": True})


@app.route("/api/studies/<int:pk>", methods=["DELETE"])
@require_role("admin")
def api_study_delete(pk):
    st = q1("SELECT * FROM studies WHERE id=?", (pk,))
    if not st:
        return jerr("No such study.", 404)
    if st["source"] == "dataset":
        db().execute("UPDATE studies SET hidden=1 WHERE id=?", (pk,))
        audit("study_hidden", pk, "dataset study removed from worklist (files untouched)")
        return jsonify({"ok": True, "hidden": True})
    folder = Path(st["folder"])
    with tx() as c:
        c.execute("DELETE FROM studies WHERE id=?", (pk,))
        c.execute("DELETE FROM patients WHERE id=? AND NOT EXISTS (SELECT 1 FROM studies WHERE patient_pk=?)",
                  (st["patient_pk"], st["patient_pk"]))
    for p in (folder, DICOM_STORE / dicom_io._safe(st["study_uid"]), DERIVED_DIR / dicom_io._safe(st["study_uid"])):
        if p.exists() and (ARCHIVE_DIR in p.parents or DICOM_STORE in p.parents or DERIVED_DIR in p.parents):
            shutil.rmtree(p, ignore_errors=True)
    thumb_file(st["study_uid"]).unlink(missing_ok=True)
    head_file(st["study_uid"]).unlink(missing_ok=True)
    with _case_lock:
        CASES.pop(case_id_for(pk), None)
    audit("study_deleted", None, f"{st['study_uid']} ({st['source']})")
    return jsonify({"ok": True, "deleted": True})


@app.route("/api/studies/<int:pk>/thumb")
def api_thumb(pk):
    st = q1("SELECT study_uid FROM studies WHERE id=?", (pk,))
    if not st:
        return Response(PLACEHOLDER_PNG, mimetype="image/png")
    f = thumb_file(st["study_uid"])
    if not f.exists():
        with _thumb_sem:
            if not f.exists():
                try:
                    make_thumbnail(pk)
                except Exception as e:
                    print(f"[thumb] study {pk}: {e}")
                    return Response(PLACEHOLDER_PNG, mimetype="image/png")
    resp = send_file(str(f), mimetype="image/jpeg", max_age=3600)
    return resp


@app.route("/api/studies/<int:pk>/open", methods=["POST"])
def api_study_open(pk):
    st = q1("SELECT * FROM studies WHERE id=?", (pk,))
    if not st:
        return jerr("No such study.", 404)
    if st["missing"]:
        return jerr(f"The study folder is missing on disk: {st['folder']}", 410)
    cid = case_id_for(pk)
    try:
        case = get_case(cid)
    except Exception as e:
        traceback.print_exc()
        return jerr(f"Could not load study: {e}", 500)
    u = current_user()
    new_status = "opened" if (st["status"] == "unread" and u["role"] != "viewer") else st["status"]
    db().execute("UPDATE studies SET last_opened=?, opened_by=?, status=? WHERE id=?",
                 (time.time(), u["username"], new_status, pk))
    audit("study_opened", pk)
    windows = {mod: default_window(v) for mod, v in case["vols"].items()}
    ann = q1("SELECT data FROM annotations WHERE study_pk=?", (pk,))
    hm_state = head_status(case)          # fitted on demand, when the 3D view asks for it
    detail = _study_detail(pk)
    return jsonify({
        "case_id": cid, "path": f"{detail['patient_id']} · {detail['description']}",
        "shape": case["shape"], "zooms": case["zooms"], "affine": np.asarray(case["affine"], float).tolist(),
        "modalities": list(case["vols"].keys()), "has_seg": case["seg"] is not None,
        "has_ground_truth": case.get("seg_ground_truth") is not None, "seg_source": case.get("seg_source"),
        "default_windows": {k: {"wc": v[0], "ww": v[1]} for k, v in windows.items()},
        "tumor_stats": compute_tumor_stats(case) if case["seg"] is not None else None,
        "study": detail, "annotations": json.loads(ann["data"]) if ann else [],
        "headmodel": {k: v for k, v in hm_state.items() if k != "started"},
        "user": u,
    })


@app.route("/api/studies/<int:pk>/prefetch", methods=["POST"])
def api_prefetch(pk):
    def job():
        try:
            get_case(case_id_for(pk))
        except Exception as e:
            print(f"[prefetch] {pk}: {e}")
    threading.Thread(target=job, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/studies/<int:pk>/annotations", methods=["GET", "PUT"])
def api_annotations(pk):
    if request.method == "GET":
        r = q1("SELECT data, updated_by, updated_at FROM annotations WHERE study_pk=?", (pk,))
        return jsonify({"measurements": json.loads(r["data"]) if r else [], "updated_by": r["updated_by"] if r else "",
                        "updated_at": r["updated_at"] if r else None})
    if current_user()["role"] == "viewer":
        return jerr("Viewers can't save measurements.", 403)
    data = request.get_json(force=True).get("measurements", [])
    if not isinstance(data, list) or len(data) > 2000:
        return jerr("bad measurements payload")
    db().execute("INSERT INTO annotations(study_pk, data, updated_by, updated_at) VALUES(?,?,?,?)"
                 " ON CONFLICT(study_pk) DO UPDATE SET data=excluded.data, updated_by=excluded.updated_by,"
                 " updated_at=excluded.updated_at", (pk, json.dumps(data), current_user()["username"], time.time()))
    return jsonify({"ok": True, "count": len(data)})


# ======================================================================================
# REPORTS (versioned, signed)
# ======================================================================================
def _report_html_name(st, version, status):
    return f"{dicom_io._safe(st['patient_id'])}_{st['study_uid'][-10:]}_v{version}_{status}.html"


@app.route("/api/studies/<int:pk>/reports")
def api_reports(pk):
    rows = qa("SELECT * FROM reports WHERE study_pk=? ORDER BY version DESC", (pk,))
    for r in rows:
        r["fields"] = json.loads(r["fields"] or "{}")
        r["url"] = f"/reports/{r['html_file']}" if r["html_file"] else ""
    return jsonify(rows)


@app.route("/api/studies/<int:pk>/reports", methods=["POST"])
@require_role("radiologist")
def api_report_save(pk):
    """action: draft | preliminary | final | addendum"""
    d = request.get_json(force=True)
    action = d.get("action", "draft")
    if action not in ("draft", "preliminary", "final", "addendum"):
        return jerr("unknown action")
    st = _study_detail(pk)
    if not st:
        return jerr("No such study.", 404)
    fields = {k: str(v)[:20000] for k, v in (d.get("fields") or {}).items()}
    u = current_user()
    now = time.time()
    latest = q1("SELECT * FROM reports WHERE study_pk=? ORDER BY version DESC LIMIT 1", (pk,))
    if latest and latest["status"] in ("final", "addendum") and action != "addendum":
        return jerr("This study has a final report. Add an addendum instead.", 409)
    if action == "addendum" and not (latest and latest["status"] in ("final", "addendum")):
        return jerr("Addenda can only follow a final report.", 409)
    if action == "addendum" and not fields.get("addendum", "").strip():
        return jerr("Write the addendum text first.")

    # version rules: a draft is edited in place (and can be signed in place); once a version is
    # signed (preliminary / final) it is kept as history and further work starts a new version;
    # a final report is locked — only addenda can follow it
    if latest is None:
        rid, version, status = None, 1, action
    elif latest["status"] == "draft":
        rid, version, status = latest["id"], latest["version"], action
    else:
        rid, version, status = None, latest["version"] + 1, action
    html_file = ""
    signed_by, signed_at = "", None
    if action in ("preliminary", "final", "addendum"):
        signed_by, signed_at = u["username"], now
        stats = compute_tumor_stats(CASES[case_id_for(pk)]) if case_id_for(pk) in CASES and CASES[case_id_for(pk)]["seg"] is not None \
            else ({"labels": {}, "total_voxels": 0, "total_volume_ml": st["tumor_ml"] or 0.0})
        parent = latest if action == "addendum" else None
        html = render_report_html(st, fields, d.get("images", []), d.get("measurements", []), stats,
                                  status=action, version=version, signer=u, signed_at=now,
                                  parent_fields=json.loads(parent["fields"]) if parent else None)
        html_file = _report_html_name(st, version, action)
        (REPORTS_DIR / html_file).write_text(html, encoding="utf-8")
    with tx() as c:
        if rid:
            c.execute("UPDATE reports SET status=?, fields=?, updated_at=?, html_file=COALESCE(NULLIF(?, ''), html_file),"
                      " signed_by=COALESCE(NULLIF(?, ''), signed_by), signed_at=COALESCE(?, signed_at) WHERE id=?",
                      (status, json.dumps(fields), now, html_file, signed_by, signed_at, rid))
        else:
            cur = c.execute("INSERT INTO reports(study_pk, version, status, fields, html_file, author, created_at, updated_at,"
                            " signed_by, signed_at, parent_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (pk, version, status, json.dumps(fields), html_file, u["username"], now, now,
                             signed_by, signed_at, latest["id"] if (latest and action == "addendum") else None))
            rid = cur.lastrowid
        study_status = {"draft": "draft", "preliminary": "preliminary", "final": "final", "addendum": "final"}[status]
        cur_status = c.execute("SELECT status FROM studies WHERE id=?", (pk,)).fetchone()["status"]
        if STATUS_ORDER.index(study_status) > STATUS_ORDER.index(cur_status if cur_status in STATUS_ORDER else "unread"):
            c.execute("UPDATE studies SET status=? WHERE id=?", (study_status, pk))
    audit(f"report_{action}", pk, f"version {version}")
    return jsonify({"ok": True, "id": rid, "version": version, "status": status,
                    "url": f"/reports/{html_file}" if html_file else ""})


@app.route("/reports/<path:fname>")
def serve_report(fname):
    return send_from_directory(REPORTS_DIR, fname)


def esc(s):
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _fmt_date(d):
    d = (d or "").replace("-", "")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else (d or "")


def render_report_html(st, fields, images, measurements, tumor_stats, *, status, version, signer, signed_at,
                       parent_fields=None):
    now = time.strftime("%d %B %Y, %H:%M", time.localtime(signed_at))
    case = CASES.get(case_id_for(st["pk"]))
    shape = case["shape"] if case else ["-", "-", "-"]
    zooms = case["zooms"] if case else [0, 0, 0]
    seg_source = (case.get("seg_source") if case else st.get("seg_source")) or ""
    label_rows = "".join(
        f"<tr><td>{esc(i['name'])}</td><td>{i['voxels']}</td><td>{i['volume_ml']} mL</td><td>{i['bbox_diag_mm']} mm</td></tr>"
        for _, i in sorted(tumor_stats.get("labels", {}).items(), key=lambda kv: int(kv[0]))) or \
        "<tr><td colspan='4' style='text-align:center;color:#888;'>No segmentation computed for this case.</td></tr>"
    meas_rows = "".join(
        f"<tr><td>{esc(m.get('type'))}</td><td>{esc(m.get('value'))}</td><td>{esc(m.get('detail'))}</td>"
        f"<td>{esc(m.get('plane'))}</td><td>{esc(m.get('slice'))}</td></tr>" for m in measurements) or \
        "<tr><td colspan='5' style='text-align:center;color:#888;'>No manual measurements recorded.</td></tr>"
    img_html = "".join(
        f"<div style='display:inline-block;width:{'98%' if im.get('wide') else '48%'};margin:1%;text-align:center;vertical-align:top;'>"
        f"<img src='{im.get('data_url')}' style='max-width:100%;border:1px solid #ccc;border-radius:3px;'>"
        f"<div style='font-size:8.5pt;color:#555;margin-top:3px;'>{esc(im.get('title'))}</div></div>"
        for im in images if str(im.get("data_url", "")).startswith("data:image/"))

    def f(name, default="Not provided", src=None):
        v = (src or fields).get(name)
        return esc(v) if v else f"<span style='color:#888;font-style:italic;'>{default}</span>"

    badge = {"preliminary": ("PRELIMINARY REPORT", "#8a5a00", "#fff3cd"),
             "final": ("FINAL REPORT", "#0b5d1e", "#e3f6e8"),
             "addendum": ("ADDENDUM TO FINAL REPORT", "#1a1a6e", "#e8ecff")}[status]
    addendum_html = ""
    body_fields = fields
    if status == "addendum":
        body_fields = parent_fields or fields
        addendum_html = (f"<div class='section'><div class='section-title'>Addendum (version {version})</div>"
                         f"<div class='field'><div class='fb'>{f('addendum')}</div></div></div>")
    pname = st.get("patient_name") or "Anonymised"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Neuro-Oncology Imaging Report — {esc(st['patient_id'])}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Times New Roman',Georgia,serif;font-size:11pt;color:#111;background:#fff;padding:22mm 18mm;max-width:210mm;margin:0 auto;}}
  h1{{font-size:16pt;text-align:center;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #111;padding-bottom:6px;margin-bottom:4px;}}
  .subtitle{{font-size:9pt;text-align:center;color:#444;margin-bottom:12px;font-style:italic;}}
  .status{{text-align:center;font:700 10pt Arial,sans-serif;letter-spacing:2px;padding:5px;margin-bottom:12px;border-radius:3px;
           color:{badge[1]};background:{badge[2]};border:1px solid {badge[1]};}}
  .banner{{background:#fff3cd;border:1px solid #d4a017;color:#7a4f00;padding:8px 12px;font-size:8.5pt;border-radius:3px;margin-bottom:16px;line-height:1.4;}}
  .section{{margin-bottom:16px;}}
  .section-title{{font-size:10pt;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;border-bottom:1px solid #ccc;padding-bottom:4px;margin-bottom:8px;color:#1a1a6e;}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:4px 20px;font-size:10pt;}}
  .row{{padding:2px 0;}} .lbl{{font-weight:700;color:#333;}}
  .field{{margin-bottom:10px;}}
  .field .fl{{font-size:9pt;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#1a1a6e;margin-bottom:3px;}}
  .field .fb{{border:1px solid #ddd;padding:7px 9px;font-size:10.5pt;line-height:1.6;border-radius:3px;background:#fafafa;white-space:pre-wrap;min-height:20px;}}
  table{{width:100%;border-collapse:collapse;font-size:9.5pt;}}
  table th{{background:#1a1a6e;color:#fff;padding:6px 10px;text-align:left;font-size:9pt;}}
  table td{{padding:5px 10px;border-bottom:1px solid #eee;}}
  table tr:nth-child(even) td{{background:#f9f9f9;}}
  .sig{{border-top:1px solid #111;margin-top:10px;padding-top:4px;font-size:9.5pt;}}
  .foot{{text-align:center;font-size:8pt;color:#888;border-top:1px solid #ccc;padding-top:6px;margin-top:20px;}}
  @media print{{ body{{padding:12mm;}} }}
</style></head>
<body>
<div style="text-align:right;font-size:8pt;color:#888;margin-bottom:10px;">Report v{version} · {esc(status)} · NeuroMPR PACS (prototype)</div>
<h1>Neuro-Oncology Imaging Report</h1>
<div class="subtitle">MRI brain · MPR / 3D volumetric review · Research / development prototype</div>
<div class="status">{badge[0]} — signed by {esc(signer.get('full_name') or signer['username'])} · {now}</div>
<div class="banner">⚠ <strong>Not for clinical use.</strong> Produced by a prototype research PACS. Segmentation may be an unvalidated
heuristic or research model. 3D images that show scalp/skull use an atlas-fitted head model (average CT anatomy fitted to this
brain), not the patient's own skull. All findings must be independently reviewed by a qualified radiologist / neuro-oncologist.</div>

<div class="section">
  <div class="section-title">Patient &amp; Study</div>
  <div class="grid2">
    <div class="row"><span class="lbl">Patient name:</span> {esc(pname)}</div>
    <div class="row"><span class="lbl">Patient ID:</span> {esc(st['patient_id'])}</div>
    <div class="row"><span class="lbl">Sex / DOB:</span> {esc(st.get('sex') or '—')} / {esc(_fmt_date(st.get('birth_date')) or '—')}</div>
    <div class="row"><span class="lbl">Accession:</span> {esc(st.get('accession') or '—')}</div>
    <div class="row"><span class="lbl">Study date:</span> {esc(_fmt_date(st.get('study_date'))) or "<span style='color:#888;font-style:italic;'>not recorded</span>"}</div>
    <div class="row"><span class="lbl">Referring physician:</span> {f('referrer')}</div>
    <div class="row"><span class="lbl">Collection:</span> {esc(st.get('collection'))}</div>
    <div class="row"><span class="lbl">Series:</span> {esc(', '.join(s['description'] for s in st.get('series', [])))}</div>
    <div class="row"><span class="lbl">Volume:</span> {shape[0]} × {shape[1]} × {shape[2]} vox, {float(zooms[0]):.2f} × {float(zooms[1]):.2f} × {float(zooms[2]):.2f} mm</div>
    <div class="row"><span class="lbl">Study UID:</span> <span style="font-size:8.5pt">{esc(st['study_uid'])}</span></div>
  </div>
</div>
{addendum_html}
<div class="section">
  <div class="section-title">Representative Images</div>
  {img_html or "<div style='color:#888;font-style:italic;'>No snapshots attached.</div>"}
</div>
<div class="section">
  <div class="section-title">Structured Report{' (original, unchanged)' if status == 'addendum' else ''}</div>
  <div class="field"><div class="fl">Clinical Indication</div><div class="fb">{f('indication', src=body_fields)}</div></div>
  <div class="field"><div class="fl">Technique</div><div class="fb">{f('technique', 'Multi-parametric MRI brain (T1, T1CE, T2, FLAIR)', src=body_fields)}</div></div>
  <div class="field"><div class="fl">Findings</div><div class="fb">{f('findings', src=body_fields)}</div></div>
  <div class="field"><div class="fl">Impression</div><div class="fb">{f('impression', src=body_fields)}</div></div>
  <div class="field"><div class="fl">Recommendations</div><div class="fb">{f('recommendations', src=body_fields)}</div></div>
</div>
<div class="section">
  <div class="section-title">Tumor Segmentation &amp; Volumetrics ({esc(seg_source) or 'not computed'})</div>
  <table><thead><tr><th>Region</th><th>Voxels</th><th>Volume</th><th>Max extent (bbox diag.)</th></tr></thead>
  <tbody>{label_rows}</tbody></table>
  <div style="margin-top:8px;font-size:10pt;"><strong>Total tumor volume:</strong> {tumor_stats.get('total_volume_ml', 0)} mL</div>
</div>
<div class="section">
  <div class="section-title">Manual Measurements</div>
  <table><thead><tr><th>Type</th><th>Value</th><th>Detail</th><th>Plane</th><th>Slice</th></tr></thead>
  <tbody>{meas_rows}</tbody></table>
</div>
<div class="section">
  <div class="section-title">Authorization</div>
  <div class="grid2">
    <div class="row"><span class="lbl">Reporting physician:</span> {f('radiologist', esc(signer.get('full_name') or signer['username']))}</div>
    <div class="row"><span class="lbl">Electronically signed:</span> {esc(signer['username'])} · {now}</div>
  </div>
  <div class="sig">Signature: ______________________________________ &nbsp;&nbsp; Date: ______________</div>
</div>
<div class="foot">NeuroMPR PACS — Prototype — Research Use Only — NOT FOR CLINICAL USE &nbsp;|&nbsp; {now}</div>
</body></html>"""


# ======================================================================================
# IMPORT (NIfTI upload, DICOM files / zip) & EXPORT
# ======================================================================================
INGEST_LOG = []                                                   # recent DICOM arrivals for the admin panel


def ingest_dicom_study(study_uid, store_folder: Path, *, username="system"):
    """Convert one study's stored DICOM instances to NIfTI and (re)index it."""
    existing = q1("SELECT id, source FROM studies WHERE study_uid=?", (study_uid,))
    if existing and existing["source"] != "dicom":
        msg = f"study {study_uid[-12:]} already archived from {existing['source']}; received copy kept, not re-imported"
        INGEST_LOG.insert(0, {"ts": time.time(), "study_uid": study_uid, "ok": True, "message": msg})
        return existing["id"]
    files = [p for p in Path(store_folder).rglob("*.dcm")]
    headers = dicom_io.scan_dicom_headers(files)
    if study_uid not in headers:
        raise ValueError("no image instances found for this study")
    st = headers[study_uid]
    out = ARCHIVE_DIR / dicom_io._safe(study_uid)
    if out.exists():
        shutil.rmtree(out)
    meta = dict(st["meta"])
    meta.setdefault("collection", "DICOM import")
    manifest = dicom_io.convert_study(st["series"], out, meta)
    pk, new = index_study(out, source="dicom", manifest=manifest)
    if meta.get("study_date"):
        db().execute("UPDATE studies SET study_date=? WHERE id=?", (meta["study_date"], pk))
    thumb_file(study_uid).unlink(missing_ok=True)
    with _case_lock:
        CASES.pop(case_id_for(pk), None)
    msg = (f"{'new' if new else 'updated'} study {meta['patient_id']} — {len(manifest['series'])} series"
           + (f", {len(manifest['skipped'])} skipped" if manifest["skipped"] else ""))
    INGEST_LOG.insert(0, {"ts": time.time(), "study_uid": study_uid, "ok": True, "message": msg})
    del INGEST_LOG[50:]
    audit("dicom_ingested", pk, msg, username=username)
    return pk


@app.route("/api/import/dicom", methods=["POST"])
@require_role("radiologist")
def api_import_dicom():
    if not dicom_io.HAVE_PYDICOM:
        return jerr("DICOM import needs pydicom:  pip install pydicom")
    files = request.files.getlist("files")
    if not files:
        return jerr("No files received.")
    work = TMP_DIR / f"imp_{uuid.uuid4().hex[:10]}"
    work.mkdir(parents=True)
    try:
        saved = []
        for i, f in enumerate(files):
            dest = work / f"{i:05d}_{secure_filename(f.filename or 'file') or 'file'}"
            f.save(str(dest))
            saved.append(dest)
        allf = dicom_io.expand_uploads(saved, work)
        studies = dicom_io.scan_dicom_headers(allf)
        if not studies:
            return jerr("No DICOM images found in the upload (DICOMDIR-only, non-image or non-DICOM files are skipped).")
        results = []
        for suid, st in studies.items():
            dest_root = DICOM_STORE / dicom_io._safe(suid)
            for seuid, paths in st["series"].items():
                sd = dest_root / dicom_io._safe(seuid)
                sd.mkdir(parents=True, exist_ok=True)
                for p in paths:
                    shutil.copyfile(p, sd / f"{hashlib.sha1(p.name.encode()).hexdigest()[:16]}.dcm")
            try:
                pk = ingest_dicom_study(suid, dest_root, username=current_user()["username"])
                results.append({"study_uid": suid, "pk": pk, "ok": True, "patient_id": st["meta"]["patient_id"],
                                "series": len(st["series"])})
            except Exception as e:
                traceback.print_exc()
                results.append({"study_uid": suid, "ok": False, "error": str(e)})
        return jsonify({"ok": any(r["ok"] for r in results), "studies": results})
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.route("/api/import/nifti", methods=["POST"])
@require_role("radiologist")
def api_import_nifti():
    files = [f for f in request.files.getlist("files") if f.filename]
    good = [f for f in files if is_nifti(secure_filename(f.filename) or "")]
    if not good:
        return jerr("No .nii / .nii.gz files received.")
    names = [secure_filename(f.filename) for f in good]
    pid = (request.form.get("patient_id") or "").strip()
    if not pid:
        m = re.match(r"^(BraTS[-_A-Za-z0-9]*?\d+(?:-\d+)?)[-_](?:t1|t2|flair|seg|t1ce|t1c|t1n|t2w|t2f)\b", names[0], re.I)
        pid = m.group(1) if m else Path(names[0]).name.split(".")[0]
    study_uid = stable_uid("upload", pid, uuid.uuid4().hex)
    folder = ARCHIVE_DIR / dicom_io._safe(study_uid)
    folder.mkdir(parents=True)
    series, rejected = [], []
    used = set()
    for f, name in zip(good, names):
        f.save(str(folder / name))
        key = classify_file(name) or re.sub(r"\.nii(\.gz)?$", "", name, flags=re.I).lower()[:24]
        if key in used:
            rejected.append(name)
            (folder / name).unlink()
            continue
        used.add(key)
        series.append({"key": key, "file": name, "description": SERIES_TITLES.get(key, key)})
    if not any(s["key"] != "seg" for s in series):
        shutil.rmtree(folder, ignore_errors=True)
        return jerr("The upload has only a segmentation file — include at least one image series.")
    new_style = any(re.search(r"[-_](t1n|t1c|t2w|t2f)\.nii", n.lower()) for n in names)
    manifest = {"source": "upload", "patient_id": pid, "patient_name": request.form.get("patient_name", ""),
                "study_uid": study_uid, "study_date": (request.form.get("study_date") or "").replace("-", ""),
                "description": request.form.get("description") or f"MRI brain — uploaded ({', '.join(s['key'] for s in series)})",
                "collection": "Upload", "label_scheme": "brats2023" if new_style else "brats2021", "series": series}
    (folder / "study.json").write_text(json.dumps(manifest, indent=1))
    pk, _ = index_study(folder, source="upload", manifest=manifest)
    audit("nifti_uploaded", pk, f"{len(series)} series")
    return jsonify({"ok": True, "pk": pk, "rejected": [f.filename for f in files if f not in good] + rejected})


def _export_study_meta(st, anon):
    meta = {"patient_id": st["patient_id"], "patient_name": st.get("patient_name") or "", "sex": st.get("sex") or "",
            "birth_date": st.get("birth_date") or "", "study_uid": st["study_uid"], "study_date": st.get("study_date") or "",
            "study_time": "", "accession": st.get("accession") or "", "description": st.get("description") or "",
            "referrer": ""}
    if anon:
        h = hashlib.sha1(f"{st['patient_id']}|{app.secret_key}".encode()).hexdigest()[:10].upper()
        meta.update(patient_id=f"ANON-{h}", patient_name="ANONYMOUS", sex="", birth_date="", accession="",
                    study_uid=dicom_io.uid_for("anon", st["study_uid"], app.secret_key), study_date="")
    return meta


def study_dicom_iter(pk, series_uids=None, anon=False):
    """Datasets for a study (optionally only some series). DICOM-sourced studies are served
    from their original instances; NIfTI studies are converted on the fly."""
    st = _study_detail(pk)
    store = DICOM_STORE / dicom_io._safe(st["study_uid"])
    if st["source"] == "dicom" and store.exists() and not anon:
        for sd in sorted(store.iterdir()):
            for f in sorted(sd.glob("*.dcm")):
                ds = dicom_io.pydicom.dcmread(str(f))
                if series_uids and str(ds.SeriesInstanceUID) not in series_uids:
                    continue
                yield ds
        return
    meta = _export_study_meta(st, anon)
    frame = dicom_io.uid_for(meta["study_uid"], "frame-of-reference")
    for s in st["series"]:
        if series_uids and s["series_uid"] not in series_uids:
            continue
        full = q1("SELECT file FROM series WHERE id=?", (s["id"],))["file"]
        img = nib.load(full)
        vol = np.asarray(img.dataobj) if s["key"] == "seg" else np.asarray(img.get_fdata(dtype=np.float32))
        if s["key"] == "seg":
            vol = remap_labels(vol, q1("SELECT label_scheme FROM studies WHERE id=?", (pk,))["label_scheme"])
        suid = dicom_io.uid_for("anon", s["series_uid"], app.secret_key) if anon else s["series_uid"]
        yield from dicom_io.volume_to_datasets(vol, img.affine, study=meta, series={
            "key": s["key"], "description": s["description"], "number": s["number"] or 1, "modality": st["modality"] or "MR",
            "series_uid": suid, "frame_uid": frame})


def _instances_in(pk, series_uids=None):
    n = 0
    for s in qa("SELECT series_uid, dims FROM series WHERE study_pk=?", (pk,)):
        if series_uids and s["series_uid"] not in series_uids:
            continue
        try:
            n += int(s["dims"].split("x")[2])
        except Exception:
            pass
    return n


def _cleanup_tmp(max_age=3600):
    now = time.time()
    for p in TMP_DIR.iterdir():
        try:
            if now - p.stat().st_mtime > max_age:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
        except OSError:
            pass


@app.route("/api/studies/<int:pk>/export")
@require_role("radiologist")
def api_export(pk):
    st = _study_detail(pk)
    if not st:
        return jerr("No such study.", 404)
    fmt = request.args.get("format", "nifti")
    anon = request.args.get("anon") == "1"
    _cleanup_tmp()
    meta = _export_study_meta(st, anon)
    base = f"{dicom_io._safe(meta['patient_id'])}_{fmt}{'_anon' if anon else ''}"
    out = TMP_DIR / f"{base}_{uuid.uuid4().hex[:6]}.zip"
    if fmt == "dicom":
        if not dicom_io.HAVE_PYDICOM:
            return jerr("DICOM export needs pydicom:  pip install pydicom")
        n = dicom_io.write_zip(study_dicom_iter(pk, anon=anon), out)
        detail = f"DICOM zip, {n} instances"
    else:
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
            for s in st["series"]:
                full = q1("SELECT file FROM series WHERE id=?", (s["id"],))["file"]
                ext = ".nii.gz" if full.lower().endswith(".gz") else ".nii"
                z.write(full, f"{meta['patient_id']}/{meta['patient_id']}_{s['key']}{ext}")
            z.writestr(f"{meta['patient_id']}/study.json", json.dumps({
                "patient_id": meta["patient_id"], "study_uid": meta["study_uid"], "description": meta["description"],
                "series": [{"key": s["key"], "description": s["description"]} for s in st["series"]],
                "exported_by": "NeuroMPR PACS", "anonymised": anon}, indent=1))
        detail = f"NIfTI zip, {len(st['series'])} series"
    audit("study_exported", pk, detail + (" (anonymised)" if anon else ""))
    return send_file(str(out), as_attachment=True, download_name=f"{base}.zip", mimetype="application/zip")


# ======================================================================================
# DICOM NETWORK
# ======================================================================================
DICOM_SERVER = None
JOBS = {}


def dicom_find(level, ident):
    def want(k):
        return k in ident

    def val(k):
        v = getattr(ident, k, "")
        return "" if v is None else str(v)

    rows = qa(f"{study_row_sql()} WHERE s.hidden=0 AND s.missing=0")
    out = []
    for r in rows:
        if want("PatientID") and not dicom_io.dicom_wildcard_match(val("PatientID"), r["patient_id"]):
            continue
        if want("PatientName"):
            pat = val("PatientName").replace("^", " ").strip()
            if not dicom_io.dicom_wildcard_match(pat, (r["patient_name"] or "").replace("^", " ")):
                continue
        if want("StudyInstanceUID") and val("StudyInstanceUID") and r["study_uid"] not in val("StudyInstanceUID").split("\\"):
            continue
        if want("StudyDate") and not dicom_io.dicom_date_match(val("StudyDate"), r["study_date"] or ""):
            continue
        if want("AccessionNumber") and not dicom_io.dicom_wildcard_match(val("AccessionNumber"), r["accession"] or ""):
            continue
        if want("StudyDescription") and not dicom_io.dicom_wildcard_match(val("StudyDescription"), r["description"] or ""):
            continue
        out.append(r)
    results = []
    if level == "PATIENT":
        seen = {}
        for r in out:
            seen.setdefault(r["patient_id"], r)
        for r in seen.values():
            ds = dicom_io.Dataset()
            ds.SpecificCharacterSet = "ISO_IR 192"
            ds.QueryRetrieveLevel = "PATIENT"
            ds.PatientID, ds.PatientName = r["patient_id"], (r["patient_name"] or "").replace(" ", "^")
            ds.PatientBirthDate, ds.PatientSex = r["birth_date"] or "", r["sex"] or ""
            ds.NumberOfPatientRelatedStudies = sum(1 for x in out if x["patient_id"] == r["patient_id"])
            ds.RetrieveAETitle = DICOM_AE_TITLE
            results.append(ds)
        return results
    for r in out:
        series = qa("SELECT * FROM series WHERE study_pk=? ORDER BY number", (r["pk"],))
        if level == "SERIES":
            for s in series:
                if want("SeriesInstanceUID") and val("SeriesInstanceUID") and s["series_uid"] not in val("SeriesInstanceUID").split("\\"):
                    continue
                ds = dicom_io.Dataset()
                ds.SpecificCharacterSet = "ISO_IR 192"
                ds.QueryRetrieveLevel = "SERIES"
                ds.StudyInstanceUID, ds.SeriesInstanceUID = r["study_uid"], s["series_uid"]
                ds.Modality, ds.SeriesNumber, ds.SeriesDescription = r["modality"] or "MR", s["number"] or 1, s["description"]
                ds.NumberOfSeriesRelatedInstances = _instances_in(r["pk"], [s["series_uid"]])
                ds.RetrieveAETitle = DICOM_AE_TITLE
                results.append(ds)
            continue
        ds = dicom_io.Dataset()
        ds.SpecificCharacterSet = "ISO_IR 192"
        ds.QueryRetrieveLevel = "STUDY"
        ds.PatientID, ds.PatientName = r["patient_id"], (r["patient_name"] or "").replace(" ", "^")
        ds.PatientBirthDate, ds.PatientSex = r["birth_date"] or "", r["sex"] or ""
        ds.StudyInstanceUID, ds.StudyDate, ds.StudyTime = r["study_uid"], r["study_date"] or "", ""
        ds.AccessionNumber, ds.StudyID = r["accession"] or "", (r["accession"] or "")[:16]
        ds.StudyDescription = (r["description"] or "")[:64]
        ds.ModalitiesInStudy = r["modality"] or "MR"
        ds.NumberOfStudyRelatedSeries = len(series)
        ds.NumberOfStudyRelatedInstances = _instances_in(r["pk"])
        ds.ReferringPhysicianName = ""
        ds.RetrieveAETitle = DICOM_AE_TITLE
        results.append(ds)
    return results


def dicom_retrieve(level, ident):
    suids = [u for u in str(getattr(ident, "StudyInstanceUID", "") or "").split("\\") if u]
    seuids = [u for u in str(getattr(ident, "SeriesInstanceUID", "") or "").split("\\") if u] if level == "SERIES" else None
    pks = []
    if suids:
        for u in suids:
            r = q1("SELECT id FROM studies WHERE study_uid=? AND hidden=0", (u,))
            if r:
                pks.append(r["id"])
    elif level == "PATIENT" and getattr(ident, "PatientID", ""):
        pks = [r["id"] for r in qa("SELECT s.id FROM studies s JOIN patients p ON p.id=s.patient_pk"
                                    " WHERE p.patient_id=? AND s.hidden=0", (str(ident.PatientID),))]
    count = sum(_instances_in(pk, seuids) for pk in pks)

    def gen():
        for pk in pks:
            audit("dicom_retrieved", pk, f"{level} retrieve", username="dicom")
            yield from study_dicom_iter(pk, seuids)
    return count, gen()


def dicom_resolve(ae):
    r = q1("SELECT host, port FROM dicom_nodes WHERE upper(ae_title)=upper(?)", (ae.strip(),))
    return (r["host"], r["port"]) if r else None


_ingest_q = queue.Queue()


def _on_dicom_received(study_uid, folder):
    _ingest_q.put((study_uid, folder))


def _ingest_worker():
    pending = {}
    while True:
        try:
            uid, folder = _ingest_q.get(timeout=1.0)
            pending[uid] = (folder, time.time())
        except queue.Empty:
            pass
        for uid, (folder, t) in list(pending.items()):
            if time.time() - t > 2.0:                                   # quiet for 2 s -> convert
                pending.pop(uid)
                try:
                    ingest_dicom_study(uid, folder, username="dicom")
                except Exception as e:
                    traceback.print_exc()
                    INGEST_LOG.insert(0, {"ts": time.time(), "study_uid": uid, "ok": False, "message": str(e)})


def start_dicom_server():
    global DICOM_SERVER
    if not (DICOM_SCP_ENABLED and dicom_io.HAVE_PYNETDICOM and dicom_io.HAVE_PYDICOM):
        return
    DICOM_SERVER = dicom_io.DicomServer(DICOM_AE_TITLE, DICOM_PORT, DICOM_STORE, _on_dicom_received,
                                        dicom_find, dicom_retrieve, dicom_resolve)
    if DICOM_SERVER.start():
        print(f"[dicom] SCP '{DICOM_AE_TITLE}' listening on port {DICOM_PORT}")
    else:
        print(f"[dicom] SCP not started: {DICOM_SERVER.error}")


@app.route("/api/dicom/status")
def api_dicom_status():
    s = DICOM_SERVER
    return jsonify({"pydicom": dicom_io.HAVE_PYDICOM, "pynetdicom": dicom_io.HAVE_PYNETDICOM,
                    "enabled": DICOM_SCP_ENABLED, "running": bool(s and s.running), "error": s.error if s else None,
                    "ae_title": DICOM_AE_TITLE, "port": DICOM_PORT, "stats": s.stats if s else {},
                    "nodes": qa("SELECT * FROM dicom_nodes ORDER BY name"), "recent": INGEST_LOG[:15]})


@app.route("/api/dicom/nodes", methods=["POST"])
@require_role("admin")
def api_node_add():
    d = request.get_json(force=True)
    ae, host = (d.get("ae_title") or "").strip().upper(), (d.get("host") or "").strip()
    try:
        port = int(d.get("port"))
    except Exception:
        return jerr("port must be a number")
    if not re.fullmatch(r"[A-Z0-9_ -]{1,16}", ae) or not host or not (0 < port < 65536):
        return jerr("AE title (max 16 chars), host and port are required.")
    db().execute("INSERT INTO dicom_nodes(name, ae_title, host, port) VALUES(?,?,?,?)",
                 ((d.get("name") or ae)[:60], ae, host, port))
    audit("dicom_node_added", detail=f"{ae}@{host}:{port}")
    return jsonify({"ok": True})


@app.route("/api/dicom/nodes/<int:nid>", methods=["DELETE"])
@require_role("admin")
def api_node_delete(nid):
    db().execute("DELETE FROM dicom_nodes WHERE id=?", (nid,))
    audit("dicom_node_removed", detail=str(nid))
    return jsonify({"ok": True})


@app.route("/api/dicom/nodes/<int:nid>/echo", methods=["POST"])
def api_node_echo(nid):
    if not dicom_io.HAVE_PYNETDICOM:
        return jerr("needs pynetdicom")
    n = q1("SELECT * FROM dicom_nodes WHERE id=?", (nid,))
    if not n:
        return jerr("No such node.", 404)
    try:
        return jsonify(dicom_io.echo(n["host"], n["port"], n["ae_title"], DICOM_AE_TITLE))
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)})


@app.route("/api/studies/<int:pk>/send", methods=["POST"])
@require_role("radiologist")
def api_send(pk):
    if not dicom_io.HAVE_PYNETDICOM:
        return jerr("DICOM send needs pynetdicom:  pip install pynetdicom")
    d = request.get_json(force=True)
    n = q1("SELECT * FROM dicom_nodes WHERE id=?", (int(d.get("node_id", 0)),))
    if not n:
        return jerr("Pick a DICOM destination (Admin → DICOM to add one).")
    anon = bool(d.get("anon"))
    total = _instances_in(pk)
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"state": "running", "sent": 0, "failed": 0, "total": total, "node": n["name"], "message": ""}
    user = current_user()["username"]

    def job():
        try:
            def prog(s, f):
                JOBS[jid].update(sent=s, failed=f)
            r = dicom_io.send_datasets(study_dicom_iter(pk, anon=anon), n["host"], n["port"], n["ae_title"],
                                       DICOM_AE_TITLE, prog)
            JOBS[jid].update(state="done" if r["ok"] else "error", message=r["message"], sent=r["sent"], failed=r["failed"])
            audit("dicom_sent", pk, f"to {n['ae_title']}@{n['host']}:{n['port']}: {r['message']}", username=user)
        except Exception as e:
            JOBS[jid].update(state="error", message=str(e))
    threading.Thread(target=job, daemon=True).start()
    return jsonify({"ok": True, "job": jid, "total": total})


@app.route("/api/jobs/<jid>")
def api_job(jid):
    return jsonify(JOBS.get(jid) or {"state": "unknown"})


# ======================================================================================
# ADMIN
# ======================================================================================
@app.route("/api/admin/datasets")
@require_role("admin")
def api_datasets():
    extra = get_setting("extra_dataset_roots", []) or []
    roots = [{"path": r, "exists": Path(r).is_dir(), "configured_in_app_py": r in DATASET_ROOTS, "removable": r in extra,
              "studies": db().execute("SELECT count(*) FROM studies WHERE dataset_root=? AND hidden=0", (r,)).fetchone()[0]}
             for r in dataset_roots()]
    return jsonify({"roots": roots, "scan": SCAN})


@app.route("/api/admin/datasets", methods=["POST"])
@require_role("admin")
def api_dataset_add():
    path = (request.get_json(force=True).get("path") or "").strip().strip('"')
    if not path or not Path(path).is_dir():
        return jerr(f"Folder not found on the server: {path}")
    extra = get_setting("extra_dataset_roots", []) or []
    if path not in extra and path not in DATASET_ROOTS:
        extra.append(path)
        set_setting("extra_dataset_roots", extra)
    audit("dataset_root_added", detail=path)
    start_scan()
    return jsonify({"ok": True})


@app.route("/api/admin/datasets", methods=["DELETE"])
@require_role("admin")
def api_dataset_remove():
    path = request.get_json(force=True).get("path")
    extra = [r for r in (get_setting("extra_dataset_roots", []) or []) if r != path]
    set_setting("extra_dataset_roots", extra)
    audit("dataset_root_removed", detail=str(path))
    return jsonify({"ok": True})


@app.route("/api/admin/scan", methods=["GET", "POST"])
def api_scan():
    if request.method == "POST":
        if current_user()["role"] not in ("admin", "radiologist"):
            return jerr("permission denied", 403)
        started = start_scan()
        return jsonify({"ok": True, "started": started, "scan": SCAN})
    return jsonify(SCAN)


def _dir_size(p: Path):
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


@app.route("/api/stats")
def api_stats():
    c = db()
    out = {
        "patients": c.execute("SELECT count(*) FROM patients").fetchone()[0],
        "studies": c.execute("SELECT count(*) FROM studies WHERE hidden=0").fetchone()[0],
        "series": c.execute("SELECT count(*) FROM series s JOIN studies t ON t.id=s.study_pk WHERE t.hidden=0").fetchone()[0],
        "reports": c.execute("SELECT count(*) FROM reports").fetchone()[0],
        "by_status": {r["status"]: r["n"] for r in qa("SELECT status, count(*) n FROM studies WHERE hidden=0 GROUP BY status")},
        "by_source": {r["source"]: r["n"] for r in qa("SELECT source, count(*) n FROM studies WHERE hidden=0 GROUP BY source")},
        "dataset_bytes": c.execute("SELECT coalesce(sum(size_bytes),0) FROM studies WHERE source='dataset' AND hidden=0").fetchone()[0],
        "loaded_cases": list(CASES.keys()), "case_cache_max": CASE_CACHE_MAX,
        "scan": SCAN,
    }
    if request.args.get("disk") == "1":
        out.update(archive_bytes=_dir_size(ARCHIVE_DIR), dicom_store_bytes=_dir_size(DICOM_STORE),
                   cache_bytes=_dir_size(CACHE_DIR), reports_bytes=_dir_size(REPORTS_DIR))
    return jsonify(out)


@app.route("/api/audit")
def api_audit():
    u = current_user()
    a = request.args
    where, args = [], []
    if a.get("study"):
        where.append("a.study_pk=?")
        args.append(int(a["study"]))
    elif u["role"] != "admin":
        return jerr("only admins can read the full audit log", 403)
    if a.get("user"):
        where.append("a.username=?")
        args.append(a["user"])
    if a.get("action"):
        where.append("a.action LIKE ?")
        args.append(f"%{a['action']}%")
    w = (" WHERE " + " AND ".join(where)) if where else ""
    lim = max(1, min(2000, int(a.get("limit", 300))))
    return jsonify(qa("SELECT a.*, p.patient_id FROM audit a LEFT JOIN studies s ON s.id=a.study_pk"
                      f" LEFT JOIN patients p ON p.id=s.patient_pk{w} ORDER BY a.ts DESC LIMIT ?", args + [lim]))


# ======================================================================================
# VIEWER DATA API (case_id = "s<study pk>")
# ======================================================================================
def _case_or_404():
    cid = request.args.get("case_id") or (request.get_json(silent=True) or {}).get("case_id")
    if not cid:
        raise KeyError("case_id missing")
    return get_case(cid)


@app.route("/api/slice")
def api_slice():
    try:
        a = request.args
        case = _case_or_404()
        modality, plane = a.get("modality", "t1"), a.get("plane", "axial")
        index = int(a.get("index", 0))
        wc = float(a["wc"]) if a.get("wc") else None
        ww = float(a["ww"]) if a.get("ww") else None
        overlay = a.get("overlay", "0") == "1"
        opacity = float(a.get("opacity", 0.45))
        target = max(64, min(1024, int(a.get("size", 512))))
        vol = case["restored"][modality] if (a.get("restored") == "1" and modality in case["restored"]) else case["vols"][modality]
        if wc is None or ww is None:
            wc, ww = default_window(vol)
        sl = extract_slice(vol, plane, index)
        _, _, rsp, csp = plane_extent(case["shape"], case["zooms"], plane)
        u8 = physically_scaled(window_to_uint8(sl, wc, ww), rsp, csp, target=target)
        bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        if overlay and case.get("seg") is not None:
            rgba = physically_scaled(seg_overlay_rgba(extract_slice(case["seg"], plane, index), opacity), rsp, csp, target=target)
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            bgr = (bgr.astype(np.float32) * (1 - alpha) + rgba[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        return Response(encode_png(bgr), mimetype="image/png", headers={"Cache-Control": "private, max-age=60"})
    except Exception as e:
        traceback.print_exc()
        return jerr(str(e))


@app.route("/api/probe")
def api_probe():
    case = _case_or_404()
    a = request.args
    plane, index = a.get("plane", "axial"), int(a.get("index", 0))
    row, col = int(a.get("row", 0)), int(a.get("col", 0))
    sl = extract_slice(case["vols"][a.get("modality", "t1")], plane, index)
    val = float(sl[row, col]) if (0 <= row < sl.shape[0] and 0 <= col < sl.shape[1]) else None
    seg_val = None
    if case.get("seg") is not None:
        msl = extract_slice(case["seg"], plane, index)
        if 0 <= row < msl.shape[0] and 0 <= col < msl.shape[1]:
            seg_val = int(msl[row, col])
    return jsonify({"value": val, "label": seg_val, "label_name": LABEL_NAMES.get(seg_val)})


@app.route("/api/roi_stats")
def api_roi_stats():
    case = _case_or_404()
    a = request.args
    plane, index, shape = a.get("plane", "axial"), int(a.get("index", 0)), a.get("shape", "rect")
    r0, c0, r1, c1 = (int(a.get(k, 0)) for k in ("r0", "c0", "r1", "c1"))
    sl = extract_slice(case["vols"][a.get("modality", "t1")], plane, index)
    rows, cols = sl.shape
    minR, maxR = max(0, min(r0, r1)), min(rows - 1, max(r0, r1))
    minC, maxC = max(0, min(c0, c1)), min(cols - 1, max(c0, c1))
    if maxR <= minR or maxC <= minC:
        return jerr("empty ROI")
    sub = sl[minR:maxR + 1, minC:maxC + 1]
    if shape == "ellipse":
        h, w = sub.shape
        yy, xx = np.mgrid[0:h, 0:w]
        vals = sub[((yy - h / 2) / max(h / 2, 1e-6)) ** 2 + ((xx - w / 2) / max(w / 2, 1e-6)) ** 2 <= 1]
    else:
        vals = sub.ravel()
    if vals.size == 0:
        return jerr("empty ROI")
    _, _, rsp, csp = plane_extent(case["shape"], case["zooms"], plane)
    return jsonify({"mean": float(vals.mean()), "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max()),
                    "n": int(vals.size), "area_mm2": round(float(vals.size * rsp * csp), 1)})


@app.route("/api/segment", methods=["POST"])
def api_segment():
    d = request.get_json(force=True)
    case = get_case(d["case_id"])
    method = d.get("method", "auto")
    if current_user()["role"] == "viewer" and method != "ground_truth":
        return jerr("Viewers can't run segmentation.", 403)
    try:
        if method == "ground_truth":
            if case.get("seg_ground_truth") is None:
                return jerr("This study has no ground-truth segmentation file — use Auto-segment or Model segment.")
            seg, used = case["seg_ground_truth"], "ground_truth"
        elif method == "model":
            seg = run_model_segment(case)
            if seg is None:
                seg, note = auto_segment(case)
                used = f"model unavailable -> {note}"
            else:
                used = "MONAI SegResNet"
        else:
            seg, used = auto_segment(case)
        case["seg"], case["seg_source"], case["restored"] = seg, used, {}
        stats = compute_tumor_stats(case)
        uid = case["study_uid"]
        if used != "ground_truth":
            dd = DERIVED_DIR / dicom_io._safe(uid)
            dd.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(seg.astype(np.uint8), case["affine"]), str(dd / "seg_active.nii.gz"))
        db().execute("UPDATE studies SET seg_source=?, has_seg=1, tumor_ml=? WHERE id=?",
                     (used, stats["total_volume_ml"], case["study_pk"]))
        audit("segmentation", case["study_pk"], f"{used}: {stats['total_volume_ml']} mL")
        return jsonify({"ok": True, "method": "ground truth (dataset labels, unmodified)" if used == "ground_truth" else used,
                        "tumor_stats": stats})
    except Exception as e:
        traceback.print_exc()
        return jerr(str(e))


@app.route("/api/nifti_volume/<path:fname>")
@app.route("/api/nifti_volume")
def api_nifti_volume(fname=None):
    """Native-resolution volume as .nii bytes for niivue (the trailing /volume.nii lets niivue
    recognise the format from the URL)."""
    try:
        case = _case_or_404()
        modality, mode = request.args.get("modality", "t1"), request.args.get("mode", "full")
        if mode == "restored":
            if case.get("seg") is None:
                return jerr("Brain restoration needs a tumor mask — run segmentation first.")
            if modality not in case["restored"]:
                build_restored_volume(case, modality)
            vol, dt = case["restored"][modality], np.float32
        else:
            vol = case["vols"][modality]
            dt = volume_send_dtype(case, modality, vol)
        aff = case["affine"]
        if request.args.get("ds") == "2":
            vol, aff = half_res(vol, aff)
            dt = np.float32 if dt == np.float32 else np.int16
        return Response(array_to_nii_bytes(vol, aff, dt), mimetype="application/octet-stream",
                        headers={"Content-Disposition": "inline; filename=volume.nii"})
    except Exception as e:
        traceback.print_exc()
        return jerr(str(e))


@app.route("/api/nifti_mask/<path:fname>")
@app.route("/api/nifti_mask")
def api_nifti_mask(fname=None):
    try:
        case = _case_or_404()
        if case.get("seg") is None:
            return jerr("No segmentation available for this case.")
        seg, aff = case["seg"], case["affine"]
        if request.args.get("ds") == "2":
            seg, aff = half_res(seg, aff, is_label=True)
        return Response(array_to_nii_bytes(seg, aff, np.uint8), mimetype="application/octet-stream",
                        headers={"Content-Disposition": "inline; filename=mask.nii"})
    except Exception as e:
        traceback.print_exc()
        return jerr(str(e))


@app.route("/api/restore", methods=["POST"])
def api_restore():
    d = request.get_json(force=True)
    case = get_case(d["case_id"])
    try:
        build_restored_volume(case, d.get("modality", "t1"))
        return jsonify({"ok": True})
    except Exception as e:
        return jerr(str(e))


@app.route("/api/headmodel/<case_id>")
def api_headmodel_status(case_id):
    try:
        case = get_case(case_id)
    except Exception as e:
        return jerr(str(e), 404)
    s = ensure_headmodel(case)
    return jsonify({k: v for k, v in s.items() if k != "started"})


@app.route("/api/headmodel/<case_id>/<part>.bin")
def api_headmodel_mesh(case_id, part):
    if part not in ("skull", "scalp"):
        return jerr("unknown part", 404)
    m = re.fullmatch(r"s(\d+)", case_id)
    st = q1("SELECT study_uid FROM studies WHERE id=?", (int(m.group(1)),)) if m else None
    f = head_file(st["study_uid"]) if st else None
    if not f or not f.exists():
        return jerr("head model not ready", 409)
    d = np.load(f)
    return Response(headmodel.mesh_to_bytes(d[f"{part}_v"], d[f"{part}_f"]), mimetype="application/octet-stream",
                    headers={"Cache-Control": "private, max-age=86400"})


# ======================================================================================
# STARTUP
# ======================================================================================
_started = False


def startup():
    global _started
    if _started:
        return
    _started = True
    init_db()
    threading.Thread(target=_ingest_worker, daemon=True, name="dicom-ingest").start()
    start_dicom_server()
    if SCAN_ON_STARTUP:
        start_scan()


if __name__ == "__main__":
    startup()
    print(f"\n  NeuroMPR PACS  ->  http://{'127.0.0.1' if HOST in ('0.0.0.0', '') else HOST}:{PORT}")
    print(f"  data folder: {DATA_DIR}")
    print(f"  dataset roots: {', '.join(dataset_roots()) or '(none)'}")
    print(f"  login {'required' if REQUIRE_LOGIN else 'disabled'}; head atlas {'found' if headmodel.atlas_available() else 'MISSING'}"
          f"; DICOM {'on' if DICOM_SERVER and DICOM_SERVER.running else 'off'}\n")
    if OPEN_BROWSER and not os.environ.get("NEUROMPR_NO_BROWSER"):
        import webbrowser
        threading.Timer(1.2, webbrowser.open, args=(f"http://127.0.0.1:{PORT}/",)).start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
