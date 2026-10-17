"""
dicom_io.py — DICOM for the NeuroMPR PACS
=========================================
  * import:  DICOM files / zip  ->  one NIfTI per series (+ study.json manifest),
             so every study, whatever its origin, opens in the same viewer
  * export:  any archived study (BraTS NIfTI included)  ->  standard MR/CT Image
             Storage instances (zip download, network send, C-GET/C-MOVE)
  * network: Storage SCP (C-STORE receive), Verification (C-ECHO), Query
             (C-FIND, Patient/Study root), Retrieve (C-GET, C-MOVE) and Storage
             SCU (send to another PACS / workstation)

pydicom is needed for import/export and pynetdicom for networking; both are
optional — the PACS runs without them and simply hides those features.
"""
from __future__ import annotations

import io
import json
import logging
import re
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian, ImplicitVRLittleEndian
    HAVE_PYDICOM = True
except ImportError:                                    # pragma: no cover
    HAVE_PYDICOM = False

try:
    from pynetdicom import AE, evt, AllStoragePresentationContexts
    from pynetdicom.sop_class import (
        Verification,
        PatientRootQueryRetrieveInformationModelFind, StudyRootQueryRetrieveInformationModelFind,
        PatientRootQueryRetrieveInformationModelGet, StudyRootQueryRetrieveInformationModelGet,
        PatientRootQueryRetrieveInformationModelMove, StudyRootQueryRetrieveInformationModelMove,
    )
    HAVE_PYNETDICOM = True
    logging.getLogger("pynetdicom").setLevel(logging.WARNING)
except ImportError:                                    # pragma: no cover
    HAVE_PYNETDICOM = False

MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
DICOMDIR_CLASS = "1.2.840.10008.1.3.10"
LPS = np.diag([-1.0, -1.0, 1.0, 1.0])                 # RAS <-> LPS (its own inverse)
SERIES_TITLES = {"flair": "FLAIR", "t1": "T1", "t1ce": "T1 post-contrast (T1CE)", "t2": "T2",
                 "seg": "Tumor segmentation labels", "ct": "CT"}
log = logging.getLogger("neurompr.dicom")


def _safe(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "_", str(s))[:80] or "x"


def _str(v) -> str:
    return "" if v is None else str(v).strip()


# ======================================================================================
# IMPORT: DICOM -> NIfTI
# ======================================================================================
def expand_uploads(paths: list[Path], work_dir: Path) -> list[Path]:
    """Unpack any .zip uploads (flattened, no path traversal) and return all files."""
    out = []
    for p in paths:
        if p.suffix.lower() == ".zip" and zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                for i, info in enumerate(z.infolist()):
                    if info.is_dir() or info.file_size == 0:
                        continue
                    dest = work_dir / f"z{i:06d}_{_safe(Path(info.filename).name)}"
                    with z.open(info) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    out.append(dest)
        elif p.is_dir():
            out.extend(f for f in p.rglob("*") if f.is_file())
        else:
            out.append(p)
    return out


def scan_dicom_headers(files: list[Path]) -> dict:
    """{study_uid: {"meta": header-ish dict, "series": {series_uid: [paths]}}}; non-DICOM skipped."""
    studies = {}
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:
            continue
        if _str(getattr(getattr(ds, "file_meta", None), "MediaStorageSOPClassUID", "")) == DICOMDIR_CLASS:
            continue
        if not all(hasattr(ds, k) for k in ("StudyInstanceUID", "SeriesInstanceUID", "Rows", "Columns")):
            continue
        st = studies.setdefault(str(ds.StudyInstanceUID), {"meta": _study_meta(ds), "series": {}})
        st["series"].setdefault(str(ds.SeriesInstanceUID), []).append(f)
    return studies


def _study_meta(ds) -> dict:
    return {
        "patient_id": _str(getattr(ds, "PatientID", "")) or "UNKNOWN",
        "patient_name": _str(getattr(ds, "PatientName", "")).replace("^", " ").strip(),
        "sex": _str(getattr(ds, "PatientSex", "")),
        "birth_date": _str(getattr(ds, "PatientBirthDate", "")),
        "study_uid": _str(ds.StudyInstanceUID),
        "study_date": _str(getattr(ds, "StudyDate", "")),
        "study_time": _str(getattr(ds, "StudyTime", "")),
        "description": _str(getattr(ds, "StudyDescription", "")),
        "accession": _str(getattr(ds, "AccessionNumber", "")),
        "referrer": _str(getattr(ds, "ReferringPhysicianName", "")).replace("^", " ").strip(),
        "institution": _str(getattr(ds, "InstitutionName", "")),
        "modality": _str(getattr(ds, "Modality", "")),
    }


def classify_series(ds) -> str | None:
    text = " ".join(_str(getattr(ds, k, "")) for k in ("SeriesDescription", "ProtocolName", "SequenceName")).lower()
    image_type = [str(x).upper() for x in (getattr(ds, "ImageType", []) or [])]
    if "SEGMENTATION" in image_type or re.search(r"\bseg\b|segmentation|label map|tumou?r labels", text):
        return "seg"
    if _str(getattr(ds, "Modality", "")).upper() == "CT":
        return "ct"
    if "flair" in text or re.search(r"\bt2f\b", text):
        return "flair"
    if re.search(r"t1\w*.*(post|gd|gad|\+c|contrast|\bce\b)|\bt1c\b|t1ce|t1gd|post.?contrast", text):
        return "t1ce"
    if re.search(r"\bt1|t1w|t1n|mprage|spgr|bravo", text):
        return "t1"
    if re.search(r"\bt2|t2w", text):
        return "t2"
    return None


def _rescale(ds, arr):
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr.astype(np.float32)
    if _str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr * slope + inter


def _orth(iop):
    x, y = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    return x / np.linalg.norm(x), y / np.linalg.norm(y)


def _single_frame_volume(dsets):
    imgs = [d for d in dsets if hasattr(d, "ImagePositionPatient") and hasattr(d, "ImageOrientationPatient")]
    if len(imgs) < 2:
        raise ValueError("fewer than 2 positioned slices")
    # keep the dominant orientation / matrix (drops localizers mixed into a series)
    key = lambda d: (tuple(np.round(np.asarray(d.ImageOrientationPatient, float), 3)), int(d.Rows), int(d.Columns))
    groups = {}
    for d in imgs:
        groups.setdefault(key(d), []).append(d)
    imgs = max(groups.values(), key=len)
    X, Y = _orth(imgs[0].ImageOrientationPatient)
    N = np.cross(X, Y)
    pos = {}
    for d in sorted(imgs, key=lambda d: int(getattr(d, "InstanceNumber", 0) or 0)):
        p = round(float(np.dot(np.asarray(d.ImagePositionPatient, float), N)), 3)
        pos.setdefault(p, d)                                          # first echo / temporal position per location
    order = sorted(pos)
    imgs = [pos[p] for p in order]
    if len(imgs) < 2:
        raise ValueError("fewer than 2 distinct slice positions")
    ps = [float(v) for v in imgs[0].PixelSpacing]
    arr = np.stack([_rescale(d, d.pixel_array) for d in imgs], axis=2)  # (rows, cols, n)
    vol = np.ascontiguousarray(arr.transpose(1, 0, 2))                   # (cols, rows, n) = (i, j, k)
    p0 = np.asarray(imgs[0].ImagePositionPatient, float)
    p1 = np.asarray(imgs[-1].ImagePositionPatient, float)
    step = (p1 - p0) / (len(imgs) - 1)
    A = np.eye(4)
    A[:3, 0] = X * ps[1]
    A[:3, 1] = Y * ps[0]
    A[:3, 2] = step
    A[:3, 3] = p0
    gaps = np.diff(order)
    irregular = bool(len(gaps) > 1 and (gaps.max() - gaps.min()) > 0.1 * max(abs(gaps.mean()), 1e-3))
    return vol, LPS @ A, irregular


def _fg(ds, seq, item=0):
    try:
        return getattr(ds, seq)[item]
    except Exception:
        return None


def _multi_frame_volume(ds):
    arr = ds.pixel_array
    if arr.ndim != 3:
        raise ValueError("unsupported multi-frame layout")
    shared = _fg(ds, "SharedFunctionalGroupsSequence")
    per = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per is None:
        raise ValueError("multi-frame without per-frame functional groups")

    def attr(i, seq, name):
        for src in (per[i], shared):
            if src is not None and hasattr(src, seq):
                s = getattr(src, seq)[0]
                if hasattr(s, name):
                    return getattr(s, name)
        return None

    iop = attr(0, "PlaneOrientationSequence", "ImageOrientationPatient")
    ps = attr(0, "PixelMeasuresSequence", "PixelSpacing")
    if iop is None or ps is None:
        raise ValueError("multi-frame without plane geometry")
    X, Y = _orth(iop)
    N = np.cross(X, Y)
    frames = {}
    for i in range(arr.shape[0]):
        ipp = attr(i, "PlanePositionSequence", "ImagePositionPatient")
        if ipp is None:
            continue
        p = round(float(np.dot(np.asarray(ipp, float), N)), 3)
        if p not in frames:
            slope = attr(i, "PixelValueTransformationSequence", "RescaleSlope") or getattr(ds, "RescaleSlope", 1) or 1
            inter = attr(i, "PixelValueTransformationSequence", "RescaleIntercept") or getattr(ds, "RescaleIntercept", 0) or 0
            frames[p] = (np.asarray(ipp, float), arr[i].astype(np.float32) * float(slope) + float(inter))
    order = sorted(frames)
    if len(order) < 2:
        raise ValueError("fewer than 2 distinct frame positions")
    vol = np.ascontiguousarray(np.stack([frames[p][1] for p in order], 2).transpose(1, 0, 2))
    p0, p1 = frames[order[0]][0], frames[order[-1]][0]
    A = np.eye(4)
    A[:3, 0] = X * float(ps[1])
    A[:3, 1] = Y * float(ps[0])
    A[:3, 2] = (p1 - p0) / (len(order) - 1)
    A[:3, 3] = p0
    return vol, LPS @ A, False


def series_to_volume(paths: list[Path]):
    dsets = []
    for p in paths:
        try:
            dsets.append(pydicom.dcmread(str(p)))
        except Exception as e:
            log.warning("unreadable DICOM %s: %s", p, e)
    if not dsets:
        raise ValueError("no readable instances")
    first = dsets[0]
    if int(getattr(first, "SamplesPerPixel", 1) or 1) != 1:
        raise ValueError("colour images are not supported for volume viewing")
    if len(dsets) == 1 and int(getattr(first, "NumberOfFrames", 1) or 1) > 1:
        vol, aff, irr = _multi_frame_volume(first)
    else:
        vol, aff, irr = _single_frame_volume(dsets)
    return vol, aff, irr, first


def _resample_to(vol, aff, ref_shape, ref_aff, order):
    from scipy import ndimage as ndi
    M = np.linalg.inv(aff) @ ref_aff
    return ndi.affine_transform(vol, M[:3, :3], offset=M[:3, 3], output_shape=tuple(ref_shape),
                                order=order, mode="constant", cval=0.0)


def convert_study(series_files: dict, out_dir: Path, meta: dict) -> dict:
    """Convert the series of one study to NIfTI files in out_dir + study.json.
    All series are resampled to one reference grid (the viewer's MPR shares one grid)."""
    import nibabel as nib
    out_dir.mkdir(parents=True, exist_ok=True)
    converted, skipped = [], []
    for suid, paths in series_files.items():
        try:
            vol, aff, irregular, ds = series_to_volume(paths)
        except Exception as e:
            skipped.append({"series_uid": suid, "reason": str(e)})
            continue
        converted.append({"series_uid": suid, "vol": vol, "aff": aff, "ds": ds, "irregular": irregular,
                          "key": classify_series(ds),
                          "description": _str(getattr(ds, "SeriesDescription", "")) or _str(getattr(ds, "ProtocolName", "")),
                          "number": int(getattr(ds, "SeriesNumber", 0) or 0),
                          "modality": _str(getattr(ds, "Modality", ""))})
    if not converted:
        raise ValueError("no convertible image series: " + "; ".join(s["reason"] for s in skipped))
    rank = {"flair": 0, "t1ce": 1, "t1": 2, "t2": 3, "ct": 4, None: 5, "seg": 9}
    converted.sort(key=lambda s: (rank.get(s["key"], 5), -s["vol"].size))
    ref = converted[0]
    used, manifest_series = set(), []
    for n, s in enumerate(converted):
        key = s["key"] or f"series{s['number'] or n + 1}"
        base, k = key, 2
        while key in used:
            key, k = f"{base}_{k}", k + 1
        used.add(key)
        vol = s["vol"]
        if vol.shape != ref["vol"].shape or not np.allclose(s["aff"], ref["aff"], atol=1e-3):
            vol = _resample_to(vol, s["aff"], ref["vol"].shape, ref["aff"], 0 if s["key"] == "seg" else 1)
        if s["key"] == "seg":
            data = np.rint(vol).astype(np.uint8)
        elif np.all(np.mod(vol, 1) == 0) and vol.min() >= -32768 and vol.max() <= 32767:
            data = vol.astype(np.int16)
        else:
            data = vol.astype(np.float32)
        fname = f"{_safe(key)}.nii.gz"
        img = nib.Nifti1Image(data, ref["aff"])
        img.header.set_data_dtype(data.dtype)
        nib.save(img, str(out_dir / fname))
        manifest_series.append({"key": key, "file": fname, "description": s["description"] or SERIES_TITLES.get(key, key),
                                "series_uid": s["series_uid"], "modality": s["modality"], "number": s["number"],
                                "irregular_spacing": s["irregular"]})
    manifest = {"source": "dicom", **meta, "series": manifest_series, "skipped": skipped,
                "converted_at": time.time()}
    (out_dir / "study.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ======================================================================================
# EXPORT: volumes -> DICOM instances
# ======================================================================================
def uid_for(*parts) -> str:
    return generate_uid(entropy_srcs=[str(p) for p in parts])


def _to_stored(vol: np.ndarray, is_seg: bool):
    """int16 pixels + rescale so float data survives the round trip."""
    if is_seg:
        return vol.astype(np.int16), 1.0, 0.0
    vmin, vmax = float(vol.min()), float(vol.max())
    if np.all(np.mod(vol, 1) == 0) and vmin >= -32767 and vmax <= 32767:
        return vol.astype(np.int16), 1.0, 0.0
    slope = (vmax - vmin) / 65000.0 or 1.0
    inter = (vmax + vmin) / 2.0
    return np.rint((vol - inter) / slope).astype(np.int16), slope, inter


def volume_to_datasets(vol, affine, *, study: dict, series: dict):
    """Yield one MR/CT Image Storage dataset per slice (k axis) of a NIfTI-style volume.
    study: patient_id, patient_name, sex, birth_date, study_uid, study_date, study_time,
           accession, description, referrer
    series: key, description, number, modality ('MR'|'CT'), series_uid, frame_uid"""
    is_seg = series["key"] == "seg"
    stored, slope, inter = _to_stored(np.asarray(vol), is_seg)
    A = LPS @ affine
    cx, cy = A[:3, 0], A[:3, 1]
    csp, rsp = float(np.linalg.norm(cx)), float(np.linalg.norm(cy))
    X, Y = cx / csp, cy / rsp
    step = A[:3, 2]
    thick = float(np.linalg.norm(step))
    normal = np.cross(X, Y)
    iop = [round(float(v), 6) for v in (*X, *Y)]
    modality = series.get("modality") or "MR"
    sop_class = CT_IMAGE_STORAGE if modality == "CT" else MR_IMAGE_STORAGE
    if is_seg:
        wc, ww = 2.5, 5.0
    else:
        nz = stored[stored != 0]
        lo, hi = (np.percentile(nz, [1, 99]) if nz.size else (0, 1))
        wc, ww = float((lo + hi) / 2 * slope + inter), float(max((hi - lo) * slope, 1))
    now = time.strftime("%Y%m%d"), time.strftime("%H%M%S")
    for k in range(stored.shape[2]):
        ds = Dataset()
        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = uid_for(series["series_uid"], k)
        ds.SpecificCharacterSet = "ISO_IR 192"
        ds.ImageType = ["DERIVED", "SECONDARY"] + (["SEGMENTATION"] if is_seg else [])
        ds.InstanceCreationDate, ds.InstanceCreationTime = now
        ds.PatientName = (study.get("patient_name") or "").replace(" ", "^") or "ANONYMOUS"
        ds.PatientID = study.get("patient_id") or "UNKNOWN"
        ds.PatientBirthDate = study.get("birth_date") or ""
        ds.PatientSex = study.get("sex") or ""
        ds.StudyInstanceUID = study["study_uid"]
        ds.StudyDate = study.get("study_date") or ""
        ds.StudyTime = study.get("study_time") or ""
        ds.StudyID = (study.get("accession") or "1")[:16]
        ds.AccessionNumber = (study.get("accession") or "")[:16]
        ds.ReferringPhysicianName = study.get("referrer") or ""
        ds.StudyDescription = (study.get("description") or "")[:64]
        ds.SeriesInstanceUID = series["series_uid"]
        ds.SeriesNumber = int(series.get("number") or 1)
        ds.SeriesDescription = (series.get("description") or SERIES_TITLES.get(series["key"], series["key"]))[:64]
        ds.Modality = modality
        ds.Manufacturer = "NeuroMPR PACS"
        ds.BodyPartExamined = "BRAIN"
        ds.DerivationDescription = "Converted from NIfTI by NeuroMPR PACS"
        ds.FrameOfReferenceUID = series["frame_uid"]
        ds.PositionReferenceIndicator = ""
        if modality != "CT":
            ds.ScanningSequence, ds.SequenceVariant, ds.ScanOptions, ds.MRAcquisitionType = "RM", "NONE", "", "3D"
        else:
            ds.KVP = ""
        ds.InstanceNumber = k + 1
        ipp = A @ np.array([0.0, 0.0, float(k), 1.0])
        ds.ImagePositionPatient = [round(float(v), 4) for v in ipp[:3]]
        ds.ImageOrientationPatient = iop
        ds.SliceLocation = round(float(np.dot(ipp[:3], normal)), 4)
        ds.SliceThickness = round(thick, 4)
        ds.SpacingBetweenSlices = round(thick, 4)
        ds.PixelSpacing = [round(rsp, 6), round(csp, 6)]
        px = np.ascontiguousarray(stored[:, :, k].T)                      # rows = j, cols = i
        ds.Rows, ds.Columns = int(px.shape[0]), int(px.shape[1])
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope = round(inter, 6), round(slope, 9)
        ds.RescaleType = "US"
        ds.WindowCenter, ds.WindowWidth = round(wc, 2), round(ww, 2)
        ds.PixelData = px.astype("<i2").tobytes()
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = sop_class
        fm.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = fm
        yield ds


def dataset_bytes(ds) -> bytes:
    buf = io.BytesIO()
    try:
        ds.save_as(buf, enforce_file_format=True)                 # pydicom 3
    except TypeError:                                             # pydicom 2.x
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def write_zip(datasets, zip_path: Path) -> int:
    n = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for ds in datasets:
            folder = f"{_safe(ds.PatientID)}/{_safe(ds.StudyDate or 'study')}_{str(ds.StudyInstanceUID)[-12:]}/" \
                     f"S{int(ds.SeriesNumber):03d}_{_safe(ds.SeriesDescription)}"
            z.writestr(f"{folder}/IM{int(ds.InstanceNumber):05d}.dcm", dataset_bytes(ds))
            n += 1
    return n


# ======================================================================================
# NETWORK
# ======================================================================================
def echo(host: str, port: int, ae_title: str, calling: str = "NEUROMPR", timeout: float = 8.0) -> dict:
    ae = AE(ae_title=calling)
    ae.acse_timeout = ae.network_timeout = timeout
    ae.add_requested_context(Verification)
    t0 = time.time()
    assoc = ae.associate(host, int(port), ae_title=ae_title)
    if not assoc.is_established:
        return {"ok": False, "message": "association rejected or peer unreachable"}
    st = assoc.send_c_echo()
    assoc.release()
    ok = bool(st) and getattr(st, "Status", None) == 0
    return {"ok": ok, "message": "C-ECHO success" if ok else f"C-ECHO status {getattr(st, 'Status', 'none')}",
            "ms": int((time.time() - t0) * 1000)}


def send_datasets(datasets, host: str, port: int, ae_title: str, calling: str = "NEUROMPR",
                  progress=None) -> dict:
    """C-STORE every dataset (an iterable) to a remote node over one association."""
    ae = AE(ae_title=calling)
    ae.acse_timeout = ae.network_timeout = 30
    for sop in (MR_IMAGE_STORAGE, CT_IMAGE_STORAGE):
        ae.add_requested_context(sop, [ExplicitVRLittleEndian, ImplicitVRLittleEndian])
    assoc = ae.associate(host, int(port), ae_title=ae_title)
    if not assoc.is_established:
        return {"ok": False, "sent": 0, "failed": 0, "message": "association rejected or peer unreachable"}
    sent = failed = 0
    try:
        for ds in datasets:
            st = assoc.send_c_store(ds)
            if st and getattr(st, "Status", 0xC000) in (0x0000, 0xB000, 0xB007, 0xB006):
                sent += 1
            else:
                failed += 1
            if progress:
                progress(sent, failed)
    finally:
        assoc.release()
    return {"ok": failed == 0 and sent > 0, "sent": sent, "failed": failed,
            "message": f"sent {sent} instance(s)" + (f", {failed} failed" if failed else "")}


class DicomServer:
    """Storage + Verification + Query/Retrieve SCP, run in background threads.

    callbacks (all supplied by the app):
      on_received(study_uid, folder)          -> called once an association has finished
      find(level, identifier) -> list[Dataset] (matching identifiers, keys filled)
      retrieve(level, identifier) -> (count, iterator of Dataset)
      resolve_ae(ae_title) -> (host, port) or None
    """

    def __init__(self, ae_title, port, store_dir: Path, on_received, find, retrieve, resolve_ae):
        self.ae_title, self.port, self.store_dir = ae_title, int(port), Path(store_dir)
        self.on_received, self.find, self.retrieve, self.resolve_ae = on_received, find, retrieve, resolve_ae
        self.server = None
        self.error = None
        self.stats = {"received": 0, "associations": 0, "find": 0, "get": 0, "move": 0, "echo": 0,
                      "last_activity": None}
        self._touched = {}
        self._lock = threading.Lock()

    # --- handlers ---------------------------------------------------------------------
    def _h_store(self, event):
        try:
            ds = event.dataset
            study, series, sop = str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID), str(ds.SOPInstanceUID)
            folder = self.store_dir / _safe(study) / _safe(series)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{_safe(sop)}.dcm").write_bytes(event.encoded_dataset())
            with self._lock:
                self._touched.setdefault(id(event.assoc), set()).add(study)
                self.stats["received"] += 1
                self.stats["last_activity"] = time.time()
            return 0x0000
        except Exception as e:                                      # out of resources / bad data
            log.exception("C-STORE failed: %s", e)
            return 0xA700

    def _h_assoc_end(self, event):
        with self._lock:
            studies = self._touched.pop(id(event.assoc), set())
        for st in studies:
            try:
                self.on_received(st, self.store_dir / _safe(st))
            except Exception:
                log.exception("ingest callback failed")

    def _h_echo(self, event):
        self.stats["echo"] += 1
        return 0x0000

    def _h_find(self, event):
        self.stats["find"] += 1
        ident = event.identifier
        level = _str(getattr(ident, "QueryRetrieveLevel", "STUDY")).upper()
        try:
            matches = self.find(level, ident)
        except Exception as e:
            log.exception("C-FIND failed")
            yield 0xC000, None
            return
        for m in matches:
            if event.is_cancelled:
                yield 0xFE00, None
                return
            yield 0xFF00, m

    def _h_get(self, event):
        self.stats["get"] += 1
        ident = event.identifier
        level = _str(getattr(ident, "QueryRetrieveLevel", "STUDY")).upper()
        count, it = self.retrieve(level, ident)
        yield count
        for ds in it:
            if event.is_cancelled:
                yield 0xFE00, None
                return
            yield 0xFF00, ds

    def _h_move(self, event):
        self.stats["move"] += 1
        dest = self.resolve_ae(_str(event.move_destination.decode() if isinstance(event.move_destination, bytes)
                                    else event.move_destination))
        if not dest:
            yield None, None                                         # 0xA801 move destination unknown
            return
        yield dest[0], int(dest[1])
        ident = event.identifier
        level = _str(getattr(ident, "QueryRetrieveLevel", "STUDY")).upper()
        count, it = self.retrieve(level, ident)
        yield count
        for ds in it:
            if event.is_cancelled:
                yield 0xFE00, None
                return
            yield 0xFF00, ds

    def _h_open(self, event):
        self.stats["associations"] += 1

    # --- lifecycle ----------------------------------------------------------------------
    def start(self):
        if not HAVE_PYNETDICOM:
            self.error = "pynetdicom not installed"
            return False
        ae = AE(ae_title=self.ae_title)
        ae.maximum_pdu_size = 0
        for cx in AllStoragePresentationContexts:
            # scu_role/scp_role=True also accepts the role-selection a C-GET requestor sends
            ae.add_supported_context(cx.abstract_syntax, cx.transfer_syntax, scu_role=True, scp_role=True)
        for sop in (Verification, PatientRootQueryRetrieveInformationModelFind, StudyRootQueryRetrieveInformationModelFind,
                    PatientRootQueryRetrieveInformationModelGet, StudyRootQueryRetrieveInformationModelGet,
                    PatientRootQueryRetrieveInformationModelMove, StudyRootQueryRetrieveInformationModelMove):
            ae.add_supported_context(sop)
        for sop in (MR_IMAGE_STORAGE, CT_IMAGE_STORAGE):           # outgoing C-STORE for C-MOVE
            ae.add_requested_context(sop, [ExplicitVRLittleEndian, ImplicitVRLittleEndian])
        handlers = [(evt.EVT_C_STORE, self._h_store), (evt.EVT_C_ECHO, self._h_echo),
                    (evt.EVT_C_FIND, self._h_find), (evt.EVT_C_GET, self._h_get), (evt.EVT_C_MOVE, self._h_move),
                    (evt.EVT_RELEASED, self._h_assoc_end), (evt.EVT_ABORTED, self._h_assoc_end),
                    (evt.EVT_ESTABLISHED, self._h_open)]
        try:
            self.server = ae.start_server(("0.0.0.0", self.port), block=False, evt_handlers=handlers)
            self.error = None
            return True
        except OSError as e:
            self.error = f"cannot listen on port {self.port}: {e}"
            return False

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server = None

    @property
    def running(self) -> bool:
        return self.server is not None


# ======================================================================================
# query helpers (used by the app's C-FIND implementation)
# ======================================================================================
def dicom_wildcard_match(pattern: str, value: str) -> bool:
    if pattern in ("", "*"):
        return True
    rx = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"
    return re.match(rx, value or "", re.IGNORECASE) is not None


def dicom_date_match(query: str, value: str) -> bool:
    if not query:
        return True
    if not value:
        return False
    if "-" in query:
        lo, hi = (query.split("-", 1) + [""])[:2]
        return (not lo or value >= lo) and (not hi or value <= hi)
    return value == query
