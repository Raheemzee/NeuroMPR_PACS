# NeuroMPR PACS

A small, complete PACS for BraTS-format brain-tumour MRI (and plain DICOM) built around the
NeuroMPR viewer. Flask + plain HTML/JS, SQLite, runs on a CPU-only Windows PC.

## Run

```
pip install -r requirements.txt
python app.py
```

The worklist opens at http://127.0.0.1:5000. On the first visit you create the admin account.

## Configure (top of `app.py`)

| Setting | What it does |
|---|---|
| `DATASET_ROOTS` | Folders scanned for BraTS case folders (indexed in place, never modified). More can be added in **Admin → Datasets**. |
| `DATA_DIR` | Where the database, imported studies, caches and reports live (`pacs_data/`). |
| `REQUIRE_LOGIN` | `False` = single user, no sign-in screen. |
| `HOST` | `"0.0.0.0"` to let other PCs on your network use it. |
| `DICOM_AE_TITLE`, `DICOM_PORT` | This PACS on the DICOM network (default `NEUROMPR`, port `11112`). Windows may ask to allow it through the firewall. |
| `CASE_CACHE_MAX` | Studies kept in RAM (~150 MB each for BraTS). |

## What's in it

- **Archive & worklist**: patients, studies and series in SQLite. BraTS 2021 (`_flair/_t1/_t1ce/_t2/_seg`) and 2023+ (`-t2f/-t1n/-t1c/-t2w/-seg`, ET label 3) naming is recognised, and timepoints of one patient are grouped as priors. You can search, filter and sort, and track reading status, priority, assignment and flags. Thumbnails are generated on demand.
- **Viewer**: NeuroMPR (MPR, measurements, segmentation, 3D) opens per study, with previous/next study navigation (`[` `]`). Measurements and auto/model segmentations are saved to the archive.
- **Reports**: draft → signed preliminary → signed final (locked) → addenda. Every version is kept.
- **Users & audit**: admin / radiologist / viewer roles. Every sign-in, open, report, export and DICOM transfer is logged.
- **DICOM**: import of files or zips, export (zip, optionally anonymised), and Storage SCP (receive). It answers C-ECHO, C-FIND, C-GET and C-MOVE, and can send to other nodes (C-STORE SCU).
- **3D scalp & skull**: BraTS images are skull-stripped, so the head layers come from `assets/head_atlas.npz`. That is real CT head anatomy (an average of many scans) fitted to each patient's brain in about 3 s and cached. Each layer has an on/off toggle and an opacity slider, with Opaque / Glass / Skull-only presets, and the cut-away and crop tools cut them too. **It is not the patient's own anatomy** and is labelled as atlas-fitted in the viewer and reports.
- **CPU performance**: the 3D quality setting (Auto / High / Balanced / Fast) times a frame on your PC. Fast uses a half-resolution 3D volume, and resolution drops while you rotate. niivue is bundled locally, so no internet connection is needed.

Research prototype — not for clinical use.

## Files

`app.py` (server), `headmodel.py` (atlas fitting), `dicom_io.py` (DICOM), `static/` (worklist, viewer, sign-in, bundled niivue),
`assets/` (head atlas + licence), `tools/build_head_atlas.py` (only needed to rebuild the atlas).
