"""
headmodel.py — atlas-fitted scalp & skull for skull-stripped brain MRI (BraTS)
=============================================================================
BraTS volumes are skull-stripped, so the patient's own scalp and skull are not in
the data. This module places real head anatomy around each patient's brain:

  * the anatomy comes from assets/head_atlas.npz, built from the average of real
    head CT scans (Rorden Clinical Toolbox CT template, BSD-2 licence): the skull
    is the CT bone surface (> 300 HU), the scalp the CT skin surface
    (tools/build_head_atlas.py rebuilds it);
  * the atlas's own brain mask is registered to the patient's brain mask with a
    12-parameter affine (two resolutions, analytic-gradient L-BFGS), so head size,
    position and proportions follow the patient's brain;
  * wherever the patient's brain still overlaps the fitted skull (individual shape
    differences, mostly at the skull base) the skull is pushed out with a smooth
    displacement field that moves scalp and skull together, keeping a CSF gap.

The result is two triangle meshes in the patient's own world millimetres (the NIfTI
affine space niivue renders in). CPU-only numpy/scipy, ~2-3 s per study; the caller
caches the result on disk.

It is a visual context layer, NOT the patient's anatomy: callers must label it so.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

ATLAS_PATH = Path(__file__).parent / "assets" / "head_atlas.npz"
ATLAS_CREDIT = ("Head atlas: average of real head CT scans, Rorden C. et al. 2012 "
                "(Clinical Toolbox, BSD-2 licence)")
_ATLAS = None


def load_atlas():
    global _ATLAS
    if _ATLAS is None:
        if not ATLAS_PATH.exists():
            raise FileNotFoundError(f"Head atlas not found: {ATLAS_PATH}")
        d = np.load(ATLAS_PATH)
        _ATLAS = {k: d[k] for k in d.files}
        _ATLAS["brain2"] = _ATLAS["brain2"].astype(np.float32)
    return _ATLAS


def atlas_available() -> bool:
    return ATLAS_PATH.exists()


# --------------------------------------------------------------------------------------
# grids
# --------------------------------------------------------------------------------------
def _block_mean(m: np.ndarray, f) -> np.ndarray:
    sx, sy, sz = (m.shape[0] // f[0]) * f[0], (m.shape[1] // f[1]) * f[1], (m.shape[2] // f[2]) * f[2]
    a = m[:sx, :sy, :sz].astype(np.float32)
    return a.reshape(sx // f[0], f[0], sy // f[1], f[1], sz // f[2], f[2]).mean(axis=(1, 3, 5))


def _downsampled_affine(affine: np.ndarray, f) -> np.ndarray:
    A = affine.copy()
    A[:3, :3] = affine[:3, :3] * np.asarray(f, float)[None, :]
    A[:3, 3] = affine[:3, 3] + affine[:3, :3] @ ((np.asarray(f, float) - 1) / 2.0)
    return A


def _zooms(A):
    return np.sqrt((A[:3, :3] ** 2).sum(0))


def _vox_to_mm(A, idx):
    return (A[:3, :3] @ idx.T).T + A[:3, 3]


# --------------------------------------------------------------------------------------
# patient brain mask (at ~2 mm, which is all the fit needs)
# --------------------------------------------------------------------------------------
def patient_brain_mask(vols: dict, affine: np.ndarray):
    """Union of non-zero voxels over the MR series (BraTS background is exactly 0),
    reduced to ~2 mm, largest component, holes filled.
    Returns (mask2 bool, affine2, volume_ml, fraction_of_fov)."""
    union = None
    for key, v in vols.items():
        if key == "seg":
            continue
        m = v != 0
        union = m if union is None else (union | m)
    if union is None:
        raise ValueError("no image volumes")
    frac = float(union.mean())
    f = tuple(int(max(1, round(2.0 / z))) for z in _zooms(affine))
    P2 = _block_mean(union, f) > 0.5
    A2 = _downsampled_affine(affine, f)
    lab, n = ndi.label(P2)
    if n > 1:
        sizes = ndi.sum(P2, lab, range(1, n + 1))
        P2 = lab == (int(np.argmax(sizes)) + 1)
    P2 = ndi.binary_fill_holes(P2)
    vol_ml = float(P2.sum()) * float(np.prod(_zooms(A2))) / 1000.0
    return P2, A2, vol_ml, frac


def skull_stripped_check(vol_ml: float, frac: float) -> tuple[bool, str]:
    """A skull-stripped brain is ~0.9-1.9 L and fills a minority of the field of view;
    a full-head MRI (or a CT) does not — and already shows its own scalp/skull."""
    if frac > 0.5:
        return False, f"non-zero signal fills {frac:.0%} of the field of view (not skull-stripped)"
    if vol_ml > 2600:
        return False, f"masked volume {vol_ml:.0f} mL is larger than a brain (not skull-stripped)"
    if vol_ml < 300:
        return False, f"masked volume {vol_ml:.0f} mL is too small for a whole brain"
    return True, f"brain mask {vol_ml:.0f} mL"


# --------------------------------------------------------------------------------------
# affine registration: atlas brain mask -> patient brain mask
# --------------------------------------------------------------------------------------
class _Level:
    """Pre-smoothed patient samples + atlas image/gradients at one resolution."""

    def __init__(self, P2, Ap2, T2, At2, sigma_mm, stride, margin_vox, band=False):
        Ps = ndi.gaussian_filter(P2.astype(np.float32), sigma_mm / _zooms(Ap2))
        nz = np.argwhere(P2)
        lo = np.maximum(nz.min(0) - margin_vox, 0)
        hi = np.minimum(nz.max(0) + margin_vox + 1, P2.shape)
        idx = np.stack(np.meshgrid(*[np.arange(lo[d], hi[d], stride) for d in range(3)], indexing="ij"), -1).reshape(-1, 3)
        pv = Ps[idx[:, 0], idx[:, 1], idx[:, 2]]
        if band:
            # once roughly aligned only the boundary band carries information
            keep = (pv > 0.01) & (pv < 0.99)
            idx, pv = idx[keep], pv[keep]
        self.p = pv.astype(np.float64)
        self.x = _vox_to_mm(Ap2, idx)                                     # patient world mm
        self.T = ndi.gaussian_filter(T2, sigma_mm / _zooms(At2)).astype(np.float32)
        self.G = [g.astype(np.float32) for g in np.gradient(self.T)]
        Ainv = np.linalg.inv(At2)
        self.J = Ainv[:3, :3]
        self.o = Ainv[:3, 3:4]

    def sample(self, y):
        u = (self.J @ y.T) + self.o
        t = ndi.map_coordinates(self.T, u, order=1, mode="constant", cval=0.0)
        gu = np.stack([ndi.map_coordinates(g, u, order=1, mode="constant", cval=0.0) for g in self.G], 1)
        return t.astype(np.float64), (gu @ self.J).astype(np.float64)    # chain rule into mm


def _optimize(level: _Level, c, S, Mt, tt, maxiter):
    xt = (level.x - c) / S

    def fg(p):
        M = p[:9].reshape(3, 3)
        y = xt @ M.T + p[9:]
        tv, gy = level.sample(y)
        r = tv - level.p
        n = len(r)
        w = (2.0 / n) * r[:, None] * gy
        return float(r @ r) / n, np.concatenate([(w.T @ xt).ravel(), w.sum(0)])

    res = minimize(fg, np.concatenate([Mt.ravel(), tt]), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    return res.x[:9].reshape(3, 3), res.x[9:], float(res.fun)


def fit_atlas_affine(P2: np.ndarray, Ap2: np.ndarray, atlas=None):
    """Return (M, t, info): y_atlas_mm = M @ x_patient_mm + t."""
    atlas = atlas or load_atlas()
    T2, At2 = atlas["brain2"], atlas["brain2_affine"]

    def moments(m, A):
        x = _vox_to_mm(A, np.argwhere(m > 0.5))
        return x.mean(0), np.cov(x.T)

    cp, Cp = moments(P2, Ap2)
    ct, Ct = moments(T2, At2)
    S = 50.0                                                              # conditioning: centred, /50 mm
    s = np.sqrt(np.diag(Ct) / np.maximum(np.diag(Cp), 1e-6))

    L1 = _Level(P2, Ap2, T2, At2, sigma_mm=6.0, stride=3, margin_vox=10)
    best = None
    # NIfTI world axes are RAS by convention; the three 180-degree turns are tried too
    # so a mislabelled orientation can't put the face on the back of the head
    for R in (np.eye(3), np.diag([-1, -1, 1]), np.diag([1, -1, -1]), np.diag([-1, 1, -1])):
        M1, t1, f1 = _optimize(L1, cp, S, np.diag(s) @ R * S, ct.copy(), 15)
        if best is None or f1 < best[2]:
            best = (M1, t1, f1, R)
    M1, t1, _, R = best
    M1, t1, _ = _optimize(L1, cp, S, M1, t1, 40)
    L2 = _Level(P2, Ap2, T2, At2, sigma_mm=3.0, stride=2, margin_vox=8, band=True)
    M2, t2, f2 = _optimize(L2, cp, S, M1, t1, 60)
    M = M2 / S
    t = t2 - M @ cp

    # quality: Dice of the warped atlas brain vs the patient's brain
    y = _vox_to_mm(Ap2, np.argwhere(np.ones(P2.shape, bool))) @ M.T + t
    Ai = np.linalg.inv(At2)
    warped = ndi.map_coordinates(T2, (Ai[:3, :3] @ y.T) + Ai[:3, 3:4], order=1, cval=0).reshape(P2.shape) > 0.5
    dice = 2.0 * float((warped & P2).sum()) / max(1.0, float(warped.sum() + P2.sum()))
    scale = float(abs(np.linalg.det(M)) ** (-1 / 3))
    return M, t, {"dice": round(dice, 3), "scale": round(scale, 3),
                  "orientation_flip": not np.allclose(R, np.eye(3))}


# --------------------------------------------------------------------------------------
# smooth push-out where the patient's brain still overlaps the fitted skull
# --------------------------------------------------------------------------------------
def _push_out(skull_v, scalp_v, P2, A2, margin_mm=1.5, iters=4):
    """Skull vertices that end up inside the patient's brain are moved out along the
    brain's distance gradient. The displacement is spread with a Gaussian
    (normalised convolution on a ~4 mm grid) and applied to BOTH meshes, so nearby
    bone and scalp move together and the layers keep their thickness."""
    pad = 12                                                              # room for the head around the brain
    P = np.pad(P2, pad)
    A = A2.copy()
    A[:3, 3] -= A[:3, :3] @ np.full(3, float(pad))
    z = _zooms(A)
    sd = (ndi.distance_transform_edt(~P, sampling=z) - ndi.distance_transform_edt(P, sampling=z)).astype(np.float32)
    g = [gi / zi for gi, zi in zip(np.gradient(sd), z)]
    Ai = np.linalg.inv(A)
    R = A[:3, :3] / z[None, :]
    # displacement field lives on a 2x coarser grid
    F = 2
    shp4 = tuple(int(np.ceil(n / F)) for n in P.shape)
    sig4 = 5.0 / (z * F)

    def u2(v):
        return (Ai[:3, :3] @ v.T) + Ai[:3, 3:4]

    moved = 0
    for _ in range(iters):
        u = u2(skull_v)
        d = ndi.map_coordinates(sd, u, order=1, mode="nearest")
        need = np.clip(margin_mm - d, 0, 20)
        hit = need > 0.05
        if not hit.any():
            break
        moved += int(hit.sum())
        gv = np.stack([ndi.map_coordinates(gi, u[:, hit], order=1, mode="nearest") for gi in g], 1) @ R.T
        gv /= np.maximum(np.linalg.norm(gv, axis=1, keepdims=True), 1e-6)
        disp = need[hit, None] * gv
        iu = np.clip(np.rint(u[:, hit] / F).astype(int), 0, np.array(shp4)[:, None] - 1)
        num = np.zeros((3,) + shp4, np.float32)
        den = np.zeros(shp4, np.float32)
        for k in range(3):
            np.add.at(num[k], (iu[0], iu[1], iu[2]), disp[:, k])
        np.add.at(den, (iu[0], iu[1], iu[2]), 1.0)
        den_s = ndi.gaussian_filter(den, sig4)
        w = np.clip(den_s / max(float(den_s.max()) * 0.02, 1e-9), 0, 1) / np.maximum(den_s, 1e-9)
        field = [ndi.gaussian_filter(num[k], sig4) * w for k in range(3)]

        def apply(v):
            uu = u2(v) / F
            return (v + np.stack([ndi.map_coordinates(fk, uu, order=1, mode="nearest") for fk in field], 1)).astype(np.float32)

        skull_v, scalp_v = apply(skull_v), apply(scalp_v)
    u = u2(skull_v)
    inside = float((ndi.map_coordinates(P.astype(np.float32), u, order=0, mode="constant") > 0.5).mean())
    return skull_v, scalp_v, {"pushed_vertex_steps": moved, "skull_inside_brain": round(inside, 4)}


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------
def build_head_meshes(vols: dict, affine: np.ndarray) -> dict:
    """vols: modality -> 3D array (native voxel order); affine: the NIfTI affine.
    Returns {"skull": (verts, faces), "scalp": (verts, faces), "info": {...}} with
    vertices in the patient's world millimetres (float32) and faces as int32."""
    t0 = time.time()
    atlas = load_atlas()
    P2, A2, vol_ml, frac = patient_brain_mask(vols, affine)
    ok, why = skull_stripped_check(vol_ml, frac)
    if not ok:
        raise ValueError(f"Head model not applicable: {why}. A full-head scan already shows "
                         f"the patient's own scalp and skull in the 3D render.")
    M, t, info = fit_atlas_affine(P2, A2, atlas)
    Minv = np.linalg.inv(M)
    skull_v = ((atlas["skull_v"] - t) @ Minv.T).astype(np.float32)
    scalp_v = ((atlas["scalp_v"] - t) @ Minv.T).astype(np.float32)
    skull_v, scalp_v, push_info = _push_out(skull_v, scalp_v, P2, A2)
    info.update(push_info)
    info["seconds"] = round(time.time() - t0, 2)
    info["brain"] = why
    info["credit"] = ATLAS_CREDIT
    return {"skull": (skull_v, atlas["skull_f"].astype(np.int32)),
            "scalp": (scalp_v, atlas["scalp_f"].astype(np.int32)),
            "info": info}


def mesh_to_bytes(v: np.ndarray, f: np.ndarray) -> bytes:
    """Little-endian: uint32 nVerts, uint32 nTris, float32 xyz * nVerts, uint32 abc * nTris."""
    v = np.ascontiguousarray(v, dtype="<f4")
    f = np.ascontiguousarray(f, dtype="<u4")
    return np.array([len(v), len(f)], dtype="<u4").tobytes() + v.tobytes() + f.tobytes()
