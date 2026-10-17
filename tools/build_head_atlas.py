"""Rebuilds assets/head_atlas.npz. NOT needed to run the PACS — the atlas ships prebuilt.

Source: Chris Rorden's Clinical Toolbox templates (BSD-2), 1 mm MNI space. Download into
tools/atlas_src/ :
  https://raw.githubusercontent.com/neurolabusc/Clinical/master/Tutorial/high_res/scct_unsmooth.nii
  https://raw.githubusercontent.com/neurolabusc/Clinical/master/Tutorial/high_res/betsct1_unsmooth.nii
    scct_unsmooth.nii    - average of real head CT scans (Hounsfield units)
    betsct1_unsmooth.nii - matching skull-stripped T1 template (brain mask for fitting)
Needs: pip install scikit-image fast-simplification
Outputs, all in template millimetres (RAS):
  skull_v/skull_f  - bone surface (CT > 300 HU): outer + inner table, face bones
  scalp_v/scalp_f  - outer skin surface (CT > -400 HU, internal air filled)
  brain2           - brain mask at 2 mm (+ affine) used to fit the atlas to each patient
"""
import sys, time
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import fast_simplification

from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE / "atlas_src"
OUT = HERE.parent / "assets" / "head_atlas.npz"
t0 = time.time()

ct_img = nib.load(f"{SRC}/scct_unsmooth.nii")
A = ct_img.affine
ct = ct_img.get_fdata().astype(np.float32)
bet = nib.load(f"{SRC}/betsct1_unsmooth.nii").get_fdata() > 0


def largest(mask, min_size=None):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    if min_size is None:
        return lab == (np.argmax(sizes) + 1)
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_size
    return keep[lab]


# --- brain mask (for registration)
brain = largest(ndi.binary_fill_holes(bet))
brain = ndi.binary_closing(brain, iterations=2)

# --- bone
cts = ndi.gaussian_filter(ct, 0.6)
bone = largest(cts > 300, min_size=4000)
print("bone vox", int(bone.sum()), "brain vox", int(brain.sum()))

# --- head (skin): everything denser than air, internal air spaces filled
head = cts > -400
head = np.stack([ndi.binary_fill_holes(head[:, :, z]) for z in range(head.shape[2])], axis=2)
head = largest(ndi.binary_fill_holes(head))


def taubin(v, f, iters=20, lam=0.5, mu=-0.53):
    """Shrink-free smoothing (Taubin 1995): removes the CT template's slice terracing."""
    import scipy.sparse as sp
    n = len(v)
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    W = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    W.data[:] = 1.0
    deg = np.asarray(W.sum(1)).ravel()
    Wn = sp.diags(1.0 / np.maximum(deg, 1)) @ W
    v = v.astype(np.float64)
    for _ in range(iters):
        v = v + lam * (Wn @ v - v)
        v = v + mu * (Wn @ v - v)
    return v.astype(np.float32)


def surface(field, level, mask_region, fill_value, target_tris, smooth_iters):
    f = np.where(mask_region, field, fill_value).astype(np.float32)
    f = np.pad(f, 2, constant_values=fill_value)          # closed caps where the template box cuts the head
    v, fc, _, _ = measure.marching_cubes(f, level=level, allow_degenerate=False)
    v -= 2.0
    mm = (A[:3, :3] @ v.T).T + A[:3, 3]
    mm = taubin(mm, fc, smooth_iters)
    n0 = len(fc)
    red = max(0.0, 1.0 - target_tris / n0)
    if red > 0:
        mm, fc = fast_simplification.simplify(mm.astype(np.float32), fc.astype(np.int32), target_reduction=red)
    mm = taubin(mm, fc, 3)
    fc = fc[:, [0, 2, 1]]          # skimage winds triangles inward for this field; niivue lights by winding
    print(f"  {n0} -> {len(fc)} triangles")
    return mm.astype(np.float32), fc.astype(np.int32)


# skull: real HU values near the kept bone components, so the iso-surface is sub-voxel accurate
skull_region = ndi.binary_dilation(bone, iterations=2)
skull_v, skull_f = surface(cts, 300.0, skull_region, -1024.0, 140_000, 8)

# scalp: CT with internal air set to soft tissue, so -400 HU gives only the outer skin
sc = cts.copy()
sc[head & (sc < 0)] = 0.0
sc[~ndi.binary_dilation(head, iterations=2)] = -1024.0
sc = ndi.gaussian_filter(sc, 1.5)       # the air/skin step is ~1 voxel wide: smooth it or marching cubes terraces
scalp_v, scalp_f = surface(sc, -400.0, np.ones_like(head), -1024.0, 70_000, 30)

# 2 mm brain mask for fitting
b2 = brain[: brain.shape[0] // 2 * 2, : brain.shape[1] // 2 * 2, : brain.shape[2] // 2 * 2]
b2 = b2.reshape(b2.shape[0] // 2, 2, b2.shape[1] // 2, 2, b2.shape[2] // 2, 2).mean(axis=(1, 3, 5)) >= 0.5
A2 = A.copy()
A2[:3, :3] = A[:3, :3] * 2
A2[:3, 3] = A[:3, 3] + A[:3, :3] @ np.array([0.5, 0.5, 0.5])

np.savez_compressed(
    OUT,
    skull_v=skull_v, skull_f=skull_f, scalp_v=scalp_v, scalp_f=scalp_f,
    brain2=b2.astype(np.uint8), brain2_affine=A2,
    source=np.array("Rorden Clinical Toolbox (BSD-2): scct_unsmooth.nii (CT head template), betsct1_unsmooth.nii"),
)
print("saved", OUT, f"{time.time() - t0:.1f}s")
