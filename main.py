# main.py
# TotalSegmentator + STU-Net-B. MONAI is currently disabled.
# Windows/CPU-friendly.

import os
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("PYVISTA_QT_API", "pyqt5")
os.environ.setdefault("PYVISTA_ALLOW_DUPLICATE_QT_INSTANCES", "true")

import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import nibabel as nib
from skimage import measure
from PyQt5 import QtWidgets, QtCore
from pyvistaqt import QtInteractor
import pyvista as pv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import (
    generate_binary_structure,
    label as cc_label,
    binary_fill_holes,
    binary_closing,
    binary_dilation,
    binary_erosion,
)

try:
    pv.global_theme.multi_samples = 0
except Exception:
    pass

# =============================== PATHS ===============================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT / "data"
CT_ROOT      = DATA_ROOT / "ct"
GT_ROOT      = DATA_ROOT / "segmentations"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"
PREVIEW_ROOT = PROJECT_ROOT / "previews"
EVAL_ROOT    = PROJECT_ROOT / "eval"
for d in [OUTPUT_ROOT, PREVIEW_ROOT, EVAL_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_CASE_ID = "s0024"
def ct_path_from_case(case_id: str) -> Path:
    return CT_ROOT / case_id / "ct.nii.gz"

# =============================== GROUND-TRUTH MAP ===============================
GT_MAP: Dict[str, List[Path]] = {
    "heart": [GT_ROOT / "heart.nii.gz", GT_ROOT / "aorta.nii.gz"],
    "liver": [GT_ROOT / "liver.nii.gz"],
    "lungs": [
        GT_ROOT / "lung_lower_lobe_left.nii.gz",
        GT_ROOT / "lung_lower_lobe_right.nii.gz",
        GT_ROOT / "lung_middle_lobe_right.nii.gz",
        GT_ROOT / "lung_upper_lobe_left.nii.gz",
        GT_ROOT / "lung_upper_lobe_right.nii.gz",
    ],
}
DEFAULT_PLANE = {"heart": "axial", "liver": "axial", "lungs": "coronal"}

# =============================== METRICS / UTILS ===============================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def window_level(img_hu: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """Apply CT windowing and normalize to [0,1]."""
    lo, hi = wl - ww / 2, wl + ww / 2
    img = np.clip(img_hu, lo, hi)
    return (img - lo) / (hi - lo + 1e-8)

def dice_coefficient(gt: np.ndarray, pr: np.ndarray) -> float:
    inter = np.logical_and(gt, pr).sum()
    return float(2.0 * inter / (gt.sum() + pr.sum() + 1e-8))

def iou_score(gt: np.ndarray, pr: np.ndarray) -> float:
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    return float(inter / (union + 1e-8))

# =============================== POST-PROCESSING (mm-aware) ===============================
_STRUCT = generate_binary_structure(3, 2)

def _keep_largest_cc(m: np.ndarray) -> np.ndarray:
    if not m.any():
        return m
    lab, n = cc_label(m, structure=_STRUCT)
    if n <= 1:
        return m
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return (lab == counts.argmax())

def _ball_structure_from_spacing(radius_mm: float, spacing_xyz: Tuple[float, float, float]) -> np.ndarray:
    """Create an ellipsoidal structuring element based on voxel spacing and desired radius in mm."""
    sx, sy, sz = spacing_xyz
    rx = max(1, int(round(radius_mm / max(sx, 1e-6))))
    ry = max(1, int(round(radius_mm / max(sy, 1e-6))))
    rz = max(1, int(round(radius_mm / max(sz, 1e-6))))
    X, Y, Z = np.ogrid[-rx:rx + 1, -ry:ry + 1, -rz:rz + 1]
    ellip = (X / (rx + 1e-6)) ** 2 + (Y / (ry + 1e-6)) ** 2 + (Z / (rz + 1e-6)) ** 2 <= 1.0
    return ellip.astype(bool)

def clean_mask_mm(
    m_bool: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    min_size_vox: int = 1500,
    close_radius_mm: float = 2.5,
) -> np.ndarray:
    """Clean binary mask with connected components, hole filling, and mm-aware closing."""
    m = m_bool.astype(bool)
    if not m.any():
        return m.astype(np.uint8)

    lab, n = cc_label(m, structure=_STRUCT)
    if n > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        keep_ids = np.where(counts >= min_size_vox)[0]
        m = np.isin(lab, keep_ids)

    if not m.any():
        return m.astype(np.uint8)

    m = _keep_largest_cc(m)
    m = binary_fill_holes(m)

    se = _ball_structure_from_spacing(close_radius_mm, spacing_xyz)
    m = binary_closing(m, structure=se, iterations=1)
    return m.astype(np.uint8)

def bridge_union_mm(
    union_bool: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    bridge_radius_mm: float = 4.5,
) -> np.ndarray:
    """Bridge nearby parts (close gaps) with mm-aware dilation/erosion."""
    se = _ball_structure_from_spacing(bridge_radius_mm, spacing_xyz)
    grown = binary_dilation(union_bool, structure=se, iterations=1)
    bridged = binary_erosion(grown, structure=se, iterations=1)
    bridged = binary_fill_holes(bridged)
    return bridged.astype(np.uint8)

def clean_mask(m_bool: np.ndarray, min_size: int = 1500, close_iters: int = 1) -> np.ndarray:
    """Lightweight cleaner (voxel-based) for less sensitive classes like lung lobes."""
    m = m_bool.astype(bool)
    if not m.any():
        return m.astype(np.uint8)

    lab, n = cc_label(m, structure=_STRUCT)
    if n > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        keep = counts >= min_size
        out = np.zeros_like(m, dtype=bool)
        for i, k in enumerate(keep):
            if i == 0 or not k:
                continue
            out |= (lab == i)
        m = out

    if not m.any():
        return m.astype(np.uint8)

    m = _keep_largest_cc(m)
    m = binary_fill_holes(m)
    if close_iters > 0:
        m = binary_closing(m, structure=_STRUCT, iterations=close_iters)
    return m.astype(np.uint8)

# =============================== TOTALSEGMENTATOR ===============================
ORGAN_TASK = {"heart": "heartchambers_highres", "liver": "liver_segments", "lungs": "total"}
ROI_SUBSETS = {
    "heart": None,
    "liver": None,
    "lungs": [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ],
}

def find_totalseg_cmd() -> Tuple[List[str], str]:
    scripts_dir = Path(sys.executable).parent / "Scripts"
    for name in ["TotalSegmentator.exe", "totalseg.exe"]:
        cand = scripts_dir / name
        if cand.exists():
            return [str(cand)], f"exe:{name}"
    for name in ["TotalSegmentator", "totalseg"]:
        on_path = shutil.which(name)
        if on_path:
            return [on_path], f"path:{name}"
    return [sys.executable, "-m", "totalsegmentator.python_api"], "module"

def run_totalseg(organ: str, ct_path: Path, out_dir: Path) -> None:
    """Run TotalSegmentator for the requested organ/task."""
    task = ORGAN_TASK[organ]
    base_cmd, mode = find_totalseg_cmd()
    cmd = base_cmd + ["-i", str(ct_path), "-o", str(out_dir), "--task", task, "--verbose"]
    rs = ROI_SUBSETS.get(organ)
    if task == "total" and rs:
        cmd += ["--roi_subset"] + rs
    print(f"🚀 Running TotalSegmentator ({mode}):", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("\n----- STDOUT (head) -----")
    print("\n".join(proc.stdout.splitlines()[:120]) or "<empty>")
    print("----- STDERR (head) -----")
    print("\n".join(proc.stderr.splitlines()[:120]) or "<empty>")
    if proc.returncode != 0:
        raise RuntimeError(f"TotalSegmentator failed (code={proc.returncode}).")
    if not list(out_dir.glob("*.nii.gz")):
        raise RuntimeError("TotalSegmentator produced no .nii.gz.")

# =============================== STU-NET-B ===============================
# Label IDs for medim STU-Net-B (dataset="TotalSegmentator")
STUNET_ID_TO_NAME = {
    22: "heart_atrium_left",
    23: "heart_atrium_right",
    24: "heart_myocardium",
    25: "heart_ventricle_left",
    26: "heart_ventricle_right",
    40: "liver",
    41: "lung_lower_lobe_left",
    42: "lung_lower_lobe_right",
    43: "lung_middle_lobe_right",
    44: "lung_upper_lobe_left",
    45: "lung_upper_lobe_right",
    3:  "aorta",
}
HEART_IDS_STUNET = [22, 23, 24, 25, 26]
LIVER_ID_STUNET  = 40
LUNG_IDS_STUNET  = [41, 42, 43, 44, 45]
AORTA_ID_STUNET  = 3

def run_stunet(organ: str, ct_path: Path, out_dir: Path) -> None:
    """
    STU-Net-B inference (via medim) with mm-aware post-processing for heart.
    NOTE: STU-Net does not output liver segments; we fallback to TotalSegmentator for liver.
    """
    # For liver, use TotalSegmentator's liver_segments task
    if organ == "liver":
        for p in Path(out_dir).glob("*.nii.gz"):
            try:
                p.unlink()
            except Exception:
                pass
        run_totalseg("liver", ct_path, out_dir)
        return

    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    try:
        from monai.inferers import sliding_window_inference
    except ImportError:
        raise ImportError("MONAI needed for sliding_window_inference. Run: pip install monai nibabel")
    try:
        import medim
    except ImportError:
        raise ImportError("Package 'medim' not installed. Try: pip install medim")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_ni = nib.load(str(ct_path))
    vol = img_ni.get_fdata().astype(np.float32)
    vol = (vol - float(vol.mean())) / (float(vol.std()) + 1e-8)
    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, Z, Y, X)

    model = medim.create_model("STU-Net-B", dataset="TotalSegmentator")
    model.eval().to(device)

    with torch.no_grad():
        logits = sliding_window_inference(x, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model, overlap=0.5)
        labels = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    ensure_dir(out_dir)
    aff = img_ni.affine

    def save_mask(m_bool: np.ndarray, name: str):
        if m_bool.any():
            nib.save(nib.Nifti1Image(m_bool.astype(np.uint8), aff), str(Path(out_dir) / f"{name}.nii.gz"))

    # Heart: mm-aware cleaning + optional bridging
    if organ == "heart":
        try:
            s = tuple(np.abs(np.diag(img_ni.affine))[:3])
            spacing = (float(s[0]), float(s[1]), float(s[2]))
        except Exception:
            spacing = (1.0, 1.0, 1.0)

        parts = {}
        MIN_ATRIA = 800
        MIN_VENTS = 1500
        MIN_MYO   = 1800

        for cid in HEART_IDS_STUNET:
            raw = (labels == cid)
            if cid in (22, 23):       # atria
                m = clean_mask_mm(raw, spacing, min_size_vox=MIN_ATRIA, close_radius_mm=2.0)
            elif cid == 24:           # myocardium
                m = clean_mask_mm(raw, spacing, min_size_vox=MIN_MYO, close_radius_mm=2.5)
            else:                     # ventricles
                m = clean_mask_mm(raw, spacing, min_size_vox=MIN_VENTS, close_radius_mm=2.5)
            name = STUNET_ID_TO_NAME.get(cid, f"id_{cid}")
            parts[name] = m
            save_mask(m, name)

        a = clean_mask_mm(labels == AORTA_ID_STUNET, spacing, min_size_vox=1200, close_radius_mm=2.0)
        save_mask(a, "aorta")

        union = np.zeros_like(labels, dtype=bool)
        for m in parts.values():
            union |= (m > 0)
        if union.any():
            union = bridge_union_mm(union, spacing, bridge_radius_mm=3.5)  # tweak 3.0–5.0 mm as needed
            union = clean_mask_mm(union, spacing, min_size_vox=4000, close_radius_mm=2.5)
            save_mask(union, "heart")

    # Lungs: light cleanup per lobe
    elif organ == "lungs":
        for cid in LUNG_IDS_STUNET:
            raw = (labels == cid)
            m = clean_mask(raw, min_size=1500, close_iters=1)
            save_mask(m, STUNET_ID_TO_NAME.get(cid, f"id_{cid}"))

    # Save full labelmap (can be deleted if not needed)
    nib.save(nib.Nifti1Image(labels, aff), str(Path(out_dir) / "stunet_labels.nii.gz"))

    # Cleanup
    del model, x, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================== EVALUATION ===============================
HEART_GROUP = [
    "heart_atrium_left.nii.gz",
    "heart_atrium_right.nii.gz",
    "heart_ventricle_left.nii.gz",
    "heart_ventricle_right.nii.gz",
]

def evaluate_folder_against_gt(gt_files: List[Path], pred_dir: Path) -> Dict[str, Dict[str, float]]:
    """Compute Dice/IoU for any GT file that has a matching predicted mask."""
    pred_paths = [p for p in pred_dir.glob("*.nii.gz")]
    if not pred_paths:
        return {}
    pred_by_name = {p.name: p for p in pred_paths}
    results: Dict[str, Dict[str, float]] = {}

    for gpath in gt_files:
        if not gpath.exists():
            continue
        gt_img = nib.load(str(gpath))
        gt = gt_img.get_fdata() > 0

        if gpath.name in pred_by_name:
            pred = nib.load(str(pred_by_name[gpath.name])).get_fdata() > 0
        elif gpath.name == "heart.nii.gz":
            union = None
            for name in HEART_GROUP:
                p = pred_by_name.get(name)
                if not p:
                    continue
                arr = nib.load(str(p)).get_fdata() > 0
                union = arr if union is None else np.logical_or(union, arr)
            if union is None:
                for p in pred_paths:
                    arr = nib.load(str(p)).get_fdata() > 0
                    union = arr if union is None else np.logical_or(union, arr)
            pred = union
        elif len(gt_files) == 1:
            union = None
            for p in pred_paths:
                arr = nib.load(str(p)).get_fdata() > 0
                union = arr if union is None else np.logical_or(union, arr)
            pred = union
        else:
            continue

        if pred is None:
            continue

        results[gpath.name] = {
            "Dice": round(dice_coefficient(gt, pred), 4),
            "IoU":  round(iou_score(gt, pred), 4),
            # SurfaceDice removed as requested. ASSD can be added later if needed.
        }
    return results

# =============================== 2D PREVIEW ===============================
def pick_best_slice(vol: np.ndarray, plane: str, masks: List[np.ndarray]) -> int:
    """Pick a slice index that maximizes overlay coverage for visualization."""
    X, Y, Z = vol.shape
    if not masks:
        return {"axial": Z // 2, "coronal": Y // 2, "sagittal": X // 2}[plane]
    if plane == "axial":
        scores = np.zeros(Z, dtype=np.int64)
        for m in masks:
            if m.shape != vol.shape:
                continue
            scores += m.sum(axis=(0, 1))
        return int(scores.argmax()) if scores.max() > 0 else Z // 2
    if plane == "coronal":
        scores = np.zeros(Y, dtype=np.int64)
        for m in masks:
            if m.shape != vol.shape:
                continue
            scores += m.sum(axis=(0, 2))
        return int(scores.argmax()) if scores.max() > 0 else Y // 2
    scores = np.zeros(X, dtype=np.int64)
    for m in masks:
        if m.shape != vol.shape:
            continue
        scores += m.sum(axis=(1, 2))
    return int(scores.argmax()) if scores.max() > 0 else X // 2

def save_2d_preview(ct_path: Path, masks_dir: Path, out_png: Path, plane: str, wl=40, ww=400, alpha=0.35):
    """Save a 2D slice preview with colored overlays."""
    img = nib.load(str(ct_path))
    vol = img.get_fdata()
    mask_files = sorted([p for p in masks_dir.glob("*.nii.gz")])
    masks = []
    for p in mask_files:
        try:
            arr = nib.load(str(p)).get_fdata()
            if arr.shape == vol.shape:
                masks.append(arr > 0)
        except Exception:
            pass
    sl_idx = pick_best_slice(vol, plane, masks)
    if plane == "axial":
        base = window_level(vol[:, :, sl_idx], wl, ww)
    elif plane == "coronal":
        base = window_level(vol[:, sl_idx, :], wl, ww)
    else:
        base = window_level(vol[sl_idx, :, :], wl, ww)
    plt.figure(figsize=(7, 7))
    plt.imshow(base, cmap="gray", origin="lower")
    plt.axis("off")
    colors = plt.get_cmap("tab20").colors
    cidx = 0
    for arr in masks:
        sl = arr[:, :, sl_idx] if plane == "axial" else (arr[:, sl_idx, :] if plane == "coronal" else arr[sl_idx, :, :])
        if sl.sum() == 0:
            continue
        c = colors[cidx % len(colors)]
        cidx += 1
        plt.imshow(np.ma.masked_where(~sl, sl), cmap=matplotlib.colors.ListedColormap([c]), alpha=alpha, origin="lower")
    ensure_dir(out_png.parent)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

# =============================== 3D VIEWER ===============================
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, ct_path: Path, organ: str, model: str, out_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Segmentation Viewer — {organ} / {model}")
        self.resize(1280, 760)
        self.frame = QtWidgets.QFrame()
        self.setCentralWidget(self.frame)
        layout = QtWidgets.QHBoxLayout(self.frame)
        self.controls = QtWidgets.QVBoxLayout()
        layout.addLayout(self.controls, 1)
        self.plotter = QtInteractor(self.frame)
        layout.addWidget(self.plotter.interactor, 4)
        try:
            rw = self.plotter.render_window
            if hasattr(rw, "SetMultiSamples"):
                rw.SetMultiSamples(0)
        except Exception:
            pass
        try:
            if hasattr(self.plotter, "enable_anti_aliasing"):
                self.plotter.enable_anti_aliasing(False)
        except Exception:
            pass

        self.ct_path, self.organ, self.model, self.out_dir = ct_path, organ, model, out_dir

        spacing = (1.0, 1.0, 1.0)
        try:
            ct_img = nib.load(str(ct_path))
            s = tuple(np.abs(np.diag(ct_img.affine)[:3]))
            if len(s) == 3 and not np.any(np.isnan(s)):
                spacing = s
        except Exception:
            pass

        self.actors: Dict[str, Dict] = {}
        base_colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "pink", "purple", "brown"]
        mask_files = sorted([p for p in out_dir.glob("*.nii.gz") if p.name != "stunet_labels.nii.gz"])
        for i, p in enumerate(mask_files):
            vol = nib.load(str(p)).get_fdata()
            if np.count_nonzero(vol) < 500:  # skip tiny fragments
                continue
            vol_bin = (vol > 0).astype(np.uint8)
            try:
                verts, faces, _, _ = measure.marching_cubes(vol_bin, 0.5, spacing=spacing)
                faces_pv = np.hstack([np.c_[np.full((faces.shape[0], 1), 3), faces]]).astype(np.int64)
                mesh = pv.PolyData(verts, faces_pv).smooth(n_iter=200, relaxation_factor=0.03)
                color = base_colors[i % len(base_colors)]
                actor = self.plotter.add_mesh(mesh, color=color, opacity=1.0, name=p.name)
                self.actors[p.name] = {"actor": actor, "color": color, "opacity": 1.0, "visible": True}
            except Exception:
                continue

        self.selector = QtWidgets.QComboBox()
        self.selector.addItems(list(self.actors.keys()))
        self.selector.currentTextChanged.connect(self.on_select)
        self.controls.addWidget(QtWidgets.QLabel("Select Mask:"))
        self.controls.addWidget(self.selector)

        self.chk_visible = QtWidgets.QCheckBox("Visible")
        self.chk_visible.setChecked(True)
        self.chk_visible.stateChanged.connect(self.on_visibility)
        self.controls.addWidget(self.chk_visible)

        self.btn_color = QtWidgets.QPushButton("Pick Color")
        self.btn_color.clicked.connect(self.on_color)
        self.controls.addWidget(self.btn_color)

        self.slider_op = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_op.setRange(0, 100)
        self.slider_op.setValue(100)
        self.slider_op.valueChanged.connect(self.on_opacity)
        self.controls.addWidget(QtWidgets.QLabel("Opacity"))
        self.controls.addWidget(self.slider_op)

        self.btn_eval = QtWidgets.QPushButton("Evaluate vs GT")
        self.btn_eval.clicked.connect(self.on_evaluate)
        self.controls.addWidget(self.btn_eval)
        self.controls.addStretch()
        self.current = list(self.actors.keys())[0] if self.actors else None

    def closeEvent(self, event):
        try:
            for name, d in list(self.actors.items()):
                try:
                    self.plotter.remove_actor(d.get("actor"))
                except Exception:
                    pass
            self.actors.clear()
            try:
                self.plotter.clear()
            except Exception:
                pass
            try:
                rw = self.plotter.render_window
                if hasattr(rw, "Finalize"):
                    rw.Finalize()
            except Exception:
                pass
            try:
                self.plotter.close()
                self.plotter.interactor.close()
            except Exception:
                pass
            try:
                self.plotter.setParent(None)
                pv.close_all()
            except Exception:
                pass
        finally:
            super().closeEvent(event)

    def on_select(self, name: str):
        self.current = name

    def on_visibility(self, state: int):
        if not self.current:
            return
        a = self.actors[self.current]["actor"]
        a.SetVisibility(1 if state == QtCore.Qt.Checked else 0)
        self.plotter.render()

    def on_color(self):
        if not self.current:
            return
        a = self.actors[self.current]["actor"]
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            a.GetProperty().SetColor((color.redF(), color.greenF(), color.blueF()))
            self.plotter.render()

    def on_opacity(self, v: int):
        if not self.current:
            return
        a = self.actors[self.current]["actor"]
        a.GetProperty().SetOpacity(v / 100.0)
        self.plotter.render()

    def on_evaluate(self):
        gt_files = GT_MAP.get(self.organ, [])
        results = evaluate_folder_against_gt(gt_files, self.out_dir)
        if not results:
            QtWidgets.QMessageBox.information(self, "Evaluation", "No matching GT/prediction pairs found.")
            return
        msg = f"Evaluation ({self.organ} / {self.model}):\n\n"
        for k, m in results.items():
            msg += f"{k}:\n  Dice = {m['Dice']:.3f}\n  IoU = {m['IoU']:.3f}\n\n"
        QtWidgets.QMessageBox.information(self, "Evaluation", msg)

# =============================== APP ===============================
class App(QtWidgets.QMainWindow):
    def __init__(self, case_id: str):
        super().__init__()
        self.setWindowTitle("Medical Image Segmentation — Single File")
        self.resize(980, 640)
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        self.cmb_organ = QtWidgets.QComboBox()
        self.cmb_organ.addItems(["heart", "liver", "lungs"])
        top.addWidget(QtWidgets.QLabel("Organ:"))
        top.addWidget(self.cmb_organ)

        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["TotalSegmentator", "STU_Net"])
        top.addWidget(QtWidgets.QLabel("Model:"))
        top.addWidget(self.cmb_model)

        self.btn_run = QtWidgets.QPushButton("Run Segmentation")
        self.btn_run.clicked.connect(self.on_run)
        top.addWidget(self.btn_run)

        self.chk_rerun = QtWidgets.QCheckBox("Re-run (ignore cache)")
        top.addWidget(self.chk_rerun)

        self.btn_preview = QtWidgets.QPushButton("Save 2D Preview")
        self.btn_preview.clicked.connect(self.on_preview)
        top.addWidget(self.btn_preview)

        self.btn_open_viewer = QtWidgets.QPushButton("Open 3D Viewer")
        self.btn_open_viewer.clicked.connect(self.on_open_viewer)
        top.addWidget(self.btn_open_viewer)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        layout.addWidget(self.txt_log, 1)

        self.case_id = case_id
        self.ct_path = ct_path_from_case(case_id)
        if not self.ct_path.exists():
            self.log(f"[ERR] CT not found: {self.ct_path}")
        else:
            self.log(f"[OK] CT: {self.ct_path}")

        self.viewer: "Viewer | None" = None

    def log(self, s: str):
        self.txt_log.appendPlainText(s)
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def output_dir(self, organ: str, model: str) -> Path:
        return OUTPUT_ROOT / self.case_id / organ / model

    def on_run(self):
        organ = self.cmb_organ.currentText()
        model = self.cmb_model.currentText()
        out_dir = ensure_dir(self.output_dir(organ, model))
        cached = any(p.suffix == ".gz" for p in out_dir.glob("*.nii.gz"))
        if cached and not self.chk_rerun.isChecked():
            self.log(f"✅ Using cached masks: {out_dir}")
            return
        try:
            if model == "TotalSegmentator":
                self.log(f"🚀 Running for {organ}...")
                run_totalseg(organ, self.ct_path, out_dir)
                self.log("✅ model finished.")
            else:  # STU-Net
                self.log(f"🚀 Running for {organ}...")
                run_stunet(organ, self.ct_path, out_dir)
                self.log("✅ model finished.")
        except Exception as e:
            self.log(f"[ERR] {e}")

    def on_preview(self):
        organ = self.cmb_organ.currentText()
        model = self.cmb_model.currentText()
        out_dir = self.output_dir(organ, model)
        if not list(out_dir.glob("*.nii.gz")):
            self.log("⚠️ No predictions found. Run segmentation first.")
            return
        plane = DEFAULT_PLANE.get(organ, "axial")
        png = PREVIEW_ROOT / self.case_id / organ / model / f"preview_{plane}.png"
        try:
            save_2d_preview(self.ct_path, out_dir, png, plane=plane, wl=40, ww=400, alpha=0.35)
            self.log(f"🖼 2D preview saved: {png}")
            try:
                os.startfile(str(png))
            except Exception:
                pass
        except Exception as e:
            self.log(f"[ERR] preview: {e}")

    def on_open_viewer(self):
        organ = self.cmb_organ.currentText()
        model = self.cmb_model.currentText()
        out_dir = self.output_dir(organ, model)
        if not list(out_dir.glob("*.nii.gz")):
            self.log("⚠️ No predictions found. Run segmentation first.")
            return
        try:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer.deleteLater()
        except Exception:
            pass
        self.viewer = None
        self.viewer = Viewer(self.ct_path, organ, model, out_dir)
        self.viewer.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.viewer.show()

# =============================== MAIN ===============================
if __name__ == "__main__":
    case_id = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_CASE_ID
    app = QtWidgets.QApplication(sys.argv)
    win = App(case_id)
    win.show()
    sys.exit(app.exec_())
