import argparse
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation as R


# =============================================================================
# FINAL SMPL-X CORRECTION SCRIPT — AUTO FPS + AUTO FRAME COUNT
# =============================================================================
# This script automatically finds the raw fitted SMPL-X output:
#
#   <TRIAL>_smplx_fitted_<FPS>fps_<N>frames.npz
#
# It ignores already corrected files and applies:
#   1. root/world rotation
#   2. neckHeadHalf
#
# FPS and number of frames are inferred automatically from the selected file.
# =============================================================================


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\sopha\synthetic-pd")

DEFAULT_TRIAL = "SUB01_off_walk_1"

parser = argparse.ArgumentParser(
    description="Create render-ready corrected SMPL-X file for a selected trial. FPS and frame count are inferred automatically."
)
parser.add_argument("--trial", default=DEFAULT_TRIAL, help="Trial name, e.g. SUB01_off_walk_1")
args = parser.parse_args()

TRIAL = args.trial

# IMPORTANT:
# Use the rotation that currently gives you the correct upright result
# in Unreal/Blender.
#
# Your latest working correction script used +90 degrees.
# If this becomes upside down, change this to:
#   ROOT_ROTATION_DEGREES = (-90, 0, 0)
#   ROOT_ROTATION_LABEL = "rootRot_Xminus90"
ROOT_ROTATION_DEGREES = (90, 0, 0)
ROOT_ROTATION_LABEL = "rootRot_X90"

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "smplx_fitted_npz"
)

OUTPUT_DIR = INPUT_DIR / "smpl_to_smplx"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Find raw fitted input file automatically
# ---------------------------------------------------------------------
# No hardcoded FPS and no hardcoded frame count.
# This will match, for example:
#   SUB01_off_walk_1_smplx_fitted_150fps_671frames.npz
pattern = f"{TRIAL}_smplx_fitted_*fps_*frames.npz"

candidates = []

for path in INPUT_DIR.glob(pattern):
    name = path.name

    # Skip corrected/derived files
    if any(tag in name for tag in ["rootRot", "neckHead", "centered", "headNeutral", "spine80"]):
        continue

    match = re.search(
        rf"{re.escape(TRIAL)}_smplx_fitted_(\d+)fps_(\d+)frames\.npz$",
        name,
    )
    if match is None:
        continue

    fps_from_name = int(match.group(1))
    n_frames_from_name = int(match.group(2))
    candidates.append((fps_from_name, n_frames_from_name, path))

if not candidates:
    raise FileNotFoundError(
        f"No raw fitted file found in:\n{INPUT_DIR}\n\n"
        f"Expected pattern:\n{pattern}\n\n"
        "Make sure you first ran 00_fit_smpl_to_smplx_one_trial.py."
    )

# Prefer the highest FPS and then the largest frame count.
# This will choose the 150fps/671frames file over older 30fps/135frames files.
candidates.sort(key=lambda item: (item[0], item[1]))
FPS_FROM_NAME, N_FRAMES_FROM_NAME, input_npz = candidates[-1]

print("Selected raw fitted input file:")
print(input_npz)
print("FPS from filename:", FPS_FROM_NAME)
print("Frame count from filename:", N_FRAMES_FROM_NAME)


# ---------------------------------------------------------------------
# Load raw fitted SMPL-X data
# ---------------------------------------------------------------------
data = dict(np.load(input_npz, allow_pickle=True))

poses = data["poses"].astype(np.float32).copy()
trans = data["trans"].astype(np.float32).copy()

print("\nLoaded raw fitted file:")
print("poses:", poses.shape)
print("trans:", trans.shape)

if poses.ndim != 2 or poses.shape[1] != 165:
    raise ValueError(f"Expected poses with shape (T, 165), got {poses.shape}")

if trans.ndim != 2 or trans.shape[1] != 3:
    raise ValueError(f"Expected trans with shape (T, 3), got {trans.shape}")

N_FRAMES = poses.shape[0]

if N_FRAMES != N_FRAMES_FROM_NAME:
    raise ValueError(
        f"Frame mismatch: filename says {N_FRAMES_FROM_NAME}, "
        f"but poses has {N_FRAMES} frames."
    )

# Prefer metadata FPS if present, but validate it against filename.
fps_metadata = data.get("fps", data.get("mocap_framerate", FPS_FROM_NAME))
fps_metadata = int(np.asarray(fps_metadata).item())

if fps_metadata != FPS_FROM_NAME:
    raise ValueError(
        f"FPS mismatch: filename says {FPS_FROM_NAME}, "
        f"but metadata says {fps_metadata}."
    )

FPS = FPS_FROM_NAME


# ---------------------------------------------------------------------
# Output path based on detected FPS and frame count
# ---------------------------------------------------------------------
output_npz = OUTPUT_DIR / (
    f"{TRIAL}_smplx_fitted_{FPS}fps_{N_FRAMES}frames_"
    f"{ROOT_ROTATION_LABEL}_neckHeadHalf.npz"
)

print("\nOutput file:")
print(output_npz)


# ---------------------------------------------------------------------
# 1. Apply root/world rotation
# ---------------------------------------------------------------------
# Only rotate:
#   - global/root orientation
#   - translation
#
# Do NOT rotate local body_pose joints.
# ---------------------------------------------------------------------
global_orient = poses[:, 0:3].copy()
rest_pose = poses[:, 3:].copy()

correction = R.from_euler("xyz", ROOT_ROTATION_DEGREES, degrees=True)

new_global = np.zeros_like(global_orient, dtype=np.float32)

for i in range(global_orient.shape[0]):
    root_old = R.from_rotvec(global_orient[i])
    root_new = correction * root_old
    new_global[i] = root_new.as_rotvec().astype(np.float32)

new_trans = correction.apply(trans).astype(np.float32)

poses_corrected = np.concatenate(
    [new_global, rest_pose],
    axis=1,
).astype(np.float32)


# ---------------------------------------------------------------------
# 2. Apply head correction: neckHeadHalf
# ---------------------------------------------------------------------
# SMPL-X pose layout:
#   0:3      global orientation
#   3:66     body_pose = 21 joints * 3
#
# neck = full-pose slice 36:39
# head = full-pose slice 45:48
# ---------------------------------------------------------------------
NECK = slice(36, 39)
HEAD = slice(45, 48)

poses_corrected[:, NECK] *= 0.5
poses_corrected[:, HEAD] *= 0.5


# ---------------------------------------------------------------------
# Save final corrected file
# ---------------------------------------------------------------------
out = data.copy()

out["poses"] = poses_corrected.astype(np.float32)
out["trans"] = new_trans.astype(np.float32)
out["transl"] = new_trans.astype(np.float32)

out["global_orient"] = poses_corrected[:, 0:3].astype(np.float32)
out["body_pose"] = poses_corrected[:, 3:66].astype(np.float32)

out["gender"] = np.array("neutral")
out["mocap_framerate"] = np.array(FPS)
out["fps"] = np.array(FPS)

np.savez(output_npz, **out)

print("\nSaved final corrected SMPL-X animation:")
print(output_npz)
print("poses:", poses_corrected.shape)
print("trans:", new_trans.shape)
print("fps:", FPS)
print("\nDone.")