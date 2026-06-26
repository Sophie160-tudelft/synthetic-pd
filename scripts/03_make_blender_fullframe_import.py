import argparse
from pathlib import Path
import re

import numpy as np


# =============================================================================
# MAKE BLENDER FULL-FRAME IMPORT COPY
# =============================================================================
# Problem:
#   The SMPL-X Blender add-on may resample 150 fps motion to 30 fps.
#   671 frames at 150 fps becomes about 135 frames at 30 fps.
#
# Solution:
#   Make a Blender-import copy that keeps all 671 poses,
#   but stores fps/mocap_framerate as 30 so the add-on imports every pose
#   as one Blender frame.
#
# IMPORTANT:
#   This is only for Blender import/export.
#   For evaluation/ground truth, keep using the original 150fps file.
# =============================================================================


PROJECT_ROOT = Path(r"C:\Users\sopha\synthetic-pd")

DEFAULT_TRIAL = "SUB01_off_walk_1"

parser = argparse.ArgumentParser(
    description="Create a Blender-import copy that preserves every pose/frame of the selected corrected SMPL-X trial."
)
parser.add_argument("--trial", default=DEFAULT_TRIAL, help="Trial name, e.g. SUB01_off_walk_1")
args = parser.parse_args()

TRIAL = args.trial

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "smplx_fitted_npz"
    / "smpl_to_smplx"
)

OUTPUT_DIR = INPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Find corrected file automatically
# ---------------------------------------------------------------------
pattern = f"{TRIAL}_smplx_fitted_*fps_*frames_*neckHeadHalf.npz"

candidates = []

for path in INPUT_DIR.glob(pattern):
    name = path.name

    # Skip already made Blender-import copies
    if "blenderFullFrames" in name:
        continue

    match = re.search(
        rf"{re.escape(TRIAL)}_smplx_fitted_(\d+)fps_(\d+)frames_.*neckHeadHalf\.npz$",
        name,
    )

    if match is None:
        continue

    fps_from_name = int(match.group(1))
    n_frames = int(match.group(2))
    candidates.append((fps_from_name, n_frames, path))

if not candidates:
    raise FileNotFoundError(
        f"No corrected 150fps neckHeadHalf file found in:\n{INPUT_DIR}\n\n"
        f"Expected pattern:\n{pattern}"
    )

# Prefer highest FPS and then largest frame count
candidates.sort(key=lambda item: (item[0], item[1]))
FPS_FROM_NAME, N_FRAMES, input_npz = candidates[-1]

print("Selected input file:")
print(input_npz)
print("Detected FPS:", FPS_FROM_NAME)
print("Detected frames:", N_FRAMES)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
data = dict(np.load(input_npz, allow_pickle=True))

poses = data["poses"]
trans = data["trans"]

print("\nLoaded data:")
print("poses:", poses.shape)
print("trans:", trans.shape)

if poses.shape[0] != N_FRAMES:
    raise ValueError(
        f"Filename says {N_FRAMES} frames, but poses has {poses.shape[0]} frames."
    )

if poses.shape[1] != 165:
    raise ValueError(f"Expected poses shape (T, 165), got {poses.shape}")


# ---------------------------------------------------------------------
# Create Blender-import version
# ---------------------------------------------------------------------
out = data.copy()

# Keep the actual animation data unchanged
out["poses"] = poses.astype(np.float32)
out["trans"] = trans.astype(np.float32)

if "transl" in out:
    out["transl"] = trans.astype(np.float32)

# Store true/original fps for your own traceability
out["source_fps"] = np.array(FPS_FROM_NAME)
out["true_fps"] = np.array(FPS_FROM_NAME)
out["original_fps"] = np.array(FPS_FROM_NAME)

# Trick the Blender add-on into importing every pose as one frame
out["fps"] = np.array(30)
out["mocap_framerate"] = np.array(30)
out["framerate"] = np.array(30)

output_npz = OUTPUT_DIR / (
    input_npz.stem + "_blenderFullFrames_importAs30fps.npz"
)

np.savez(output_npz, **out)

print("\nSaved Blender full-frame import copy:")
print(output_npz)
print("poses:", poses.shape)
print("trans:", trans.shape)
print("\nUse this file in Blender.")
print("Then set Blender scene FPS to", FPS_FROM_NAME, "and timeline End to", N_FRAMES)