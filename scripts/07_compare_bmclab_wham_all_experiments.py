"""
Fully standalone batch WHAM-vs-BMCLab metric script.

This file does NOT need a second/base metric script. It contains both:
1. the recommended thesis metric functions, and
2. the all-experiments batch runner.

It finds all WHAM PKL files for one subject's rendered experiments (E0, E1, ...),
computes per-frame and summary metrics for each experiment, and writes one
combined summary CSV/JSON plus per-experiment metric CSVs.

Run from project root:
    python scripts\06_compare_bmclab_wham_all_experiments_FULL_STANDALONE.py --subject SUB03 --trial SUB03_off_walk_1

Useful explicit run:
    python scripts\06_compare_bmclab_wham_all_experiments_FULL_STANDALONE.py ^
        --subject SUB03 ^
        --trial SUB03_off_walk_1 ^
        --wham-pkl-root data\wham_outputs ^
        --fps-override 150
"""

# =============================================================================
# EMBEDDED BASE METRIC FUNCTIONS
# =============================================================================

"""
Compare BMCLab SMPL motion against WHAM output and compute the final
recommended thesis metric set:

A. Pose preservation
   - PA-MPJPE
   - Knee range-of-motion error

B. Gait-cycle / temporal preservation
   - Step/contact timing error
   - Cadence error

C. Local clinical gait preservation
   - Foot clearance error
   - Pelvis-relative step length error
   - Pelvis-relative stride length error

Additional diagnostic metrics are still saved for traceability:
   - Root-aligned MPJPE
   - Walking speed errors
   - Pelvis trajectory error

Important interpretation:
   - MPJPE / PA-MPJPE are computed on raw, unsmoothed joints.
   - Gait features are computed on smoothed joints.
   - Step/stride percentages are saved, but not emphasized, because pelvis-relative
     foot-placement features can have small denominators.
   - Pelvis trajectory error should be treated as diagnostic if global WHAM trajectory
     is not directly comparable to BMCLab.

Expected project structure:
C:/Users/sopha/synthetic-pd/
    data/raw/BMCLab.pkl
    data/wham_outputs/wham_output.pkl
    models/smpl/SMPL_NEUTRAL.pkl
    scripts/15_compare_bmclab_wham_recommended_metrics.py

Run from repo root:
    python scripts/06_compare_bmclab_wham_recommended_metrics_standalone.py

Optional:
    python scripts/06_compare_bmclab_wham_recommended_metrics_standalone.py --prefer-world
    python scripts/06_compare_bmclab_wham_recommended_metrics_standalone.py --fps-override 150
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import torch
from smplx import SMPL


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BMCLAB_PKL = DEFAULT_REPO_ROOT / "data" / "raw" / "BMCLab.pkl"
DEFAULT_WHAM_PKL = DEFAULT_REPO_ROOT / "data" / "wham_outputs" / "wham_output.pkl"
DEFAULT_SMPL_MODEL_DIR = DEFAULT_REPO_ROOT / "models" / "smpl"

DEFAULT_SUBJECT = "SUB01"
DEFAULT_SEQUENCE = "SUB01_off_walk_1"
DEFAULT_WHAM_SUBJECT_ID = 0

DEFAULT_OUT_DIR = None

ROOT_IDX = 0
LEFT_FOOT_IDX = 10
RIGHT_FOOT_IDX = 11

LEFT_HIP_IDX = 1
RIGHT_HIP_IDX = 2
LEFT_KNEE_IDX = 4
RIGHT_KNEE_IDX = 5
LEFT_ANKLE_IDX = 7
RIGHT_ANKLE_IDX = 8

UNIT_SCALE = 1000.0
SMOOTHING_WINDOW = 7


# =============================================================================
# Loading helpers
# =============================================================================

def load_pickle_any(path):
    path = Path(path)

    try:
        return joblib.load(path)
    except Exception:
        pass

    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def inspect_top_level(obj, name):
    print(f"\n{name} type: {type(obj)}")

    if isinstance(obj, dict):
        print(f"{name} keys:")
        for k in list(obj.keys())[:20]:
            print(f"  {repr(k)}")


def get_bmclab_sequence(bmclab_data, subject, sequence):
    if isinstance(bmclab_data, dict):
        if subject in bmclab_data:
            subj_data = bmclab_data[subject]

            if isinstance(subj_data, dict) and sequence in subj_data:
                return subj_data[sequence]

        if sequence in bmclab_data:
            return bmclab_data[sequence]

        if (subject, sequence) in bmclab_data:
            return bmclab_data[(subject, sequence)]

    raise KeyError(
        f"Could not find subject={subject!r}, sequence={sequence!r} in BMCLab.pkl."
    )


def extract_bmclab_pose_trans_beta(seq_data):
    if not isinstance(seq_data, dict):
        raise TypeError(f"Expected BMCLab sequence data to be dict, got {type(seq_data)}")

    pose_keys = ["pose", "poses", "body_pose"]
    trans_keys = ["trans", "transl", "translation", "translations"]
    beta_keys = ["beta", "betas", "shape"]

    pose = None
    trans = None
    beta = None
    fps = None

    for key in pose_keys:
        if key in seq_data:
            pose = np.asarray(seq_data[key])
            break

    for key in trans_keys:
        if key in seq_data:
            trans = np.asarray(seq_data[key])
            break

    for key in beta_keys:
        if key in seq_data:
            beta = np.asarray(seq_data[key])
            break

    if "fps" in seq_data:
        fps = int(np.asarray(seq_data["fps"]).item())
    elif "mocap_framerate" in seq_data:
        fps = int(np.asarray(seq_data["mocap_framerate"]).item())

    if pose is None:
        raise KeyError(f"Could not find pose key. Available keys: {list(seq_data.keys())}")

    if trans is None:
        raise KeyError(f"Could not find trans key. Available keys: {list(seq_data.keys())}")

    if beta is None:
        raise KeyError(f"Could not find beta/betas key. Available keys: {list(seq_data.keys())}")

    pose = pose.astype(np.float32)
    trans = trans.astype(np.float32)
    beta = beta.astype(np.float32)

    if beta.ndim == 1:
        beta = beta[None, :]

    return pose, trans, beta, fps


def get_wham_subject(wham_data, subject_id):
    if subject_id in wham_data:
        return wham_data[subject_id]

    subject_id_str = str(subject_id)
    if subject_id_str in wham_data:
        return wham_data[subject_id_str]

    raise KeyError(
        f"Could not find WHAM subject {subject_id}. "
        f"Available WHAM keys: {list(wham_data.keys())}"
    )


def extract_wham_pose_trans_beta(wham_subject, prefer_world=False, beta_mode="mean"):
    if prefer_world and "pose_world" in wham_subject and "trans_world" in wham_subject:
        pose_key = "pose_world"
        trans_key = "trans_world"
    else:
        pose_key = "pose"
        trans_key = "trans"

    pose = np.asarray(wham_subject[pose_key]).astype(np.float32)
    trans = np.asarray(wham_subject[trans_key]).astype(np.float32)

    if "betas" in wham_subject:
        betas = np.asarray(wham_subject["betas"]).astype(np.float32)
    elif "beta" in wham_subject:
        betas = np.asarray(wham_subject["beta"]).astype(np.float32)
    else:
        raise KeyError(f"Could not find betas in WHAM subject. Keys: {list(wham_subject.keys())}")

    if betas.ndim == 1:
        beta = betas[None, :]
    elif betas.ndim == 2:
        if beta_mode == "mean":
            beta = betas.mean(axis=0, keepdims=True)
        elif beta_mode == "first":
            beta = betas[:1]
        else:
            raise ValueError(f"Unknown beta_mode: {beta_mode}")
    else:
        raise ValueError(f"Unexpected WHAM betas shape: {betas.shape}")

    return pose, trans, beta, pose_key, trans_key


# =============================================================================
# SMPL reconstruction
# =============================================================================

def build_smpl_model(model_dir, device):
    model = SMPL(
        model_path=str(model_dir),
        gender="neutral",
        batch_size=1,
        create_transl=True,
    ).to(device)

    model.eval()
    return model


def smpl_joints_from_pose_trans_beta(
    smpl_model,
    pose,
    trans,
    beta,
    device,
    batch_size=128,
):
    pose = np.asarray(pose, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    if pose.ndim != 2 or pose.shape[1] != 72:
        raise ValueError(f"Expected pose shape (F, 72), got {pose.shape}")

    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Expected trans shape (F, 3), got {trans.shape}")

    num_frames = pose.shape[0]

    if trans.shape[0] != num_frames:
        raise ValueError(
            f"pose has {num_frames} frames but trans has {trans.shape[0]} frames"
        )

    if beta.ndim == 1:
        beta = beta[None, :]

    if beta.shape[0] == 1:
        beta_full = np.repeat(beta, num_frames, axis=0)
    elif beta.shape[0] == num_frames:
        beta_full = beta
    else:
        raise ValueError(f"beta should have shape (1, 10) or (F, 10), got {beta.shape}")

    all_joints = []

    with torch.no_grad():
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)

            pose_batch = torch.tensor(pose[start:end], dtype=torch.float32, device=device)
            trans_batch = torch.tensor(trans[start:end], dtype=torch.float32, device=device)
            beta_batch = torch.tensor(beta_full[start:end], dtype=torch.float32, device=device)

            global_orient = pose_batch[:, 0:3]
            body_pose = pose_batch[:, 3:72]

            output = smpl_model(
                betas=beta_batch,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=trans_batch,
                return_verts=False,
            )

            joints = output.joints[:, :24, :].detach().cpu().numpy()
            all_joints.append(joints)

    return np.concatenate(all_joints, axis=0).astype(np.float32)


# =============================================================================
# Technical pose metrics
# =============================================================================

def root_align(joints, root_idx=0):
    return joints - joints[:, root_idx:root_idx + 1, :]


def mpjpe_per_frame(pred, gt, unit_scale=1000.0):
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=1) * unit_scale


def compute_root_aligned_mpjpe(pred_joints, gt_joints, root_idx=0, unit_scale=1000.0):
    pred_ra = root_align(pred_joints, root_idx=root_idx)
    gt_ra = root_align(gt_joints, root_idx=root_idx)
    errors = mpjpe_per_frame(pred_ra, gt_ra, unit_scale=unit_scale)

    return {
        "root_aligned_mpjpe_mean_mm": float(errors.mean()),
        "root_aligned_mpjpe_std_mm": float(errors.std()),
        "root_aligned_mpjpe_median_mm": float(np.median(errors)),
        "root_aligned_mpjpe_min_mm": float(errors.min()),
        "root_aligned_mpjpe_max_mm": float(errors.max()),
    }, errors


def batch_compute_similarity_transform(source, target):
    num_frames, num_joints, _ = source.shape
    aligned = np.zeros_like(source)

    for frame in range(num_frames):
        x = source[frame]
        y = target[frame]

        mu_x = x.mean(axis=0, keepdims=True)
        mu_y = y.mean(axis=0, keepdims=True)

        x0 = x - mu_x
        y0 = y - mu_y

        norm_x = np.sqrt((x0 ** 2).sum())
        norm_y = np.sqrt((y0 ** 2).sum())

        if norm_x < 1e-8 or norm_y < 1e-8:
            aligned[frame] = x
            continue

        x0n = x0 / norm_x
        y0n = y0 / norm_y

        h = x0n.T @ y0n
        u, s, vt = np.linalg.svd(h)

        # Row-vector convention: x @ r ≈ y
        r = u @ vt

        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = u @ vt

        scale = norm_y / norm_x * s.sum()
        translation = mu_y - scale * (mu_x @ r)

        aligned[frame] = scale * (x @ r) + translation

    return aligned


def compute_pa_mpjpe(pred_joints, gt_joints, unit_scale=1000.0):
    pred_pa = batch_compute_similarity_transform(pred_joints, gt_joints)
    errors = mpjpe_per_frame(pred_pa, gt_joints, unit_scale=unit_scale)

    return {
        "pa_mpjpe_mean_mm": float(errors.mean()),
        "pa_mpjpe_std_mm": float(errors.std()),
        "pa_mpjpe_median_mm": float(np.median(errors)),
        "pa_mpjpe_min_mm": float(errors.min()),
        "pa_mpjpe_max_mm": float(errors.max()),
    }, errors


# =============================================================================
# Gait-coordinate helpers
# =============================================================================

def infer_vertical_axis_from_joints(joints):
    extents = np.ptp(joints, axis=1)
    mean_extents = extents.mean(axis=0)
    vertical_axis = int(np.argmax(mean_extents))
    return vertical_axis, mean_extents


def compute_forward_axis_from_pelvis(gt_joints, vertical_axis, root_idx=0):
    """
    Defines the forward/progression axis used for step/stride projection.

    Preferred method:
        Use horizontal pelvis displacement from first to last frame.

    Fallback method:
        If pelvis displacement is too small, for example after root alignment,
        infer the progression axis from the horizontal axis with the largest
        combined left/right foot excursion. This is appropriate for local
        gait metrics such as pelvis-relative step length, stride length, and
        foot clearance, where the pelvis may intentionally be fixed at zero.
    """
    joints = np.asarray(gt_joints, dtype=np.float64)
    pelvis = joints[:, root_idx, :]

    direction = pelvis[-1] - pelvis[0]
    direction = direction.astype(np.float64)
    direction[vertical_axis] = 0.0

    norm = np.linalg.norm(direction)

    if norm >= 1e-6:
        return (direction / norm).astype(np.float32)

    # Fallback for root-aligned sequences: infer forward axis from foot excursion.
    ground_axes = [axis for axis in range(3) if axis != vertical_axis]

    try:
        feet = joints[:, [LEFT_FOOT_IDX, RIGHT_FOOT_IDX], :].reshape(-1, 3)
        ranges = np.ptp(feet[:, ground_axes], axis=0)
        best_ground_axis = ground_axes[int(np.argmax(ranges))]

        if np.max(ranges) < 1e-8:
            raise ValueError("Foot excursion is also too small.")

        forward = np.zeros(3, dtype=np.float64)
        forward[best_ground_axis] = 1.0

        # Assign a stable sign from mean foot displacement if possible.
        mean_foot_start = joints[0, [LEFT_FOOT_IDX, RIGHT_FOOT_IDX], :].mean(axis=0)
        mean_foot_end = joints[-1, [LEFT_FOOT_IDX, RIGHT_FOOT_IDX], :].mean(axis=0)
        sign_source = mean_foot_end[best_ground_axis] - mean_foot_start[best_ground_axis]
        if abs(sign_source) > 1e-8:
            forward[best_ground_axis] = np.sign(sign_source)

        return forward.astype(np.float32)

    except Exception as error:
        ground_axis = ground_axes[0]
        forward = np.zeros(3, dtype=np.float32)
        forward[ground_axis] = 1.0
        print(
            "WARNING: Could not infer walking direction from pelvis or feet. "
            f"Using coordinate axis {ground_axis} as fallback. Original error: {error}"
        )
        return forward


def compute_lateral_axis(forward_axis, vertical_axis):
    vertical = np.zeros(3, dtype=np.float32)
    vertical[vertical_axis] = 1.0

    lateral = np.cross(vertical, forward_axis)
    norm = np.linalg.norm(lateral)

    if norm < 1e-8:
        raise ValueError("Could not define lateral axis.")

    return (lateral / norm).astype(np.float32)


def yaw_align_wham_to_gt_pelvis(gt_joints, wham_joints, vertical_axis, root_idx=0):
    """
    Fits one horizontal yaw rotation to align the WHAM pelvis path to the BMCLab pelvis path.
    This does not apply scaling.
    """
    gt_pelvis = gt_joints[:, root_idx, :]
    wham_pelvis = wham_joints[:, root_idx, :]

    gt0 = gt_pelvis - gt_pelvis[0]
    wham0 = wham_pelvis - wham_pelvis[0]

    ground_axes = [axis for axis in range(3) if axis != vertical_axis]

    x = wham0[:, ground_axes].astype(np.float64)
    y = gt0[:, ground_axes].astype(np.float64)

    h = x.T @ y
    u, s, vt = np.linalg.svd(h)
    # Row-vector convention: x @ r2 ≈ y
    r2 = u @ vt

    if np.linalg.det(r2) < 0:
        vt[-1, :] *= -1
        r2 = u @ vt

    wham_aligned = wham_joints.copy().astype(np.float32)

    origin = wham_pelvis[0].copy()
    centered = wham_aligned - origin[None, None, :]

    flat = centered.reshape(-1, 3)
    flat_ground = flat[:, ground_axes] @ r2
    flat[:, ground_axes] = flat_ground

    aligned = flat.reshape(wham_aligned.shape)
    aligned = aligned + gt_pelvis[0][None, None, :]

    r3 = np.eye(3, dtype=np.float32)
    for i, axis_i in enumerate(ground_axes):
        for j, axis_j in enumerate(ground_axes):
            r3[axis_i, axis_j] = r2[i, j]

    return aligned.astype(np.float32), r3


# =============================================================================
# Smoothing
# =============================================================================

def moving_average_trajectory(x, window=7):
    """
    Centered moving-average smoothing.

    Used only for gait-feature extraction.
    MPJPE and PA-MPJPE stay on raw unsmoothed joints.
    """
    if window <= 1:
        return x

    if window % 2 == 0:
        raise ValueError("Smoothing window must be odd, for example 5, 7, or 9.")

    pad = window // 2
    pad_width = [(pad, pad)] + [(0, 0)] * (x.ndim - 1)

    x_pad = np.pad(x, pad_width, mode="edge")
    out = np.zeros_like(x)

    for i in range(x.shape[0]):
        out[i] = x_pad[i:i + window].mean(axis=0)

    return out


# =============================================================================
# Walking speed metrics
# =============================================================================

def compute_displacement_walking_speed(joints, forward_axis, fps, root_idx=0):
    """
    Computes straight-line walking speed along the original BMCLab walking direction.
    """
    pelvis = joints[:, root_idx, :]
    projected = pelvis @ forward_axis

    duration = (len(pelvis) - 1) / float(fps)

    if duration <= 0:
        raise ValueError("Invalid duration. Check fps and number of frames.")

    speed = abs((projected[-1] - projected[0]) / duration)
    return float(speed)


def compute_path_based_walking_speed(joints, fps, root_idx=0):
    """
    Computes pelvis path speed as total pelvis path length divided by duration.
    This is less sensitive to global direction mismatch, but can be inflated by jitter.
    Therefore it is computed on smoothed joints.
    """
    pelvis = joints[:, root_idx, :]

    frame_displacements = np.linalg.norm(
        pelvis[1:] - pelvis[:-1],
        axis=1,
    )

    path_length = frame_displacements.sum()
    duration = (len(pelvis) - 1) / float(fps)

    if duration <= 0:
        raise ValueError("Invalid duration. Check fps and number of frames.")

    speed = path_length / duration
    return float(speed)


def compute_walking_speed_metrics(
    gt_joints,
    wham_joints_raw,
    wham_joints_aligned,
    forward_axis,
    fps,
    root_idx=0,
):
    gt_disp_speed = compute_displacement_walking_speed(
        gt_joints, forward_axis, fps, root_idx=root_idx
    )
    wham_disp_speed_raw = compute_displacement_walking_speed(
        wham_joints_raw, forward_axis, fps, root_idx=root_idx
    )
    wham_disp_speed_aligned = compute_displacement_walking_speed(
        wham_joints_aligned, forward_axis, fps, root_idx=root_idx
    )

    gt_path_speed = compute_path_based_walking_speed(
        gt_joints, fps, root_idx=root_idx
    )
    wham_path_speed_raw = compute_path_based_walking_speed(
        wham_joints_raw, fps, root_idx=root_idx
    )
    wham_path_speed_aligned = compute_path_based_walking_speed(
        wham_joints_aligned, fps, root_idx=root_idx
    )

    disp_raw_abs_error = abs(wham_disp_speed_raw - gt_disp_speed)
    disp_aligned_abs_error = abs(wham_disp_speed_aligned - gt_disp_speed)

    path_raw_abs_error = abs(wham_path_speed_raw - gt_path_speed)
    path_aligned_abs_error = abs(wham_path_speed_aligned - gt_path_speed)

    disp_raw_pct_error = (
        disp_raw_abs_error / abs(gt_disp_speed) * 100.0
        if abs(gt_disp_speed) > 1e-8
        else None
    )
    disp_aligned_pct_error = (
        disp_aligned_abs_error / abs(gt_disp_speed) * 100.0
        if abs(gt_disp_speed) > 1e-8
        else None
    )

    path_raw_pct_error = (
        path_raw_abs_error / abs(gt_path_speed) * 100.0
        if abs(gt_path_speed) > 1e-8
        else None
    )
    path_aligned_pct_error = (
        path_aligned_abs_error / abs(gt_path_speed) * 100.0
        if abs(gt_path_speed) > 1e-8
        else None
    )

    return {
        "walking_speed_displacement_original_m_per_s": float(gt_disp_speed),
        "walking_speed_displacement_wham_raw_m_per_s": float(wham_disp_speed_raw),
        "walking_speed_displacement_raw_abs_error_m_per_s": float(disp_raw_abs_error),
        "walking_speed_displacement_raw_pct_error": (
            float(disp_raw_pct_error) if disp_raw_pct_error is not None else None
        ),
        "walking_speed_displacement_wham_aligned_m_per_s": float(wham_disp_speed_aligned),
        "walking_speed_displacement_aligned_abs_error_m_per_s": float(disp_aligned_abs_error),
        "walking_speed_displacement_aligned_pct_error": (
            float(disp_aligned_pct_error) if disp_aligned_pct_error is not None else None
        ),

        "walking_speed_path_original_m_per_s": float(gt_path_speed),
        "walking_speed_path_wham_raw_m_per_s": float(wham_path_speed_raw),
        "walking_speed_path_raw_abs_error_m_per_s": float(path_raw_abs_error),
        "walking_speed_path_raw_pct_error": (
            float(path_raw_pct_error) if path_raw_pct_error is not None else None
        ),
        "walking_speed_path_wham_aligned_m_per_s": float(wham_path_speed_aligned),
        "walking_speed_path_aligned_abs_error_m_per_s": float(path_aligned_abs_error),
        "walking_speed_path_aligned_pct_error": (
            float(path_aligned_pct_error) if path_aligned_pct_error is not None else None
        ),
    }


# =============================================================================
# Pelvis trajectory metrics
# =============================================================================

def compute_pelvis_trajectory_metrics(
    gt_joints,
    wham_joints_aligned,
    forward_axis,
    lateral_axis,
    vertical_axis,
    root_idx=0,
    unit_scale=1000.0,
):
    gt_pelvis = gt_joints[:, root_idx, :]
    wham_pelvis = wham_joints_aligned[:, root_idx, :]

    gt0 = gt_pelvis - gt_pelvis[0]
    wham0 = wham_pelvis - wham_pelvis[0]

    diff = wham0 - gt0

    euclidean_errors_mm = np.linalg.norm(diff, axis=1) * unit_scale
    forward_errors_mm = np.abs(diff @ forward_axis) * unit_scale
    lateral_errors_mm = np.abs(diff @ lateral_axis) * unit_scale
    vertical_errors_mm = np.abs(diff[:, vertical_axis]) * unit_scale

    summary = {
        "pelvis_trajectory_error_mean_mm": float(euclidean_errors_mm.mean()),
        "pelvis_trajectory_error_std_mm": float(euclidean_errors_mm.std()),
        "pelvis_trajectory_error_median_mm": float(np.median(euclidean_errors_mm)),
        "pelvis_trajectory_error_min_mm": float(euclidean_errors_mm.min()),
        "pelvis_trajectory_error_max_mm": float(euclidean_errors_mm.max()),

        "pelvis_forward_error_mean_mm": float(forward_errors_mm.mean()),
        "pelvis_lateral_error_mean_mm": float(lateral_errors_mm.mean()),
        "pelvis_vertical_error_mean_mm": float(vertical_errors_mm.mean()),
    }

    per_frame = {
        "pelvis_trajectory_error_mm": euclidean_errors_mm,
        "pelvis_forward_error_mm": forward_errors_mm,
        "pelvis_lateral_error_mm": lateral_errors_mm,
        "pelvis_vertical_error_mm": vertical_errors_mm,
    }

    return summary, per_frame


# =============================================================================
# Step and stride metrics
# =============================================================================

def compute_foot_velocity(foot_positions, fps):
    velocity = np.zeros_like(foot_positions)
    velocity[1:] = (foot_positions[1:] - foot_positions[:-1]) * float(fps)
    speed = np.linalg.norm(velocity, axis=1)
    return speed


def detect_contact_events(
    joints,
    foot_idx,
    vertical_axis,
    fps,
    height_quantile=0.35,
    velocity_quantile=0.35,
    min_separation_s=0.45,
):
    """
    Kinematic foot-contact approximation.

    A foot-contact event is detected when the foot is relatively low and relatively slow.
    This is not a force-plate heel strike; it is a consistent approximation for comparing
    original and recovered motion.
    """
    foot = joints[:, foot_idx, :]

    height = foot[:, vertical_axis]
    speed = compute_foot_velocity(foot, fps)

    height_threshold = np.quantile(height, height_quantile)
    velocity_threshold = np.quantile(speed, velocity_quantile)

    contact = (height <= height_threshold) & (speed <= velocity_threshold)

    starts = []
    in_contact = False

    for frame_idx, is_contact in enumerate(contact):
        if is_contact and not in_contact:
            starts.append(frame_idx)
            in_contact = True
        elif not is_contact:
            in_contact = False

    min_sep_frames = max(1, int(round(min_separation_s * fps)))

    filtered = []
    for frame in starts:
        if not filtered or frame - filtered[-1] >= min_sep_frames:
            filtered.append(frame)

    return {
        "events": filtered,
        "height_threshold": float(height_threshold),
        "velocity_threshold": float(velocity_threshold),
        "num_contact_frames": int(contact.sum()),
        "min_separation_s": float(min_separation_s),
        "min_separation_frames": int(min_sep_frames),
    }


def build_gait_events(joints, forward_axis, vertical_axis, fps, pelvis_relative=True):
    left = detect_contact_events(
        joints=joints,
        foot_idx=LEFT_FOOT_IDX,
        vertical_axis=vertical_axis,
        fps=fps,
    )

    right = detect_contact_events(
        joints=joints,
        foot_idx=RIGHT_FOOT_IDX,
        vertical_axis=vertical_axis,
        fps=fps,
    )

    events = []

    for frame in left["events"]:
        foot_pos = joints[frame, LEFT_FOOT_IDX, :]
        pelvis_pos = joints[frame, ROOT_IDX, :]

        if pelvis_relative:
            pos_for_projection = foot_pos - pelvis_pos
        else:
            pos_for_projection = foot_pos

        events.append({
            "frame": int(frame),
            "time_s": float(frame / fps),
            "side": "L",
            "forward_position_m": float(pos_for_projection @ forward_axis),
        })

    for frame in right["events"]:
        foot_pos = joints[frame, RIGHT_FOOT_IDX, :]
        pelvis_pos = joints[frame, ROOT_IDX, :]

        if pelvis_relative:
            pos_for_projection = foot_pos - pelvis_pos
        else:
            pos_for_projection = foot_pos

        events.append({
            "frame": int(frame),
            "time_s": float(frame / fps),
            "side": "R",
            "forward_position_m": float(pos_for_projection @ forward_axis),
        })

    events = sorted(events, key=lambda event: event["frame"])

    return events, {
        "left_contact_detection": left,
        "right_contact_detection": right,
        "pelvis_relative_foot_placement": bool(pelvis_relative),
    }


def compute_step_lengths_from_events(events):
    steps = []

    for prev_event, curr_event in zip(events[:-1], events[1:]):
        if prev_event["side"] == curr_event["side"]:
            continue

        length = abs(curr_event["forward_position_m"] - prev_event["forward_position_m"])

        steps.append({
            "from_side": prev_event["side"],
            "to_side": curr_event["side"],
            "start_frame": int(prev_event["frame"]),
            "end_frame": int(curr_event["frame"]),
            "length_m": float(length),
        })

    return steps


def compute_stride_lengths_from_events(events):
    strides = []

    by_side = {"L": [], "R": []}
    for event in events:
        by_side[event["side"]].append(event)

    for side in ["L", "R"]:
        side_events = by_side[side]

        for prev_event, curr_event in zip(side_events[:-1], side_events[1:]):
            length = abs(curr_event["forward_position_m"] - prev_event["forward_position_m"])

            strides.append({
                "side": side,
                "start_frame": int(prev_event["frame"]),
                "end_frame": int(curr_event["frame"]),
                "length_m": float(length),
            })

    strides = sorted(strides, key=lambda event: event["start_frame"])
    return strides


def summarize_lengths(lengths, prefix):
    values = np.array([item["length_m"] for item in lengths], dtype=np.float64)

    if len(values) == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean_m": None,
            f"{prefix}_std_m": None,
            f"{prefix}_median_m": None,
            f"{prefix}_min_m": None,
            f"{prefix}_max_m": None,
        }

    return {
        f"{prefix}_count": int(len(values)),
        f"{prefix}_mean_m": float(values.mean()),
        f"{prefix}_std_m": float(values.std()),
        f"{prefix}_median_m": float(np.median(values)),
        f"{prefix}_min_m": float(values.min()),
        f"{prefix}_max_m": float(values.max()),
    }


def compare_length_means(gt_lengths, wham_lengths, prefix):
    gt_values = np.array([item["length_m"] for item in gt_lengths], dtype=np.float64)
    wham_values = np.array([item["length_m"] for item in wham_lengths], dtype=np.float64)

    if len(gt_values) == 0 or len(wham_values) == 0:
        return {
            f"{prefix}_mean_abs_error_m": None,
            f"{prefix}_mean_abs_error_mm": None,
            f"{prefix}_mean_pct_error": None,
            f"{prefix}_paired_count_used": 0,
            f"{prefix}_paired_mae_m": None,
            f"{prefix}_paired_mae_mm": None,
            f"{prefix}_event_count_difference": int(len(wham_values) - len(gt_values)),
        }

    gt_mean = float(gt_values.mean())
    wham_mean = float(wham_values.mean())

    abs_error_m = abs(wham_mean - gt_mean)
    pct_error = abs_error_m / abs(gt_mean) * 100.0 if abs(gt_mean) > 1e-8 else None

    paired_n = min(len(gt_values), len(wham_values))
    paired_errors = np.abs(wham_values[:paired_n] - gt_values[:paired_n])

    return {
        f"{prefix}_mean_abs_error_m": float(abs_error_m),
        f"{prefix}_mean_abs_error_mm": float(abs_error_m * 1000.0),
        f"{prefix}_mean_pct_error": float(pct_error) if pct_error is not None else None,
        f"{prefix}_paired_count_used": int(paired_n),
        f"{prefix}_paired_mae_m": float(paired_errors.mean()) if paired_n > 0 else None,
        f"{prefix}_paired_mae_mm": float(paired_errors.mean() * 1000.0) if paired_n > 0 else None,
        f"{prefix}_event_count_difference": int(len(wham_values) - len(gt_values)),
    }


def compute_step_stride_metrics(gt_joints, wham_joints_aligned, forward_axis, vertical_axis, fps):
    gt_events, gt_contact_info = build_gait_events(
        joints=gt_joints,
        forward_axis=forward_axis,
        vertical_axis=vertical_axis,
        fps=fps,
        pelvis_relative=True,
    )

    wham_events, wham_contact_info = build_gait_events(
        joints=wham_joints_aligned,
        forward_axis=forward_axis,
        vertical_axis=vertical_axis,
        fps=fps,
        pelvis_relative=True,
    )

    gt_steps = compute_step_lengths_from_events(gt_events)
    wham_steps = compute_step_lengths_from_events(wham_events)

    gt_strides = compute_stride_lengths_from_events(gt_events)
    wham_strides = compute_stride_lengths_from_events(wham_events)

    summary = {
        "gt_num_gait_events": int(len(gt_events)),
        "wham_num_gait_events": int(len(wham_events)),
        "gait_event_count_difference": int(len(wham_events) - len(gt_events)),

        **summarize_lengths(gt_steps, "gt_step_length"),
        **summarize_lengths(wham_steps, "wham_step_length"),
        **compare_length_means(gt_steps, wham_steps, "step_length"),

        **summarize_lengths(gt_strides, "gt_stride_length"),
        **summarize_lengths(wham_strides, "wham_stride_length"),
        **compare_length_means(gt_strides, wham_strides, "stride_length"),

        "contact_detection": {
            "gt": gt_contact_info,
            "wham": wham_contact_info,
        },
    }

    details = {
        "gt_events": gt_events,
        "wham_events": wham_events,
        "gt_steps": gt_steps,
        "wham_steps": wham_steps,
        "gt_strides": gt_strides,
        "wham_strides": wham_strides,
    }

    return summary, details



# =============================================================================
# Recommended clinical / motion-preservation metrics
# =============================================================================

def angle_between_vectors_deg(a, b, eps=1e-8):
    """
    Returns the angle between vectors a and b in degrees.

    a, b: (..., 3)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosang = np.sum(a * b, axis=-1) / np.maximum(denom, eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def compute_knee_angles_deg(joints):
    """
    Computes left and right knee flexion-extension angles from hip-knee-ankle triplets.

    The returned angle is the anatomical angle at the knee. A smaller/larger absolute
    convention is less important here than preserving the same convention for BMCLab
    and WHAM; the ROM comparison is therefore meaningful even if the absolute angle
    convention differs from a clinical goniometer convention.
    """
    left_thigh = joints[:, LEFT_HIP_IDX, :] - joints[:, LEFT_KNEE_IDX, :]
    left_shank = joints[:, LEFT_ANKLE_IDX, :] - joints[:, LEFT_KNEE_IDX, :]

    right_thigh = joints[:, RIGHT_HIP_IDX, :] - joints[:, RIGHT_KNEE_IDX, :]
    right_shank = joints[:, RIGHT_ANKLE_IDX, :] - joints[:, RIGHT_KNEE_IDX, :]

    left_angle = angle_between_vectors_deg(left_thigh, left_shank)
    right_angle = angle_between_vectors_deg(right_thigh, right_shank)

    return {
        "left_knee_angle_deg": left_angle.astype(np.float32),
        "right_knee_angle_deg": right_angle.astype(np.float32),
    }


def compute_knee_rom_metrics(gt_joints, wham_joints):
    gt_angles = compute_knee_angles_deg(gt_joints)
    wham_angles = compute_knee_angles_deg(wham_joints)

    summary = {}
    per_frame = {}
    rom_errors = []

    for side in ["left", "right"]:
        key = f"{side}_knee_angle_deg"
        gt = gt_angles[key]
        wham = wham_angles[key]

        gt_rom = float(np.max(gt) - np.min(gt))
        wham_rom = float(np.max(wham) - np.min(wham))
        error = abs(wham_rom - gt_rom)
        pct_error = error / abs(gt_rom) * 100.0 if abs(gt_rom) > 1e-8 else None

        summary[f"gt_{side}_knee_rom_deg"] = gt_rom
        summary[f"wham_{side}_knee_rom_deg"] = wham_rom
        summary[f"{side}_knee_rom_abs_error_deg"] = float(error)
        summary[f"{side}_knee_rom_pct_error"] = float(pct_error) if pct_error is not None else None

        per_frame[f"gt_{side}_knee_angle_deg"] = gt
        per_frame[f"wham_{side}_knee_angle_deg"] = wham
        per_frame[f"{side}_knee_angle_abs_error_deg"] = np.abs(wham - gt)

        rom_errors.append(error)

    summary["knee_rom_mean_abs_error_deg"] = float(np.mean(rom_errors))
    summary["knee_rom_max_abs_error_deg"] = float(np.max(rom_errors))

    return summary, per_frame


def pair_events_by_side_and_order(gt_events, wham_events):
    """
    Pairs contact events by foot side and chronological order.

    This intentionally avoids nearest-neighbour matching across sides, because the aim is
    not to hide missed or extra contacts. Event count differences are reported separately.
    """
    pairs = []

    for side in ["L", "R"]:
        gt_side = [event for event in gt_events if event["side"] == side]
        wham_side = [event for event in wham_events if event["side"] == side]
        n = min(len(gt_side), len(wham_side))

        for i in range(n):
            pairs.append((gt_side[i], wham_side[i]))

    pairs = sorted(pairs, key=lambda pair: pair[0]["frame"])
    return pairs


def compute_contact_timing_metrics(gt_events, wham_events, fps):
    pairs = pair_events_by_side_and_order(gt_events, wham_events)

    if len(pairs) == 0:
        return {
            "step_contact_timing_paired_count": 0,
            "step_contact_timing_mae_s": None,
            "step_contact_timing_mae_frames": None,
            "step_contact_timing_mean_signed_error_s": None,
            "step_contact_timing_max_abs_error_s": None,
            "left_contact_event_count_difference": len([e for e in wham_events if e["side"] == "L"]) - len([e for e in gt_events if e["side"] == "L"]),
            "right_contact_event_count_difference": len([e for e in wham_events if e["side"] == "R"]) - len([e for e in gt_events if e["side"] == "R"]),
        }

    frame_errors = np.array(
        [pair[1]["frame"] - pair[0]["frame"] for pair in pairs],
        dtype=np.float64,
    )
    time_errors = frame_errors / float(fps)

    return {
        "step_contact_timing_paired_count": int(len(pairs)),
        "step_contact_timing_mae_s": float(np.mean(np.abs(time_errors))),
        "step_contact_timing_mae_frames": float(np.mean(np.abs(frame_errors))),
        "step_contact_timing_mean_signed_error_s": float(np.mean(time_errors)),
        "step_contact_timing_max_abs_error_s": float(np.max(np.abs(time_errors))),
        "left_contact_event_count_difference": int(len([e for e in wham_events if e["side"] == "L"]) - len([e for e in gt_events if e["side"] == "L"])),
        "right_contact_event_count_difference": int(len([e for e in wham_events if e["side"] == "R"]) - len([e for e in gt_events if e["side"] == "R"])),
    }


def compute_cadence_from_events(events, n_frames, fps):
    duration_min = ((n_frames - 1) / float(fps)) / 60.0
    if duration_min <= 0:
        raise ValueError("Invalid duration for cadence computation.")
    return len(events) / duration_min


def compute_cadence_metrics(gt_events, wham_events, n_frames, fps):
    gt_cadence = compute_cadence_from_events(gt_events, n_frames, fps)
    wham_cadence = compute_cadence_from_events(wham_events, n_frames, fps)
    abs_error = abs(wham_cadence - gt_cadence)
    pct_error = abs_error / abs(gt_cadence) * 100.0 if abs(gt_cadence) > 1e-8 else None

    return {
        "gt_cadence_steps_per_min": float(gt_cadence),
        "wham_cadence_steps_per_min": float(wham_cadence),
        "cadence_abs_error_steps_per_min": float(abs_error),
        "cadence_pct_error": float(pct_error) if pct_error is not None else None,
    }


def swing_clearances_for_side(joints, events, foot_idx, side, vertical_axis, unit_scale=1000.0):
    """
    Estimates swing foot clearance from same-foot contact-to-contact intervals.

    For each interval between two contacts of the same foot, the script takes the maximum
    vertical foot height in the interval minus the lower of the two contact heights. This
    gives a consistent video/pose-based proxy for foot lift without requiring force plates.
    """
    side_events = [event for event in events if event["side"] == side]
    foot_z = joints[:, foot_idx, vertical_axis]

    clearances_mm = []
    intervals = []

    for start_event, end_event in zip(side_events[:-1], side_events[1:]):
        start = int(start_event["frame"])
        end = int(end_event["frame"])

        if end <= start + 1:
            continue

        segment = foot_z[start:end + 1]
        contact_reference = min(foot_z[start], foot_z[end])
        clearance = (np.max(segment) - contact_reference) * unit_scale

        clearances_mm.append(float(max(clearance, 0.0)))
        intervals.append({
            "side": side,
            "start_frame": start,
            "end_frame": end,
            "clearance_mm": float(max(clearance, 0.0)),
        })

    return np.array(clearances_mm, dtype=np.float64), intervals


def compute_foot_clearance_metrics(gt_joints, wham_joints, gt_events, wham_events, vertical_axis, unit_scale=1000.0):
    summary = {}
    details = {"gt_foot_clearances": [], "wham_foot_clearances": []}
    side_errors = []

    side_specs = [
        ("left", "L", LEFT_FOOT_IDX),
        ("right", "R", RIGHT_FOOT_IDX),
    ]

    for side_name, side_code, foot_idx in side_specs:
        gt_values, gt_intervals = swing_clearances_for_side(
            gt_joints, gt_events, foot_idx, side_code, vertical_axis, unit_scale=unit_scale
        )
        wham_values, wham_intervals = swing_clearances_for_side(
            wham_joints, wham_events, foot_idx, side_code, vertical_axis, unit_scale=unit_scale
        )

        details["gt_foot_clearances"].extend(gt_intervals)
        details["wham_foot_clearances"].extend(wham_intervals)

        summary[f"gt_{side_name}_foot_clearance_count"] = int(len(gt_values))
        summary[f"wham_{side_name}_foot_clearance_count"] = int(len(wham_values))
        summary[f"gt_{side_name}_foot_clearance_mean_mm"] = float(gt_values.mean()) if len(gt_values) else None
        summary[f"wham_{side_name}_foot_clearance_mean_mm"] = float(wham_values.mean()) if len(wham_values) else None

        if len(gt_values) and len(wham_values):
            gt_mean = float(gt_values.mean())
            wham_mean = float(wham_values.mean())
            abs_error = abs(wham_mean - gt_mean)
            pct_error = abs_error / abs(gt_mean) * 100.0 if abs(gt_mean) > 1e-8 else None
            paired_n = min(len(gt_values), len(wham_values))
            paired_mae = float(np.mean(np.abs(wham_values[:paired_n] - gt_values[:paired_n])))

            summary[f"{side_name}_foot_clearance_abs_error_mm"] = float(abs_error)
            summary[f"{side_name}_foot_clearance_pct_error"] = float(pct_error) if pct_error is not None else None
            summary[f"{side_name}_foot_clearance_paired_mae_mm"] = paired_mae
            summary[f"{side_name}_foot_clearance_paired_count_used"] = int(paired_n)
            side_errors.append(abs_error)
        else:
            summary[f"{side_name}_foot_clearance_abs_error_mm"] = None
            summary[f"{side_name}_foot_clearance_pct_error"] = None
            summary[f"{side_name}_foot_clearance_paired_mae_mm"] = None
            summary[f"{side_name}_foot_clearance_paired_count_used"] = 0

    summary["foot_clearance_mean_abs_error_mm"] = float(np.mean(side_errors)) if side_errors else None
    summary["foot_clearance_max_abs_error_mm"] = float(np.max(side_errors)) if side_errors else None

    return summary, details


def save_final_recommended_metrics_csv(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "category": "Pose preservation",
            "metric": "PA-MPJPE",
            "value": metrics.get("pa_mpjpe_mean_mm"),
            "unit": "mm",
            "interpretation": "Relative 3D body pose preservation",
        },
        {
            "category": "Pose preservation",
            "metric": "Knee ROM error",
            "value": metrics.get("knee_rom_mean_abs_error_deg"),
            "unit": "degrees",
            "interpretation": "Lower-limb movement amplitude preservation",
        },
        {
            "category": "Gait-cycle / temporal",
            "metric": "Step/contact timing error",
            "value": metrics.get("step_contact_timing_mae_s"),
            "unit": "seconds",
            "interpretation": "Gait-event timing preservation",
        },
        {
            "category": "Gait-cycle / temporal",
            "metric": "Cadence error",
            "value": metrics.get("cadence_abs_error_steps_per_min"),
            "unit": "steps/min",
            "interpretation": "Walking rhythm preservation",
        },
        {
            "category": "Local clinical gait",
            "metric": "Foot clearance error",
            "value": metrics.get("foot_clearance_mean_abs_error_mm"),
            "unit": "mm",
            "interpretation": "Foot-lift/shuffling-related preservation",
        },
        {
            "category": "Local clinical gait",
            "metric": "Pelvis-relative step length error",
            "value": metrics.get("step_length_mean_abs_error_mm"),
            "unit": "mm",
            "interpretation": "Local foot-placement preservation",
        },
        {
            "category": "Local clinical gait",
            "metric": "Pelvis-relative stride length error",
            "value": metrics.get("stride_length_mean_abs_error_mm"),
            "unit": "mm",
            "interpretation": "Local foot-placement preservation",
        },
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "metric", "value", "unit", "interpretation"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_knee_angles_csv(path, knee_per_frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(knee_per_frame["gt_left_knee_angle_deg"])
    data = np.column_stack([
        np.arange(n),
        knee_per_frame["gt_left_knee_angle_deg"],
        knee_per_frame["wham_left_knee_angle_deg"],
        knee_per_frame["left_knee_angle_abs_error_deg"],
        knee_per_frame["gt_right_knee_angle_deg"],
        knee_per_frame["wham_right_knee_angle_deg"],
        knee_per_frame["right_knee_angle_abs_error_deg"],
    ])

    header = (
        "frame,"
        "gt_left_knee_angle_deg,wham_left_knee_angle_deg,left_knee_angle_abs_error_deg,"
        "gt_right_knee_angle_deg,wham_right_knee_angle_deg,right_knee_angle_abs_error_deg"
    )

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"],
    )


def save_foot_clearance_csv(path, details):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for source_name in ["gt", "wham"]:
        for item in details[f"{source_name}_foot_clearances"]:
            rows.append({
                "source": source_name,
                "side": item["side"],
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "clearance_mm": item["clearance_mm"],
            })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "side", "start_frame", "end_frame", "clearance_mm"],
        )
        writer.writeheader()
        writer.writerows(rows)

# =============================================================================
# Saving helpers
# =============================================================================

def save_per_frame_csv(path, root_errors, pa_errors, pelvis_per_frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(root_errors)

    data = np.column_stack([
        np.arange(n),
        root_errors,
        pa_errors,
        pelvis_per_frame["pelvis_trajectory_error_mm"],
        pelvis_per_frame["pelvis_forward_error_mm"],
        pelvis_per_frame["pelvis_lateral_error_mm"],
        pelvis_per_frame["pelvis_vertical_error_mm"],
    ])

    header = (
        "frame,"
        "root_aligned_mpjpe_mm,"
        "pa_mpjpe_mm,"
        "pelvis_trajectory_error_mm,"
        "pelvis_forward_error_mm,"
        "pelvis_lateral_error_mm,"
        "pelvis_vertical_error_mm"
    )

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"],
    )


def save_events_csv(path, details):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for source_name in ["gt", "wham"]:
        for event in details[f"{source_name}_events"]:
            rows.append({
                "source": source_name,
                "type": "contact_event",
                "side": event["side"],
                "frame": event["frame"],
                "time_s": event["time_s"],
                "length_m": "",
                "forward_position_m": event["forward_position_m"],
                "start_frame": "",
                "end_frame": "",
            })

        for step in details[f"{source_name}_steps"]:
            rows.append({
                "source": source_name,
                "type": "step",
                "side": f"{step['from_side']}->{step['to_side']}",
                "frame": "",
                "time_s": "",
                "length_m": step["length_m"],
                "forward_position_m": "",
                "start_frame": step["start_frame"],
                "end_frame": step["end_frame"],
            })

        for stride in details[f"{source_name}_strides"]:
            rows.append({
                "source": source_name,
                "type": "stride",
                "side": stride["side"],
                "frame": "",
                "time_s": "",
                "length_m": stride["length_m"],
                "forward_position_m": "",
                "start_frame": stride["start_frame"],
                "end_frame": stride["end_frame"],
            })

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source",
            "type",
            "side",
            "frame",
            "time_s",
            "length_m",
            "forward_position_m",
            "start_frame",
            "end_frame",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    return obj


def safe_pct(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def make_interpretation(metrics):
    lines = []

    lines.append(
        f"PA-MPJPE is {metrics['pa_mpjpe_mean_mm']:.3f} mm, indicating the error after "
        f"frame-wise similarity alignment of the recovered pose to the BMCLab reference."
    )

    lines.append(
        f"Root-aligned MPJPE is {metrics['root_aligned_mpjpe_mean_mm']:.3f} mm. "
        f"Because this is much larger than PA-MPJPE, it should be interpreted as evidence "
        f"of a global orientation or coordinate-frame mismatch rather than only local pose error."
    )

    lines.append(
        f"Displacement-based walking speed error after trajectory alignment is "
        f"{metrics['walking_speed_displacement_aligned_abs_error_m_per_s']:.4f} m/s "
        f"({safe_pct(metrics['walking_speed_displacement_aligned_pct_error'])})."
    )

    lines.append(
        f"Path-based walking speed error after trajectory alignment is "
        f"{metrics['walking_speed_path_aligned_abs_error_m_per_s']:.4f} m/s "
        f"({safe_pct(metrics['walking_speed_path_aligned_pct_error'])})."
    )

    lines.append(
        f"Mean pelvis trajectory error after start and yaw alignment is "
        f"{metrics['pelvis_trajectory_error_mean_mm']:.3f} mm. "
        f"If this remains very large, it should be treated as a diagnostic metric for global "
        f"translation mismatch rather than as a clinically interpretable gait-feature error."
    )

    if metrics["step_length_mean_abs_error_mm"] is not None:
        lines.append(
            f"Mean pelvis-relative step length error is "
            f"{metrics['step_length_mean_abs_error_mm']:.3f} mm. "
            f"The percentage error is saved in the JSON but not emphasized because the "
            f"pelvis-relative denominator can be small."
        )
    else:
        lines.append(
            "Step length error could not be computed because too few alternating foot-contact events were detected."
        )

    if metrics["stride_length_mean_abs_error_mm"] is not None:
        lines.append(
            f"Mean pelvis-relative stride length error is "
            f"{metrics['stride_length_mean_abs_error_mm']:.3f} mm. "
            f"The percentage error is saved in the JSON but not emphasized because the "
            f"pelvis-relative denominator can be small."
        )
    else:
        lines.append(
            "Stride length error could not be computed because too few same-foot contact events were detected."
        )

    return lines


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bmclab-pkl", type=str, default=str(DEFAULT_BMCLAB_PKL))
    parser.add_argument("--wham-pkl", type=str, default=str(DEFAULT_WHAM_PKL))
    parser.add_argument("--smpl-model-dir", type=str, default=str(DEFAULT_SMPL_MODEL_DIR))

    parser.add_argument("--subject", type=str, default=DEFAULT_SUBJECT)
    parser.add_argument("--sequence", type=str, default=DEFAULT_SEQUENCE)
    parser.add_argument("--wham-subject-id", type=int, default=DEFAULT_WHAM_SUBJECT_ID)

    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory. Default: results/clinically_relevant_metrics_v1/<subject>_<sequence>_vs_wham_subject_<id>",
    )

    parser.add_argument("--prefer-world", action="store_true")
    parser.add_argument("--wham-beta-mode", type=str, default="mean", choices=["mean", "first"])
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument(
        "--fps-override",
        type=float,
        default=None,
        help="Override FPS used for speed and gait-event timing. Useful if metadata is wrong.",
    )

    args = parser.parse_args()

    bmclab_pkl = Path(args.bmclab_pkl)
    wham_pkl = Path(args.wham_pkl)
    smpl_model_dir = Path(args.smpl_model_dir)
    if args.out_dir is None:
        out_dir = (
            DEFAULT_REPO_ROOT
            / "results"
            / "clinically_relevant_metrics_v1"
            / f"{args.subject}_{args.sequence}_vs_wham_subject_{args.wham_subject_id}"
        )
    else:
        out_dir = Path(args.out_dir)

    print("\nConfiguration")
    print("-------------")
    print(f"BMCLab pkl:       {bmclab_pkl}")
    print(f"WHAM pkl:         {wham_pkl}")
    print(f"SMPL model dir:   {smpl_model_dir}")
    print(f"Subject:          {args.subject}")
    print(f"Sequence:         {args.sequence}")
    print(f"WHAM subject id:  {args.wham_subject_id}")
    print(f"Prefer world:     {args.prefer_world}")
    print(f"WHAM beta mode:   {args.wham_beta_mode}")
    print(f"Output dir:       {out_dir}")

    if not bmclab_pkl.exists():
        raise FileNotFoundError(f"BMCLab pkl not found: {bmclab_pkl}")

    if not wham_pkl.exists():
        raise FileNotFoundError(f"WHAM pkl not found: {wham_pkl}")

    if not smpl_model_dir.exists():
        raise FileNotFoundError(f"SMPL model dir not found: {smpl_model_dir}")

    print("\nLoading files...")
    bmclab_data = load_pickle_any(bmclab_pkl)
    wham_data = load_pickle_any(wham_pkl)

    inspect_top_level(bmclab_data, "BMCLab")
    inspect_top_level(wham_data, "WHAM")

    print("\nExtracting BMCLab sequence...")
    bmclab_seq = get_bmclab_sequence(
        bmclab_data,
        subject=args.subject,
        sequence=args.sequence,
    )

    gt_pose, gt_trans, gt_beta, fps = extract_bmclab_pose_trans_beta(bmclab_seq)

    if args.fps_override is not None:
        fps_used = float(args.fps_override)
    elif fps is not None:
        fps_used = float(fps)
    else:
        raise ValueError("No FPS found in BMCLab data. Use --fps-override 150.")

    print("BMCLab extracted:")
    print(f"  pose:  {gt_pose.shape}")
    print(f"  trans: {gt_trans.shape}")
    print(f"  beta:  {gt_beta.shape}")
    print(f"  fps metadata: {fps}")
    print(f"  fps used:     {fps_used}")

    print("\nExtracting WHAM subject...")
    wham_subject = get_wham_subject(wham_data, args.wham_subject_id)

    wham_pose, wham_trans, wham_beta, wham_pose_key, wham_trans_key = extract_wham_pose_trans_beta(
        wham_subject,
        prefer_world=args.prefer_world,
        beta_mode=args.wham_beta_mode,
    )

    print("WHAM extracted:")
    print(f"  pose key:  {wham_pose_key}")
    print(f"  trans key: {wham_trans_key}")
    print(f"  pose:      {wham_pose.shape}")
    print(f"  trans:     {wham_trans.shape}")
    print(f"  beta:      {wham_beta.shape}")

    n_frames = min(gt_pose.shape[0], wham_pose.shape[0])

    print(f"\nFrames used: {n_frames}")

    gt_pose = gt_pose[:n_frames]
    gt_trans = gt_trans[:n_frames]
    wham_pose = wham_pose[:n_frames]
    wham_trans = wham_trans[:n_frames]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\nBuilding SMPL model...")
    smpl_model = build_smpl_model(smpl_model_dir, device=device)

    print("\nReconstructing BMCLab joints...")
    gt_joints = smpl_joints_from_pose_trans_beta(
        smpl_model=smpl_model,
        pose=gt_pose,
        trans=gt_trans,
        beta=gt_beta,
        device=device,
        batch_size=args.batch_size,
    )

    print(f"  gt_joints: {gt_joints.shape}")

    print("\nReconstructing WHAM joints...")
    wham_joints_raw = smpl_joints_from_pose_trans_beta(
        smpl_model=smpl_model,
        pose=wham_pose,
        trans=wham_trans,
        beta=wham_beta,
        device=device,
        batch_size=args.batch_size,
    )

    print(f"  wham_joints_raw: {wham_joints_raw.shape}")

    print("\nInferring gait coordinate axes...")
    vertical_axis, body_axis_extents = infer_vertical_axis_from_joints(gt_joints)
    forward_axis = compute_forward_axis_from_pelvis(
        gt_joints,
        vertical_axis,
        root_idx=ROOT_IDX,
    )
    lateral_axis = compute_lateral_axis(forward_axis, vertical_axis)

    print(f"  vertical axis index: {vertical_axis}")
    print(f"  body extents per axis: {body_axis_extents}")
    print(f"  forward axis: {forward_axis}")
    print(f"  lateral axis: {lateral_axis}")

    print("\nAligning WHAM pelvis trajectory to BMCLab trajectory using one horizontal yaw rotation...")
    wham_joints_aligned, yaw_rotation_matrix = yaw_align_wham_to_gt_pelvis(
        gt_joints=gt_joints,
        wham_joints=wham_joints_raw,
        vertical_axis=vertical_axis,
        root_idx=ROOT_IDX,
    )

    print("  yaw rotation matrix:")
    print(yaw_rotation_matrix)

    print("\nComputing technical pose metrics on raw unsmoothed joints...")
    root_summary, root_errors = compute_root_aligned_mpjpe(
        pred_joints=wham_joints_raw,
        gt_joints=gt_joints,
        root_idx=ROOT_IDX,
        unit_scale=UNIT_SCALE,
    )

    pa_summary, pa_errors = compute_pa_mpjpe(
        pred_joints=wham_joints_raw,
        gt_joints=gt_joints,
        unit_scale=UNIT_SCALE,
    )

    print("\nSmoothing joints for gait-feature extraction...")
    gt_joints_gait = moving_average_trajectory(gt_joints, window=SMOOTHING_WINDOW)
    wham_joints_raw_gait = moving_average_trajectory(wham_joints_raw, window=SMOOTHING_WINDOW)
    wham_joints_aligned_gait = moving_average_trajectory(wham_joints_aligned, window=SMOOTHING_WINDOW)

    print(f"  smoothing window: {SMOOTHING_WINDOW} frames")

    print("\nComputing gait-feature metrics on smoothed joints...")
    walking_speed_summary = compute_walking_speed_metrics(
        gt_joints=gt_joints_gait,
        wham_joints_raw=wham_joints_raw_gait,
        wham_joints_aligned=wham_joints_aligned_gait,
        forward_axis=forward_axis,
        fps=fps_used,
        root_idx=ROOT_IDX,
    )

    pelvis_summary, pelvis_per_frame = compute_pelvis_trajectory_metrics(
        gt_joints=gt_joints_gait,
        wham_joints_aligned=wham_joints_aligned_gait,
        forward_axis=forward_axis,
        lateral_axis=lateral_axis,
        vertical_axis=vertical_axis,
        root_idx=ROOT_IDX,
        unit_scale=UNIT_SCALE,
    )

    step_stride_summary, step_stride_details = compute_step_stride_metrics(
        gt_joints=gt_joints_gait,
        wham_joints_aligned=wham_joints_aligned_gait,
        forward_axis=forward_axis,
        vertical_axis=vertical_axis,
        fps=fps_used,
    )

    print("\nComputing final recommended metrics...")
    knee_rom_summary, knee_per_frame = compute_knee_rom_metrics(
        gt_joints=gt_joints_gait,
        wham_joints=wham_joints_aligned_gait,
    )

    contact_timing_summary = compute_contact_timing_metrics(
        gt_events=step_stride_details["gt_events"],
        wham_events=step_stride_details["wham_events"],
        fps=fps_used,
    )

    cadence_summary = compute_cadence_metrics(
        gt_events=step_stride_details["gt_events"],
        wham_events=step_stride_details["wham_events"],
        n_frames=n_frames,
        fps=fps_used,
    )

    foot_clearance_summary, foot_clearance_details = compute_foot_clearance_metrics(
        gt_joints=gt_joints_gait,
        wham_joints=wham_joints_aligned_gait,
        gt_events=step_stride_details["gt_events"],
        wham_events=step_stride_details["wham_events"],
        vertical_axis=vertical_axis,
        unit_scale=UNIT_SCALE,
    )

    metrics_summary = {
        **root_summary,
        **pa_summary,
        **walking_speed_summary,
        **pelvis_summary,
        **step_stride_summary,
        **knee_rom_summary,
        **contact_timing_summary,
        **cadence_summary,
        **foot_clearance_summary,

        "frames_used": int(n_frames),
        "num_joints": int(gt_joints.shape[1]),
        "root_idx": int(ROOT_IDX),
        "left_foot_idx": int(LEFT_FOOT_IDX),
        "right_foot_idx": int(RIGHT_FOOT_IDX),
        "fps_used": float(fps_used),

        "gait_smoothing_window_frames": int(SMOOTHING_WINDOW),
        "step_stride_uses_pelvis_relative_foot_placement": True,

        "vertical_axis": int(vertical_axis),
        "body_axis_extents": body_axis_extents.tolist(),
        "forward_axis": forward_axis.tolist(),
        "lateral_axis": lateral_axis.tolist(),
        "yaw_rotation_matrix_wham_to_gt": yaw_rotation_matrix.tolist(),

        "wham_beta_mode": args.wham_beta_mode,
        "gt_beta_mode": "constant",
        "wham_pose_key": wham_pose_key,
        "wham_trans_key": wham_trans_key,
        "prefer_world": bool(args.prefer_world),
    }

    interpretation = make_interpretation(metrics_summary)

    full_summary = {
        "configuration": {
            "bmclab_pkl": str(bmclab_pkl),
            "wham_pkl": str(wham_pkl),
            "smpl_model_dir": str(smpl_model_dir),
            "subject": args.subject,
            "sequence": args.sequence,
            "wham_subject_id": args.wham_subject_id,
            "fps_metadata": fps,
            "fps_used": fps_used,
            "prefer_world": bool(args.prefer_world),
        },
        "metrics": metrics_summary,
        "interpretation": interpretation,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / (
        f"{args.subject}_{args.sequence}_vs_wham_subject_"
        f"{args.wham_subject_id}_all_metrics_v3_summary.json"
    )

    per_frame_path = out_dir / (
        f"{args.subject}_{args.sequence}_vs_wham_subject_"
        f"{args.wham_subject_id}_per_frame_metrics_v3.csv"
    )

    events_path = out_dir / (
        f"{args.subject}_{args.sequence}_vs_wham_subject_"
        f"{args.wham_subject_id}_gait_events_steps_strides_v3.csv"
    )

    final_metrics_path = out_dir / (
        f"{args.subject}_{args.sequence}_vs_wham_subject_"
        f"{args.wham_subject_id}_final_recommended_metrics_v1.csv"
    )

    knee_angles_path = out_dir / (
        f"{args.subject}_{args.sequence}_vs_wham_subject_"
        f"{args.wham_subject_id}_knee_angles_v1.csv"
    )

    foot_clearance_path = out_dir / (
        f"{args.subject}_{args.sequence}_vs_wham_subject_"
        f"{args.wham_subject_id}_foot_clearances_v1.csv"
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(full_summary), f, indent=2)

    save_per_frame_csv(
        per_frame_path,
        root_errors=root_errors,
        pa_errors=pa_errors,
        pelvis_per_frame=pelvis_per_frame,
    )

    save_events_csv(
        events_path,
        details=step_stride_details,
    )

    save_final_recommended_metrics_csv(
        final_metrics_path,
        metrics=metrics_summary,
    )

    save_knee_angles_csv(
        knee_angles_path,
        knee_per_frame=knee_per_frame,
    )

    save_foot_clearance_csv(
        foot_clearance_path,
        details=foot_clearance_details,
    )

    print("\nMain metrics")
    print("------------")
    print(f"Root-Aligned MPJPE:                         {metrics_summary['root_aligned_mpjpe_mean_mm']:.3f} mm")
    print(f"PA-MPJPE:                                   {metrics_summary['pa_mpjpe_mean_mm']:.3f} mm")
    print(f"Knee ROM mean absolute error:               {metrics_summary['knee_rom_mean_abs_error_deg']:.3f} deg")
    if metrics_summary["step_contact_timing_mae_s"] is not None:
        print(f"Step/contact timing MAE:                    {metrics_summary['step_contact_timing_mae_s']:.4f} s")
    else:
        print("Step/contact timing MAE:                    not available")
    print(f"Cadence error:                              {metrics_summary['cadence_abs_error_steps_per_min']:.3f} steps/min")
    if metrics_summary["foot_clearance_mean_abs_error_mm"] is not None:
        print(f"Foot clearance mean absolute error:         {metrics_summary['foot_clearance_mean_abs_error_mm']:.3f} mm")
    else:
        print("Foot clearance mean absolute error:         not available")

    print(
        f"Original walking speed, displacement:       "
        f"{metrics_summary['walking_speed_displacement_original_m_per_s']:.4f} m/s"
    )
    print(
        f"WHAM walking speed, displacement aligned:   "
        f"{metrics_summary['walking_speed_displacement_wham_aligned_m_per_s']:.4f} m/s"
    )
    print(
        f"Walking speed error, displacement aligned:  "
        f"{metrics_summary['walking_speed_displacement_aligned_abs_error_m_per_s']:.4f} m/s "
        f"({safe_pct(metrics_summary['walking_speed_displacement_aligned_pct_error'])})"
    )

    print(
        f"Original walking speed, path-based:         "
        f"{metrics_summary['walking_speed_path_original_m_per_s']:.4f} m/s"
    )
    print(
        f"WHAM walking speed, path-based aligned:     "
        f"{metrics_summary['walking_speed_path_wham_aligned_m_per_s']:.4f} m/s"
    )
    print(
        f"Walking speed error, path-based aligned:    "
        f"{metrics_summary['walking_speed_path_aligned_abs_error_m_per_s']:.4f} m/s "
        f"({safe_pct(metrics_summary['walking_speed_path_aligned_pct_error'])})"
    )

    print(f"Pelvis trajectory error:                    {metrics_summary['pelvis_trajectory_error_mean_mm']:.3f} mm")
    print(f"GT gait events:                             {metrics_summary['gt_num_gait_events']}")
    print(f"WHAM gait events:                           {metrics_summary['wham_num_gait_events']}")

    if metrics_summary["step_length_mean_abs_error_mm"] is not None:
        print(
            f"Pelvis-relative step length error:          "
            f"{metrics_summary['step_length_mean_abs_error_mm']:.3f} mm "
            f"(percentage saved in JSON, not emphasized)"
        )
    else:
        print("Pelvis-relative step length error:          not available")

    if metrics_summary["stride_length_mean_abs_error_mm"] is not None:
        print(
            f"Pelvis-relative stride length error:        "
            f"{metrics_summary['stride_length_mean_abs_error_mm']:.3f} mm "
            f"(percentage saved in JSON, not emphasized)"
        )
    else:
        print("Pelvis-relative stride length error:        not available")

    print("\nSaved outputs")
    print("-------------")
    print(f"Summary JSON:       {summary_path}")
    print(f"Per-frame CSV:      {per_frame_path}")
    print(f"Gait events CSV:    {events_path}")
    print(f"Final metrics CSV:  {final_metrics_path}")
    print(f"Knee angles CSV:    {knee_angles_path}")
    print(f"Foot clearance CSV: {foot_clearance_path}")

    print("\nInterpretation")
    print("--------------")
    for line in interpretation:
        print(f"- {line}")



# =============================================================================
# BATCH ALL-EXPERIMENTS RUNNER
# =============================================================================

"""
Batch version of the WHAM-vs-BMCLab metric script.

Purpose
-------
This script automatically finds all WHAM PKL files for one subject's rendered
experiments (E0, E1, ..., E13), computes the same recommended thesis metrics
for each experiment, and writes both per-experiment output files and one combined
summary CSV.

It reuses the metric functions from:
    06_compare_bmclab_wham_recommended_metrics_standalone_auto_frames.py

Therefore, place this script in the same scripts/ folder as that file.

Expected WHAM PKL input structure, for example:
    C:/Users/sopha/synthetic-pd/data/wham_outputs/SUB01/E0/SUB01_E0_wham_output.pkl
    C:/Users/sopha/synthetic-pd/data/wham_outputs/SUB01/E1/SUB01_E1_wham_output.pkl
    ...

Also accepted:
    C:/Users/sopha/synthetic-pd/data/wham_outputs/SUB01_E0_wham_output.pkl
    C:/Users/sopha/synthetic-pd/data/wham_outputs/.../SUB01_E0.../wham_output.pkl

Run from project root:
    python scripts/06_compare_bmclab_wham_all_experiments_auto_subject.py --subject SUB03 --trial SUB03_off_walk_1

Useful explicit run:
    python scripts/06_compare_bmclab_wham_all_experiments_auto_subject.py ^
        --subject SUB03 ^
        --trial SUB03_off_walk_1 ^
        --wham-pkl-root data/wham_outputs/SUB01 ^
        --fps-override 150
"""

import argparse
import csv
import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Import the original standalone metric script even though its filename starts
# with a number.
# -----------------------------------------------------------------------------

def load_metric_module(base_script_path: Path):
    base_script_path = Path(base_script_path)
    if not base_script_path.exists():
        raise FileNotFoundError(
            "Could not find the base metric script:\n"
            f"{base_script_path}\n\n"
            "Place this batch script in the same folder as:\n"
            "06_compare_bmclab_wham_recommended_metrics_standalone_auto_frames.py\n"
            "or pass --base-script with the correct path."
        )

    spec = importlib.util.spec_from_file_location("wham_metric_base", str(base_script_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["wham_metric_base"] = module
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Experiment discovery helpers
# -----------------------------------------------------------------------------

def experiment_number_from_path(path: Path):
    """Return integer experiment number from E0, E1, ..., E13 in path/name."""
    path = Path(path)

    # Prefer folder names exactly like E0, E1, ...
    for part in path.parts:
        match = re.fullmatch(r"E(\d+)", part)
        if match:
            return int(match.group(1))

    # Then try names like SUB01_E0_wham_output.pkl or SUB01_E0_wham_rgb_150fps
    match = re.search(r"_E(\d+)(?:_|\b)", path.name)
    if match:
        return int(match.group(1))

    # Last fallback: any E-number in the full path
    match = re.search(r"(?:^|[\\/_-])E(\d+)(?:[\\/_-]|$)", str(path))
    if match:
        return int(match.group(1))

    return None


def find_wham_pkls(wham_pkl_root: Path, subject: str):
    """Find all WHAM output PKLs for a subject and sort them numerically by experiment."""
    wham_pkl_root = Path(wham_pkl_root)
    if not wham_pkl_root.exists():
        raise FileNotFoundError(f"WHAM PKL root does not exist:\n{wham_pkl_root}")

    all_candidates = []

    # Metrics-ready files, e.g. SUB01_E0_wham_output.pkl
    all_candidates.extend(wham_pkl_root.rglob(f"{subject}_E*_wham_output.pkl"))

    # Raw WHAM files under folders, e.g. output/demo/SUB01_E0.../wham_output.pkl
    for p in wham_pkl_root.rglob("wham_output.pkl"):
        if subject in str(p):
            all_candidates.append(p)

    # De-duplicate and keep only paths with an experiment number.
    unique = {}
    for p in all_candidates:
        exp = experiment_number_from_path(p)
        if exp is None:
            continue
        # Prefer metrics-ready renamed files over generic wham_output.pkl if both exist.
        current = unique.get(exp)
        if current is None:
            unique[exp] = p
        else:
            current_is_generic = current.name == "wham_output.pkl"
            new_is_renamed = p.name != "wham_output.pkl"
            if current_is_generic and new_is_renamed:
                unique[exp] = p

    if not unique:
        raise FileNotFoundError(
            f"No experiment WHAM PKLs found for subject {subject!r} under:\n{wham_pkl_root}\n\n"
            "Expected examples:\n"
            f"  {subject}/E0/{subject}_E0_wham_output.pkl\n"
            f"  {subject}_E0_wham_output.pkl\n"
            f"  .../{subject}_E0_wham_rgb_150fps/wham_output.pkl"
        )

    return [(exp, unique[exp]) for exp in sorted(unique)]


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def metric_value(metrics, key):
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return value


def write_combined_summary_csv(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subject",
        "sequence",
        "experiment",
        "experiment_number",
        "wham_pkl",
        "frames_used",
        "fps_used",
        "pa_mpjpe_mean_mm",
        "root_aligned_mpjpe_mean_mm",
        "knee_rom_mean_abs_error_deg",
        "step_contact_timing_mae_s",
        "cadence_abs_error_steps_per_min",
        "foot_clearance_mean_abs_error_mm",
        "step_length_mean_abs_error_mm",
        "stride_length_mean_abs_error_mm",
        "walking_speed_displacement_aligned_abs_error_m_per_s",
        "walking_speed_path_aligned_abs_error_m_per_s",
        "pelvis_trajectory_error_mean_mm",
        "gt_num_gait_events",
        "wham_num_gait_events",
        "gait_event_count_difference",
        "summary_json",
        "final_metrics_csv",
        "per_frame_csv",
        "gait_events_csv",
        "knee_angles_csv",
        "foot_clearance_csv",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Core metric computation for one WHAM PKL
# -----------------------------------------------------------------------------

def compute_metrics_for_one_experiment(
    base,
    smpl_model,
    gt_pose,
    gt_trans,
    gt_beta,
    gt_joints,
    gt_joints_gait,
    vertical_axis,
    body_axis_extents,
    forward_axis,
    lateral_axis,
    fps_used,
    n_gt_frames,
    wham_pkl,
    experiment_name,
    args,
    out_dir,
):
    wham_data = base.load_pickle_any(wham_pkl)
    wham_subject = base.get_wham_subject(wham_data, args.wham_subject_id)

    wham_pose, wham_trans, wham_beta, wham_pose_key, wham_trans_key = base.extract_wham_pose_trans_beta(
        wham_subject,
        prefer_world=args.prefer_world,
        beta_mode=args.wham_beta_mode,
    )

    n_frames = min(n_gt_frames, wham_pose.shape[0], wham_trans.shape[0])

    wham_pose = wham_pose[:n_frames]
    wham_trans = wham_trans[:n_frames]
    gt_joints_use = gt_joints[:n_frames]
    gt_joints_gait_use = gt_joints_gait[:n_frames]

    wham_joints_raw = base.smpl_joints_from_pose_trans_beta(
        smpl_model=smpl_model,
        pose=wham_pose,
        trans=wham_trans,
        beta=wham_beta,
        device=args.device,
        batch_size=args.batch_size,
    )

    wham_joints_aligned, yaw_rotation_matrix = base.yaw_align_wham_to_gt_pelvis(
        gt_joints=gt_joints_use,
        wham_joints=wham_joints_raw,
        vertical_axis=vertical_axis,
        root_idx=base.ROOT_IDX,
    )

    root_summary, root_errors = base.compute_root_aligned_mpjpe(
        pred_joints=wham_joints_raw,
        gt_joints=gt_joints_use,
        root_idx=base.ROOT_IDX,
        unit_scale=base.UNIT_SCALE,
    )

    pa_summary, pa_errors = base.compute_pa_mpjpe(
        pred_joints=wham_joints_raw,
        gt_joints=gt_joints_use,
        unit_scale=base.UNIT_SCALE,
    )

    wham_joints_raw_gait = base.moving_average_trajectory(wham_joints_raw, window=base.SMOOTHING_WINDOW)
    wham_joints_aligned_gait = base.moving_average_trajectory(wham_joints_aligned, window=base.SMOOTHING_WINDOW)

    walking_speed_summary = base.compute_walking_speed_metrics(
        gt_joints=gt_joints_gait_use,
        wham_joints_raw=wham_joints_raw_gait,
        wham_joints_aligned=wham_joints_aligned_gait,
        forward_axis=forward_axis,
        fps=fps_used,
        root_idx=base.ROOT_IDX,
    )

    pelvis_summary, pelvis_per_frame = base.compute_pelvis_trajectory_metrics(
        gt_joints=gt_joints_gait_use,
        wham_joints_aligned=wham_joints_aligned_gait,
        forward_axis=forward_axis,
        lateral_axis=lateral_axis,
        vertical_axis=vertical_axis,
        root_idx=base.ROOT_IDX,
        unit_scale=base.UNIT_SCALE,
    )

    step_stride_summary, step_stride_details = base.compute_step_stride_metrics(
        gt_joints=gt_joints_gait_use,
        wham_joints_aligned=wham_joints_aligned_gait,
        forward_axis=forward_axis,
        vertical_axis=vertical_axis,
        fps=fps_used,
    )

    knee_rom_summary, knee_per_frame = base.compute_knee_rom_metrics(
        gt_joints=gt_joints_gait_use,
        wham_joints=wham_joints_aligned_gait,
    )

    contact_timing_summary = base.compute_contact_timing_metrics(
        gt_events=step_stride_details["gt_events"],
        wham_events=step_stride_details["wham_events"],
        fps=fps_used,
    )

    cadence_summary = base.compute_cadence_metrics(
        gt_events=step_stride_details["gt_events"],
        wham_events=step_stride_details["wham_events"],
        n_frames=n_frames,
        fps=fps_used,
    )

    foot_clearance_summary, foot_clearance_details = base.compute_foot_clearance_metrics(
        gt_joints=gt_joints_gait_use,
        wham_joints=wham_joints_aligned_gait,
        gt_events=step_stride_details["gt_events"],
        wham_events=step_stride_details["wham_events"],
        vertical_axis=vertical_axis,
        unit_scale=base.UNIT_SCALE,
    )

    metrics_summary = {
        **root_summary,
        **pa_summary,
        **walking_speed_summary,
        **pelvis_summary,
        **step_stride_summary,
        **knee_rom_summary,
        **contact_timing_summary,
        **cadence_summary,
        **foot_clearance_summary,
        "experiment": experiment_name,
        "wham_pkl": str(wham_pkl),
        "frames_used": int(n_frames),
        "num_joints": int(gt_joints_use.shape[1]),
        "root_idx": int(base.ROOT_IDX),
        "left_foot_idx": int(base.LEFT_FOOT_IDX),
        "right_foot_idx": int(base.RIGHT_FOOT_IDX),
        "fps_used": float(fps_used),
        "gait_smoothing_window_frames": int(base.SMOOTHING_WINDOW),
        "step_stride_uses_pelvis_relative_foot_placement": True,
        "vertical_axis": int(vertical_axis),
        "body_axis_extents": body_axis_extents.tolist(),
        "forward_axis": forward_axis.tolist(),
        "lateral_axis": lateral_axis.tolist(),
        "yaw_rotation_matrix_wham_to_gt": yaw_rotation_matrix.tolist(),
        "wham_beta_mode": args.wham_beta_mode,
        "gt_beta_mode": "constant",
        "wham_pose_key": wham_pose_key,
        "wham_trans_key": wham_trans_key,
        "prefer_world": bool(args.prefer_world),
    }

    interpretation = base.make_interpretation(metrics_summary)

    full_summary = {
        "configuration": {
            "bmclab_pkl": str(args.bmclab_pkl),
            "wham_pkl": str(wham_pkl),
            "smpl_model_dir": str(args.smpl_model_dir),
            "subject": args.subject,
            "sequence": args.sequence,
            "experiment": experiment_name,
            "wham_subject_id": args.wham_subject_id,
            "fps_used": fps_used,
            "prefer_world": bool(args.prefer_world),
        },
        "metrics": metrics_summary,
        "interpretation": interpretation,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.subject}_{experiment_name}_{args.sequence}_vs_wham_subject_{args.wham_subject_id}"

    summary_path = out_dir / f"{stem}_summary.json"
    per_frame_path = out_dir / f"{stem}_per_frame_metrics.csv"
    events_path = out_dir / f"{stem}_gait_events_steps_strides.csv"
    final_metrics_path = out_dir / f"{stem}_final_recommended_metrics.csv"
    knee_angles_path = out_dir / f"{stem}_knee_angles.csv"
    foot_clearance_path = out_dir / f"{stem}_foot_clearances.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(full_summary), f, indent=2)

    base.save_per_frame_csv(
        per_frame_path,
        root_errors=root_errors,
        pa_errors=pa_errors,
        pelvis_per_frame=pelvis_per_frame,
    )

    base.save_events_csv(events_path, details=step_stride_details)
    base.save_final_recommended_metrics_csv(final_metrics_path, metrics=metrics_summary)
    base.save_knee_angles_csv(knee_angles_path, knee_per_frame=knee_per_frame)
    base.save_foot_clearance_csv(foot_clearance_path, details=foot_clearance_details)

    combined_row = {
        "subject": args.subject,
        "sequence": args.sequence,
        "experiment": experiment_name,
        "experiment_number": int(experiment_name.replace("E", "")),
        "wham_pkl": str(wham_pkl),
        "frames_used": metrics_summary.get("frames_used"),
        "fps_used": metrics_summary.get("fps_used"),
        "pa_mpjpe_mean_mm": metric_value(metrics_summary, "pa_mpjpe_mean_mm"),
        "root_aligned_mpjpe_mean_mm": metric_value(metrics_summary, "root_aligned_mpjpe_mean_mm"),
        "knee_rom_mean_abs_error_deg": metric_value(metrics_summary, "knee_rom_mean_abs_error_deg"),
        "step_contact_timing_mae_s": metric_value(metrics_summary, "step_contact_timing_mae_s"),
        "cadence_abs_error_steps_per_min": metric_value(metrics_summary, "cadence_abs_error_steps_per_min"),
        "foot_clearance_mean_abs_error_mm": metric_value(metrics_summary, "foot_clearance_mean_abs_error_mm"),
        "step_length_mean_abs_error_mm": metric_value(metrics_summary, "step_length_mean_abs_error_mm"),
        "stride_length_mean_abs_error_mm": metric_value(metrics_summary, "stride_length_mean_abs_error_mm"),
        "walking_speed_displacement_aligned_abs_error_m_per_s": metric_value(metrics_summary, "walking_speed_displacement_aligned_abs_error_m_per_s"),
        "walking_speed_path_aligned_abs_error_m_per_s": metric_value(metrics_summary, "walking_speed_path_aligned_abs_error_m_per_s"),
        "pelvis_trajectory_error_mean_mm": metric_value(metrics_summary, "pelvis_trajectory_error_mean_mm"),
        "gt_num_gait_events": metrics_summary.get("gt_num_gait_events"),
        "wham_num_gait_events": metrics_summary.get("wham_num_gait_events"),
        "gait_event_count_difference": metrics_summary.get("gait_event_count_difference"),
        "summary_json": str(summary_path),
        "final_metrics_csv": str(final_metrics_path),
        "per_frame_csv": str(per_frame_path),
        "gait_events_csv": str(events_path),
        "knee_angles_csv": str(knee_angles_path),
        "foot_clearance_csv": str(foot_clearance_path),
    }

    return combined_row


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    default_repo_root = Path(__file__).resolve().parents[1]

    parser.add_argument("--repo-root", type=str, default=str(default_repo_root))
    parser.add_argument("--base-script", type=str, default=None)

    parser.add_argument("--subject", type=str, default="SUB01", help="BMCLab subject, e.g. SUB03")
    parser.add_argument(
        "--trial",
        type=str,
        default=None,
        help="BMCLab trial/sequence name, e.g. SUB03_off_walk_1. Alias for --sequence, matching script 04.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="BMCLab sequence name. Optional; --trial is preferred. If omitted, defaults to <subject>_off_walk_1.",
    )
    parser.add_argument("--wham-subject-id", type=int, default=0)

    parser.add_argument("--bmclab-pkl", type=str, default=None)
    parser.add_argument("--smpl-model-dir", type=str, default=None)
    parser.add_argument("--wham-pkl-root", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=None)

    parser.add_argument("--prefer-world", action="store_true")
    parser.add_argument("--wham-beta-mode", type=str, default="mean", choices=["mean", "first"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fps-override", type=float, default=None)

    args = parser.parse_args()

    # Make script 06 behave like script 04:
    #     python scripts\06_compare_bmclab_wham_all_experiments.py --subject SUB03 --trial SUB03_off_walk_1
    # --trial and --sequence mean the same thing. --trial is preferred because it
    # matches the naming used in the render-input validation script.
    if args.trial is not None and args.sequence is not None and args.trial != args.sequence:
        raise ValueError(
            f"You provided both --trial={args.trial!r} and --sequence={args.sequence!r}. "
            "Use only one, or make them identical."
        )

    if args.trial is not None:
        args.sequence = args.trial
    elif args.sequence is None:
        args.sequence = f"{args.subject}_off_walk_1"

    repo_root = Path(args.repo_root)

    if args.base_script is None:
        base_script = Path(__file__).resolve().parent / "06_compare_bmclab_wham_recommended_metrics_standalone_auto_frames.py"
    else:
        base_script = Path(args.base_script)

    args.bmclab_pkl = Path(args.bmclab_pkl) if args.bmclab_pkl else repo_root / "data" / "raw" / "BMCLab.pkl"
    args.smpl_model_dir = Path(args.smpl_model_dir) if args.smpl_model_dir else repo_root / "models" / "smpl"

    if args.wham_pkl_root is None:
        subject_specific_root = repo_root / "data" / "wham_outputs" / args.subject
        general_root = repo_root / "data" / "wham_outputs"
        args.wham_pkl_root = subject_specific_root if subject_specific_root.exists() else general_root
    else:
        args.wham_pkl_root = Path(args.wham_pkl_root)

    if args.out_root is None:
        args.out_root = repo_root / "results" / "clinically_relevant_metrics_v1" / "batch_wham_experiments"
    else:
        args.out_root = Path(args.out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    print("\nConfiguration")
    print("-------------")
    print(f"Repo root:        {repo_root}")
    print("Base script:      embedded in this standalone file")
    print(f"BMCLab pkl:       {args.bmclab_pkl}")
    print(f"SMPL model dir:   {args.smpl_model_dir}")
    print(f"WHAM PKL root:    {args.wham_pkl_root}")
    print(f"Subject:          {args.subject}")
    print(f"Sequence:         {args.sequence}")
    print(f"WHAM subject id:  {args.wham_subject_id}")
    print(f"Prefer world:     {args.prefer_world}")
    print(f"WHAM beta mode:   {args.wham_beta_mode}")
    print(f"Output root:      {args.out_root}")
    print(f"Device:           {device}")

    if not args.bmclab_pkl.exists():
        raise FileNotFoundError(f"BMCLab pkl not found:\n{args.bmclab_pkl}")
    if not args.smpl_model_dir.exists():
        raise FileNotFoundError(f"SMPL model dir not found:\n{args.smpl_model_dir}")

    # This full-standalone script already contains the base metric functions.
    # Create a lightweight namespace so the original batch code can keep using base.<function>.
    class _EmbeddedBase:
        pass

    base = _EmbeddedBase()
    for _name, _value in list(globals().items()):
        setattr(base, _name, _value)

    experiments = find_wham_pkls(args.wham_pkl_root, args.subject)

    print("\nFound WHAM experiment PKLs")
    print("--------------------------")
    for exp_num, pkl_path in experiments:
        print(f"E{exp_num}: {pkl_path}")

    print("\nLoading BMCLab reference once...")
    bmclab_data = base.load_pickle_any(args.bmclab_pkl)
    bmclab_seq = base.get_bmclab_sequence(bmclab_data, subject=args.subject, sequence=args.sequence)
    gt_pose, gt_trans, gt_beta, fps = base.extract_bmclab_pose_trans_beta(bmclab_seq)

    if args.fps_override is not None:
        fps_used = float(args.fps_override)
    elif fps is not None:
        fps_used = float(fps)
    else:
        raise ValueError("No FPS found in BMCLab data. Use --fps-override 150.")

    print(f"BMCLab pose:  {gt_pose.shape}")
    print(f"BMCLab trans: {gt_trans.shape}")
    print(f"BMCLab beta:  {gt_beta.shape}")
    print(f"FPS used:     {fps_used}")

    print("\nBuilding SMPL model once...")
    smpl_model = base.build_smpl_model(args.smpl_model_dir, device=device)

    print("\nReconstructing BMCLab joints once...")
    gt_joints = base.smpl_joints_from_pose_trans_beta(
        smpl_model=smpl_model,
        pose=gt_pose,
        trans=gt_trans,
        beta=gt_beta,
        device=device,
        batch_size=args.batch_size,
    )

    vertical_axis, body_axis_extents = base.infer_vertical_axis_from_joints(gt_joints)
    forward_axis = base.compute_forward_axis_from_pelvis(gt_joints, vertical_axis, root_idx=base.ROOT_IDX)
    lateral_axis = base.compute_lateral_axis(forward_axis, vertical_axis)
    gt_joints_gait = base.moving_average_trajectory(gt_joints, window=base.SMOOTHING_WINDOW)

    print(f"GT joints:              {gt_joints.shape}")
    print(f"Vertical axis:          {vertical_axis}")
    print(f"Forward axis:           {forward_axis}")
    print(f"Lateral axis:           {lateral_axis}")
    print(f"Smoothing window:       {base.SMOOTHING_WINDOW} frames")

    subject_out_root = args.out_root / args.subject / args.sequence
    subject_out_root.mkdir(parents=True, exist_ok=True)

    combined_rows = []

    for exp_num, wham_pkl in experiments:
        experiment_name = f"E{exp_num}"
        experiment_out_dir = subject_out_root / experiment_name

        print("\n" + "#" * 88)
        print(f"Computing metrics for {args.subject} {experiment_name}")
        print(f"WHAM PKL: {wham_pkl}")
        print("#" * 88)

        row = compute_metrics_for_one_experiment(
            base=base,
            smpl_model=smpl_model,
            gt_pose=gt_pose,
            gt_trans=gt_trans,
            gt_beta=gt_beta,
            gt_joints=gt_joints,
            gt_joints_gait=gt_joints_gait,
            vertical_axis=vertical_axis,
            body_axis_extents=body_axis_extents,
            forward_axis=forward_axis,
            lateral_axis=lateral_axis,
            fps_used=fps_used,
            n_gt_frames=gt_pose.shape[0],
            wham_pkl=wham_pkl,
            experiment_name=experiment_name,
            args=args,
            out_dir=experiment_out_dir,
        )

        combined_rows.append(row)

        print("Main outputs:")
        print(f"  PA-MPJPE:                    {row['pa_mpjpe_mean_mm']} mm")
        print(f"  Knee ROM error:              {row['knee_rom_mean_abs_error_deg']} deg")
        print(f"  Contact timing error:        {row['step_contact_timing_mae_s']} s")
        print(f"  Cadence error:               {row['cadence_abs_error_steps_per_min']} steps/min")
        print(f"  Foot clearance error:        {row['foot_clearance_mean_abs_error_mm']} mm")
        print(f"  Step length error:           {row['step_length_mean_abs_error_mm']} mm")
        print(f"  Stride length error:         {row['stride_length_mean_abs_error_mm']} mm")

    combined_csv = subject_out_root / f"{args.subject}_{args.sequence}_all_experiments_metric_summary.csv"
    write_combined_summary_csv(combined_csv, combined_rows)

    # Also save JSON summary for reproducibility.
    combined_json = subject_out_root / f"{args.subject}_{args.sequence}_all_experiments_metric_summary.json"
    with combined_json.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(combined_rows), f, indent=2)

    print("\nDone.")
    print("-----")
    print(f"Experiments processed: {len(combined_rows)}")
    print(f"Combined CSV:          {combined_csv}")
    print(f"Combined JSON:         {combined_json}")
    print(f"Per-experiment files:  {subject_out_root}")


if __name__ == "__main__":
    main()
