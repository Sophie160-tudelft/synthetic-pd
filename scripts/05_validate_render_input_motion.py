r"""
Validate whether the final SMPL-X file used for rendering preserves the original
BMCLab SMPL motion before WHAM is involved.

Updated version: computes the same final recommended thesis metric set as
15_compare_bmclab_wham_recommended_metrics.py:

1. PA-MPJPE
2. Knee ROM error
3. Step/contact timing error
4. Cadence error
5. Foot clearance error
6. Pelvis-relative step length error
7. Pelvis-relative stride length error

This script compares:
    GT   = original BMCLab SMPL motion
    PRED = final SMPL-X motion used for Blender/Unreal rendering

Run from project root:
    python scripts/04_validate_render_input_motion_standalone.py

Important:
    This standalone version embeds the recommended metric functions directly.
    It does not require script 15 to be present in the scripts folder.
"""

import argparse
import csv
import json
import pickle
from pathlib import Path
import re

import numpy as np

# -----------------------------------------------------------------------------
# Compatibility patch for older SMPL/chumpy model files with NumPy >= 2.0
# -----------------------------------------------------------------------------
# The SMPL pickle can import chumpy internally. Older chumpy versions use aliases
# such as np.int and np.float, which were removed in NumPy 2.x. Define them
# before smplx loads the model so unpickling does not fail.
for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

# Some old dependencies also expect inspect.getargspec, removed in Python 3.11.
try:
    import inspect as _inspect
    if not hasattr(_inspect, "getargspec"):
        _inspect.getargspec = _inspect.getfullargspec
except Exception:
    pass

import torch
import smplx


# =============================================================================
# Settings
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\sopha\synthetic-pd")

BMCLAB_PKL = PROJECT_ROOT / "data" / "raw" / "BMCLab.pkl"
MODEL_ROOT = PROJECT_ROOT / "models"

SUBJECT = "SUB02"
TRIAL = "SUB02_off_walk_2"

# Optional manual override. Keep as None to auto-detect by TRIAL and frame count.
FINAL_SMPLX_NPZ = None

OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "render_input_validation"
    / f"{SUBJECT}_{TRIAL}"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

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
NUM_COMMON_JOINTS = 22


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
# Standalone replacement for script-15 import
# =============================================================================

def load_embedded_metric_functions():
    import types
    rec = types.SimpleNamespace(
        compute_root_aligned_mpjpe=compute_root_aligned_mpjpe,
        compute_pa_mpjpe=compute_pa_mpjpe,
        moving_average_trajectory=moving_average_trajectory,
        infer_vertical_axis_from_joints=infer_vertical_axis_from_joints,
        compute_forward_axis_from_pelvis=compute_forward_axis_from_pelvis,
        build_gait_events=build_gait_events,
        compute_step_stride_metrics=compute_step_stride_metrics,
        compute_knee_rom_metrics=compute_knee_rom_metrics,
        compute_contact_timing_metrics=compute_contact_timing_metrics,
        compute_cadence_metrics=compute_cadence_metrics,
        compute_foot_clearance_metrics=compute_foot_clearance_metrics,
        make_json_safe=make_json_safe,
        save_final_recommended_metrics_csv=save_final_recommended_metrics_csv,
        save_events_csv=save_events_csv,
        save_knee_angles_csv=save_knee_angles_csv,
        save_foot_clearance_csv=save_foot_clearance_csv
    )
    return rec, Path(__file__).name + ' (standalone embedded metric functions)'


def parse_fps_frames_from_name(path):
    match = re.search(r"_smplx_fitted_(\d+)fps_(\d+)frames", Path(path).name)
    if match is None:
        return -1, -1
    return int(match.group(1)), int(match.group(2))


def corrected_render_file_score(path):
    """Higher score means more likely to be the final render-ready SMPL-X file."""
    name = Path(path).name
    fps, frames = parse_fps_frames_from_name(path)

    score = 0.0
    if "rootRot_X90" in name:
        score += 50
    elif "rootRot" in name:
        score += 30
    if "neckHeadHalf" in name:
        score += 50
    elif "neckHead" in name:
        score += 30
    if "blenderFullFrames" in name:
        score += 20
    if "importAs30fps" in name:
        score += 5

    # Prefer original/high-FPS, full-frame files if several candidates exist.
    score += fps / 1000.0
    score += frames / 1_000_000.0
    return score


def find_final_smplx_npz(project_root, trial):
    """
    Auto-detect the final render-ready SMPL-X NPZ for a trial.

    Accepts both the older name:
        <trial>_smplx_fitted_150fps_671frames_rootRot_X90_neckHeadHalf.npz

    and the Blender full-frame/import naming variant:
        <trial>_smplx_fitted_150fps_671frames_rootRot_X90_neckHeadHalf_blenderFullFrames_importAs30fps.npz
    """
    input_dir = (
        project_root
        / "data"
        / "processed"
        / "smplx_fitted_npz"
        / "smpl_to_smplx"
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Corrected SMPL-X folder not found:\n{input_dir}")

    candidates = []
    pattern = f"{trial}_smplx_fitted_*fps_*frames*.npz"

    for path in input_dir.glob(pattern):
        name = path.name
        # The final render-ready file should contain at least one correction tag.
        if "neckHeadHalf" not in name and "rootRot" not in name:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No final corrected/render-ready SMPL-X file found for trial {trial!r} in:\n{input_dir}\n\n"
            f"Expected something like:\n"
            f"{trial}_smplx_fitted_150fps_671frames_rootRot_X90_neckHeadHalf*.npz"
        )

    candidates = sorted(candidates, key=corrected_render_file_score, reverse=True)

    if len(candidates) > 1:
        print("\nCandidate corrected render-ready SMPL-X files:")
        for i, candidate in enumerate(candidates, start=1):
            print(f"  {i}. {candidate}")
        print(f"Selected: {candidates[0]}")

    return candidates[0]


# =============================================================================
# Loading helpers
# =============================================================================

def load_bmclab_sequence(path, subject, trial):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    seq = data[subject][trial]

    pose = seq["pose"].astype(np.float32)
    trans = seq["trans"].astype(np.float32)
    beta = seq["beta"].reshape(-1).astype(np.float32)
    fps = int(seq["fps"])

    return pose, trans, beta, fps


def load_final_smplx_npz(path):
    data = dict(np.load(path, allow_pickle=True))

    poses = data["poses"].astype(np.float32)

    if "trans" in data:
        trans = data["trans"].astype(np.float32)
    elif "transl" in data:
        trans = data["transl"].astype(np.float32)
    else:
        raise KeyError(f"No trans/transl found in {path}")

    if "betas" in data:
        betas = data["betas"].reshape(-1).astype(np.float32)
    elif "beta" in data:
        betas = data["beta"].reshape(-1).astype(np.float32)
    else:
        raise KeyError(f"No betas/beta found in {path}")

    fps = None
    for key in ["true_fps", "source_fps", "original_fps", "fps", "mocap_framerate"]:
        if key in data:
            fps = int(np.asarray(data[key]).item())
            break

    return poses, trans, betas, fps, data


# =============================================================================
# Model construction and joint reconstruction
# =============================================================================

def build_smpl_model(model_root):
    model = smplx.create(
        model_path=str(model_root),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
        num_betas=10,
    ).to(DEVICE)
    model.eval()
    return model


def build_smplx_model(model_root):
    model = smplx.create(
        model_path=str(model_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        batch_size=1,
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
    ).to(DEVICE)
    model.eval()
    return model


def smpl_joints_from_pose_trans_beta(model, pose, trans, beta, batch_size=128):
    num_frames = pose.shape[0]
    beta_full = np.repeat(beta.reshape(1, 10), num_frames, axis=0).astype(np.float32)

    all_joints = []

    with torch.no_grad():
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)

            pose_t = torch.tensor(pose[start:end], dtype=DTYPE, device=DEVICE)
            trans_t = torch.tensor(trans[start:end], dtype=DTYPE, device=DEVICE)
            beta_t = torch.tensor(beta_full[start:end], dtype=DTYPE, device=DEVICE)

            out = model(
                betas=beta_t,
                global_orient=pose_t[:, 0:3],
                body_pose=pose_t[:, 3:72],
                transl=trans_t,
                return_verts=False,
            )

            all_joints.append(out.joints[:, :24, :].detach().cpu().numpy())

    return np.concatenate(all_joints, axis=0).astype(np.float32)


def smplx_joints_from_npz(model, poses, trans, betas, batch_size=64):
    num_frames = poses.shape[0]
    beta_full = np.repeat(betas.reshape(1, 10), num_frames, axis=0).astype(np.float32)

    global_orient = poses[:, 0:3]
    body_pose = poses[:, 3:66]
    jaw_pose = poses[:, 66:69]
    leye_pose = poses[:, 69:72]
    reye_pose = poses[:, 72:75]
    left_hand_pose = poses[:, 75:120]
    right_hand_pose = poses[:, 120:165]
    expression = np.zeros((num_frames, 10), dtype=np.float32)

    all_joints = []

    with torch.no_grad():
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)

            out = model(
                betas=torch.tensor(beta_full[start:end], dtype=DTYPE, device=DEVICE),
                global_orient=torch.tensor(global_orient[start:end], dtype=DTYPE, device=DEVICE),
                body_pose=torch.tensor(body_pose[start:end], dtype=DTYPE, device=DEVICE),
                transl=torch.tensor(trans[start:end], dtype=DTYPE, device=DEVICE),
                jaw_pose=torch.tensor(jaw_pose[start:end], dtype=DTYPE, device=DEVICE),
                leye_pose=torch.tensor(leye_pose[start:end], dtype=DTYPE, device=DEVICE),
                reye_pose=torch.tensor(reye_pose[start:end], dtype=DTYPE, device=DEVICE),
                left_hand_pose=torch.tensor(left_hand_pose[start:end], dtype=DTYPE, device=DEVICE),
                right_hand_pose=torch.tensor(right_hand_pose[start:end], dtype=DTYPE, device=DEVICE),
                expression=torch.tensor(expression[start:end], dtype=DTYPE, device=DEVICE),
                return_verts=False,
            )

            all_joints.append(out.joints.detach().cpu().numpy())

    return np.concatenate(all_joints, axis=0).astype(np.float32)


# =============================================================================
# Alignment helpers for validation-only local gait metrics
# =============================================================================

def root_align(joints, root_idx=0):
    return joints - joints[:, root_idx:root_idx + 1, :]


def fit_constant_rotation_root_aligned(pred, gt, root_idx=0):
    """
    Finds one constant rotation R such that root-aligned pred @ R ~= root-aligned gt.
    This is used only so local gait metrics are not dominated by the render-coordinate correction.
    """
    pred_ra = root_align(pred, root_idx)
    gt_ra = root_align(gt, root_idx)

    source = pred_ra.reshape(-1, 3)
    target = gt_ra.reshape(-1, 3)

    h = source.T @ target
    u, s, vt = np.linalg.svd(h)
    r = u @ vt

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = u @ vt

    aligned = (source @ r).reshape(pred_ra.shape)

    before = np.linalg.norm(pred_ra - gt_ra, axis=-1).mean(axis=1) * UNIT_SCALE
    after = np.linalg.norm(aligned - gt_ra, axis=-1).mean(axis=1) * UNIT_SCALE

    return aligned.astype(np.float32), {
        "before_constant_rotation_mean_mm": float(before.mean()),
        "after_constant_rotation_mean_mm": float(after.mean()),
        "constant_rotation_improvement_percent": float(
            100.0 * (before.mean() - after.mean()) / max(before.mean(), 1e-8)
        ),
        "rotation_matrix_pred_to_gt": r.tolist(),
        "det_rotation": float(np.linalg.det(r)),
    }


def save_recommended_validation_outputs(rec, out_dir, prefix, metrics, root_errors, pa_errors, knee_per_frame, step_stride_details, clearance_details):
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"{prefix}_summary.json"
    per_frame_path = out_dir / f"{prefix}_per_frame_pose_errors.csv"
    recommended_csv = out_dir / f"{prefix}_final_recommended_metrics.csv"
    events_csv = out_dir / f"{prefix}_gait_events_steps_strides.csv"
    knee_csv = out_dir / f"{prefix}_knee_angles.csv"
    clearance_csv = out_dir / f"{prefix}_foot_clearance.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(rec.make_json_safe(metrics), f, indent=2)

    np.savetxt(
        per_frame_path,
        np.column_stack([np.arange(len(root_errors)), root_errors, pa_errors]),
        delimiter=",",
        header="frame,root_aligned_mpjpe_mm,pa_mpjpe_mm",
        comments="",
        fmt=["%d", "%.6f", "%.6f"],
    )

    rec.save_final_recommended_metrics_csv(recommended_csv, metrics["metrics"])
    rec.save_events_csv(events_csv, step_stride_details)
    rec.save_knee_angles_csv(knee_csv, knee_per_frame)
    rec.save_foot_clearance_csv(clearance_csv, clearance_details)

    return {
        "summary_json": summary_path,
        "per_frame_csv": per_frame_path,
        "recommended_metrics_csv": recommended_csv,
        "events_csv": events_csv,
        "knee_angles_csv": knee_csv,
        "foot_clearance_csv": clearance_csv,
    }


# =============================================================================
# Any-subject / any-trial standalone runner
# =============================================================================

def extract_trial_from_corrected_npz(path):
    """Extracts the BMCLab trial name from a corrected SMPL-X file name."""
    name = Path(path).name
    marker = "_smplx_fitted_"
    if marker not in name:
        return None
    return name.split(marker, 1)[0]


def infer_subject_from_trial(trial):
    """Infers subject code from a trial name such as SUB02_off_walk_2."""
    match = re.match(r"^(SUB\d+)_", trial)
    return match.group(1) if match else None


def get_corrected_input_dir(project_root):
    return (
        project_root
        / "data"
        / "processed"
        / "smplx_fitted_npz"
        / "smpl_to_smplx"
    )


def is_corrected_render_ready_npz(path):
    name = Path(path).name
    if not name.endswith(".npz"):
        return False
    if "_smplx_fitted_" not in name:
        return False
    # The render-ready files produced by your correction step contain these tags.
    return ("rootRot" in name) or ("neckHeadHalf" in name)


def find_available_corrected_trials(project_root, subject=None):
    """
    Finds available corrected/render-ready SMPL-X NPZ files.

    Returns:
        dict[str, Path]: trial -> best corrected file for that trial
    """
    input_dir = get_corrected_input_dir(project_root)

    if not input_dir.exists():
        raise FileNotFoundError(f"Corrected SMPL-X folder not found:\n{input_dir}")

    grouped = {}

    if subject:
        pattern = f"{subject}_*_smplx_fitted_*fps_*frames*.npz"
    else:
        pattern = "SUB*_smplx_fitted_*fps_*frames*.npz"

    for path in input_dir.glob(pattern):
        if not is_corrected_render_ready_npz(path):
            continue
        trial = extract_trial_from_corrected_npz(path)
        if trial is None:
            continue
        if subject and not trial.startswith(subject + "_"):
            continue
        grouped.setdefault(trial, []).append(path)

    best_by_trial = {}
    for trial, candidates in grouped.items():
        best_by_trial[trial] = sorted(
            candidates,
            key=corrected_render_file_score,
            reverse=True,
        )[0]

    return dict(sorted(best_by_trial.items()))


def print_available_trials(files_by_trial):
    print("\nAvailable corrected/render-ready trials")
    print("---------------------------------------")

    if not files_by_trial:
        print("No corrected/render-ready trials found.")
        return

    for trial, path in files_by_trial.items():
        fps, frames = parse_fps_frames_from_name(path)
        print(f"{trial:<32} fps={fps:>7}  frames={frames:>7}")
        print(f"    {path}")


def validate_one(subject, trial, final_smplx_npz):
    """Runs the full validation directly. This function does not call another script."""

    final_smplx_npz = Path(final_smplx_npz)

    out_dir = (
        PROJECT_ROOT
        / "results"
        / "render_input_validation"
        / f"{subject}_{trial}"
    )

    rec, rec_path = load_embedded_metric_functions()

    print("\nConfiguration")
    print("-------------")
    print("BMCLab pkl:             ", BMCLAB_PKL)
    print("Final SMPL-X npz:       ", final_smplx_npz)
    print("Model root:             ", MODEL_ROOT)
    print("Recommended metric file:", rec_path)
    print("Subject:                ", subject)
    print("Trial:                  ", trial)
    print("Device:                 ", DEVICE)

    if not BMCLAB_PKL.exists():
        raise FileNotFoundError(f"BMCLab pkl not found:\n{BMCLAB_PKL}")
    if not final_smplx_npz.exists():
        raise FileNotFoundError(f"Final SMPL-X npz not found:\n{final_smplx_npz}")

    print("\nLoading input data...")
    smpl_pose, smpl_trans, smpl_beta, bmclab_fps = load_bmclab_sequence(BMCLAB_PKL, subject, trial)
    smplx_poses, smplx_trans, smplx_betas, smplx_fps, _ = load_final_smplx_npz(final_smplx_npz)

    num_frames = min(smpl_pose.shape[0], smplx_poses.shape[0])
    fps_used = bmclab_fps

    smpl_pose = smpl_pose[:num_frames]
    smpl_trans = smpl_trans[:num_frames]
    smplx_poses = smplx_poses[:num_frames]
    smplx_trans = smplx_trans[:num_frames]

    print("BMCLab pose/trans/fps:", smpl_pose.shape, smpl_trans.shape, bmclab_fps)
    print("SMPL-X poses/trans/fps:", smplx_poses.shape, smplx_trans.shape, smplx_fps)
    print("Frames compared:", num_frames)

    if smplx_fps is not None and int(smplx_fps) != int(bmclab_fps):
        print(
            "WARNING: BMCLab fps and SMPL-X stored fps differ. "
            f"Using BMCLab fps for metric timing: {bmclab_fps}."
        )

    print("\nBuilding models...")
    smpl_model = build_smpl_model(MODEL_ROOT)
    smplx_model = build_smplx_model(MODEL_ROOT)

    print("\nReconstructing joints...")
    smpl_joints = smpl_joints_from_pose_trans_beta(smpl_model, smpl_pose, smpl_trans, smpl_beta)
    smplx_joints_all = smplx_joints_from_npz(smplx_model, smplx_poses, smplx_trans, smplx_betas)

    gt = smpl_joints[:, :NUM_COMMON_JOINTS, :]
    pred = smplx_joints_all[:, :NUM_COMMON_JOINTS, :]

    print("GT joints:", gt.shape)
    print("PRED joints:", pred.shape)

    print("\nComputing pose metrics on raw joints...")
    root_summary, root_errors = rec.compute_root_aligned_mpjpe(
        pred, gt, root_idx=ROOT_IDX, unit_scale=UNIT_SCALE
    )
    pa_summary, pa_errors = rec.compute_pa_mpjpe(pred, gt, unit_scale=UNIT_SCALE)

    print("\nAligning local body coordinates for recommended gait metrics...")
    pred_local_aligned, rotation_diagnostic = fit_constant_rotation_root_aligned(
        pred, gt, root_idx=ROOT_IDX
    )
    gt_local = root_align(gt, ROOT_IDX)

    print("\nSmoothing joints for event/clearance/step metrics...")
    gt_gait = rec.moving_average_trajectory(gt_local, window=SMOOTHING_WINDOW)
    pred_gait = rec.moving_average_trajectory(pred_local_aligned, window=SMOOTHING_WINDOW)

    vertical_axis, body_extents = rec.infer_vertical_axis_from_joints(gt_gait)
    forward_axis = rec.compute_forward_axis_from_pelvis(
        gt_gait, vertical_axis, root_idx=ROOT_IDX
    )

    gt_events, gt_contact_info = rec.build_gait_events(
        gt_gait, forward_axis, vertical_axis, fps_used, pelvis_relative=True
    )
    pred_events, pred_contact_info = rec.build_gait_events(
        pred_gait, forward_axis, vertical_axis, fps_used, pelvis_relative=True
    )

    step_stride_summary, step_stride_details = rec.compute_step_stride_metrics(
        gt_joints=gt_gait,
        wham_joints_aligned=pred_gait,
        forward_axis=forward_axis,
        vertical_axis=vertical_axis,
        fps=fps_used,
    )

    knee_summary, knee_per_frame = rec.compute_knee_rom_metrics(gt_gait, pred_gait)
    timing_summary = rec.compute_contact_timing_metrics(gt_events, pred_events, fps=fps_used)
    cadence_summary = rec.compute_cadence_metrics(
        gt_events, pred_events, n_frames=num_frames, fps=fps_used
    )
    clearance_summary, clearance_details = rec.compute_foot_clearance_metrics(
        gt_gait, pred_gait, gt_events, pred_events, vertical_axis, unit_scale=UNIT_SCALE
    )

    metrics_summary = {
        **root_summary,
        **pa_summary,
        **knee_summary,
        **timing_summary,
        **cadence_summary,
        **clearance_summary,
        **step_stride_summary,
        "frames_compared": int(num_frames),
        "fps_used": float(fps_used),
        "num_common_joints": int(NUM_COMMON_JOINTS),
        "gait_smoothing_window_frames": int(SMOOTHING_WINDOW),
        "vertical_axis": int(vertical_axis),
        "body_axis_extents": body_extents.tolist(),
        "forward_axis": forward_axis.tolist(),
        "metric_context": "BMCLab SMPL versus final render-input SMPL-X before WHAM",
        "note_on_gait_alignment": (
            "Recommended gait metrics are computed after one constant root-aligned rotation, "
            "so they measure local motion preservation rather than render-coordinate-frame changes."
        ),
        "local_rotation_diagnostic": rotation_diagnostic,
        "contact_detection": {
            "gt": gt_contact_info,
            "pred": pred_contact_info,
        },
    }

    full_summary = {
        "configuration": {
            "bmclab_pkl": str(BMCLAB_PKL),
            "final_smplx_npz": str(final_smplx_npz),
            "model_root": str(MODEL_ROOT),
            "subject": subject,
            "trial": trial,
            "embedded_metric_functions": str(rec_path),
        },
        "metrics": metrics_summary,
        "interpretation_rules": {
            "motion_preserved_before_wham": (
                "Supported if PA-MPJPE, knee ROM error, timing error, cadence error, "
                "foot clearance error, and pelvis-relative step/stride errors are small before WHAM."
            ),
            "render_input_problem": (
                "Likely if these recommended metrics are already large before WHAM."
            ),
            "coordinate_frame_problem": (
                "Likely if root-aligned MPJPE is high but the local constant-rotation diagnostic improves strongly."
            ),
        },
    }

    saved = save_recommended_validation_outputs(
        rec=rec,
        out_dir=out_dir,
        prefix="render_input_motion_validation_recommended",
        metrics=full_summary,
        root_errors=root_errors,
        pa_errors=pa_errors,
        knee_per_frame=knee_per_frame,
        step_stride_details=step_stride_details,
        clearance_details=clearance_details,
    )

    print("\nMain recommended metrics")
    print("------------------------")
    print(f"PA-MPJPE:                              {metrics_summary['pa_mpjpe_mean_mm']:.3f} mm")
    print(f"Knee ROM error:                       {metrics_summary['knee_rom_mean_abs_error_deg']:.3f} deg")
    print(f"Step/contact timing error:            {metrics_summary['step_contact_timing_mae_s']} s")
    print(f"Cadence error:                        {metrics_summary['cadence_abs_error_steps_per_min']:.3f} steps/min")
    print(f"Foot clearance error:                 {metrics_summary['foot_clearance_mean_abs_error_mm']} mm")
    print(f"Pelvis-relative step length error:    {metrics_summary['step_length_mean_abs_error_mm']} mm")
    print(f"Pelvis-relative stride length error:  {metrics_summary['stride_length_mean_abs_error_mm']} mm")

    print("\nSaved outputs")
    print("-------------")
    for label, path in saved.items():
        print(f"{label}: {path}")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone render-input validation for any BMCLab subject/trial that has "
            "a corrected render-ready SMPL-X NPZ file."
        )
    )
    parser.add_argument("--subject", default=SUBJECT, help="BMCLab subject, e.g. SUB01 or SUB02")
    parser.add_argument(
        "--trial",
        default=None,
        help=(
            "BMCLab trial, e.g. SUB02_off_walk_2. If omitted, the script uses TRIAL "
            "from the settings section when available, otherwise the first available trial."
        ),
    )
    parser.add_argument(
        "--final-smplx-npz",
        default=None,
        help="Optional explicit corrected/render-ready SMPL-X NPZ path.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list available corrected/render-ready trials for the selected subject.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all available corrected/render-ready trials for the selected subject.",
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Validate all available corrected/render-ready trials for all subjects.",
    )

    args = parser.parse_args()

    if args.final_smplx_npz is not None and args.trial is None:
        trial_from_file = extract_trial_from_corrected_npz(Path(args.final_smplx_npz))
        if trial_from_file is None:
            raise ValueError(
                "When using --final-smplx-npz with a non-standard file name, also provide --trial."
            )
        args.trial = trial_from_file

    if args.all_subjects:
        files_by_trial = find_available_corrected_trials(PROJECT_ROOT, subject=None)
        print_available_trials(files_by_trial)
        if not files_by_trial:
            raise FileNotFoundError("No corrected/render-ready SMPL-X files found for any subject.")

        for trial, path in files_by_trial.items():
            subject = infer_subject_from_trial(trial)
            if subject is None:
                print(f"Skipping trial with unknown subject format: {trial}")
                continue
            print("\n" + "=" * 80)
            print(f"Validating {subject} / {trial}")
            print("=" * 80)
            validate_one(subject, trial, path)
        return

    subject = args.subject
    files_by_trial = find_available_corrected_trials(PROJECT_ROOT, subject=subject)

    if args.list:
        print_available_trials(files_by_trial)
        return

    if args.all:
        print_available_trials(files_by_trial)
        if not files_by_trial:
            raise FileNotFoundError(
                f"No corrected/render-ready SMPL-X files found for subject {subject}."
            )
        for trial, path in files_by_trial.items():
            print("\n" + "=" * 80)
            print(f"Validating {subject} / {trial}")
            print("=" * 80)
            validate_one(subject, trial, path)
        return

    if args.final_smplx_npz is not None:
        trial = args.trial
        file_path = Path(args.final_smplx_npz)
        validate_one(subject, trial, file_path)
        return

    if not files_by_trial:
        raise FileNotFoundError(
            f"No corrected/render-ready SMPL-X files found for subject {subject} in:\n"
            f"{get_corrected_input_dir(PROJECT_ROOT)}\n\n"
            "Run the SMPL-to-SMPLX fitting/correction step for this subject first."
        )

    requested_trial = args.trial

    if requested_trial is None:
        if TRIAL in files_by_trial and TRIAL.startswith(subject + "_"):
            requested_trial = TRIAL
        else:
            requested_trial = next(iter(files_by_trial.keys()))
            print(
                f"\nNo --trial was provided. Using first available corrected trial: {requested_trial}"
            )

    if requested_trial not in files_by_trial:
        print_available_trials(files_by_trial)
        raise FileNotFoundError(
            f"Requested trial not found as a corrected/render-ready file: {requested_trial}\n\n"
            "Check the available trials printed above and copy the trial name exactly."
        )

    validate_one(subject, requested_trial, files_by_trial[requested_trial])


if __name__ == "__main__":
    main()
