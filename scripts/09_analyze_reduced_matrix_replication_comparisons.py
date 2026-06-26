"""
Analyze reduced-matrix WHAM experiment results for replication walking sequences.

This script is the reduced-matrix equivalent of script 07. It does NOT rerun
SMPL, WHAM, or BMCLab reconstruction. It reads the summary/per-frame outputs
created by the script 06 batch metric pipeline and performs only the comparisons
needed for the reduced replication matrix.

Reduced matrix assumed by default
---------------------------------
R0 = neutral technical baseline
R1 = home/living-room baseline
R2 = best/stable frontal setup
R3 = upper-corner viewpoint
R4 = strong lower-body occlusion
R5 = partial frontal one-leg occlusion

If your script 06 outputs still use the original IDs, this script automatically
maps:
    E0 -> R0
    E1 -> R1
    E3 -> R2
    E4 -> R3
    E8 -> R4
    E12 -> R5

Main comparison logic
---------------------
1. R1 - R0: room complexity / home baseline effect
2. R2 - R1: stable frontal setup relative to home baseline
3. R3 - R2: camera-view effect, upper corner versus best frontal setup
4. R4 - R1: severe lower-body occlusion versus home baseline
5. R5 - R1: partial frontal occlusion versus home baseline
6. R5 - R4: recoverability of partial occlusion versus strong occlusion

Example run
-----------
From project root:

    conda activate metrics
    cd C:\\Users\\sopha\\synthetic-pd

    python scripts\\08_analyze_reduced_matrix_replication_comparisons.py ^
        --subject SUB02 ^
        --trial SUB02_off_walk_2

Optional explicit summary CSV:

    python scripts\\08_analyze_reduced_matrix_replication_comparisons.py ^
        --subject SUB08 ^
        --trial SUB08_off_walk_4 ^
        --summary-csv results\\clinically_relevant_metrics_v1\\batch_wham_experiments\\SUB08\\SUB08_off_walk_4\\SUB08_SUB08_off_walk_4_all_experiments_metric_summary.csv

Outputs
-------
Saved to:
    results/clinically_relevant_metrics_v1/batch_wham_experiments/<SUBJECT>/<SEQUENCE>/reduced_matrix_analysis/

Core CSVs:
    reduced_matrix_key_results_table.csv
    reduced_matrix_pairwise_comparisons.csv
    reduced_matrix_reliability_classification.csv
    reduced_matrix_condition_map_template.csv
    reduced_matrix_frame_level_pairwise_summary.csv
    reduced_matrix_frame_level_pairwise_deltas_selected.csv

Plots:
    reduced_pairwise_delta_pa_mpjpe.png
    reduced_pairwise_delta_foot_clearance.png
    reduced_pairwise_delta_stride_length.png
    reduced_pairwise_delta_contact_timing.png
    reduced_frame_pa_mpjpe_selected.png
    reduced_frame_delta_pa_mpjpe_key_comparisons.png
    reduced_scatter_pa_mpjpe_vs_stride_length.png

Interpretation note
-------------------
Positive deltas mean the test condition has a higher error than the baseline.
Negative deltas mean the test condition has a lower error than the baseline.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

ORIGINAL_TO_REDUCED = {
    "E0": "R0",
    "E1": "R1",
    "E3": "R2",
    "E4": "R3",
    "E8": "R4",
    "E12": "R5",
}

REDUCED_TO_ORIGINAL = {v: k for k, v in ORIGINAL_TO_REDUCED.items()}

DEFAULT_CONDITION_ROWS = [
    {
        "reduced_id": "R0",
        "original_id": "E0",
        "purpose": "Neutral baseline",
        "environment": "Neutral",
        "lighting": "Bright/day",
        "camera_view": "Frontal/table",
        "occlusion": "None",
        "why_include": "Technical baseline",
    },
    {
        "reduced_id": "R1",
        "original_id": "E1",
        "purpose": "Home baseline",
        "environment": "Living room with table",
        "lighting": "Bright/day",
        "camera_view": "Frontal/table",
        "occlusion": "None",
        "why_include": "Checks room complexity",
    },
    {
        "reduced_id": "R2",
        "original_id": "E3",
        "purpose": "Best/stable frontal setup",
        "environment": "Living room without table",
        "lighting": "Bright/day",
        "camera_view": "Frontal/table 0.95 m",
        "occlusion": "None",
        "why_include": "Best-performing setup from SUB01",
    },
    {
        "reduced_id": "R3",
        "original_id": "E4",
        "purpose": "Upper-corner viewpoint",
        "environment": "Living room without table",
        "lighting": "Bright/day",
        "camera_view": "Upper corner",
        "occlusion": "None",
        "why_include": "Tests camera-angle failure without occlusion",
    },
    {
        "reduced_id": "R4",
        "original_id": "E8",
        "purpose": "Strong occlusion",
        "environment": "Living room with table",
        "lighting": "Bright/day",
        "camera_view": "Frontal/table",
        "occlusion": "Both-leg occlusion",
        "why_include": "Tests severe lower-body visibility loss",
    },
    {
        "reduced_id": "R5",
        "original_id": "E12",
        "purpose": "Partial occlusion, frontal",
        "environment": "Living room with table",
        "lighting": "Bright/day",
        "camera_view": "Frontal/table",
        "occlusion": "One-leg occlusion, partial frames",
        "why_include": "Tests whether partial occlusion remains recoverable",
    },
]

# Pairwise comparisons used for the reduced matrix.
DEFAULT_COMPARISON_ROWS = [
    {
        "comparison_id": "C1",
        "test_id": "R1",
        "baseline_id": "R0",
        "comparison_name": "Home baseline vs neutral baseline",
        "question_answered": "Does room complexity/home environment affect recovered motion?",
    },
    {
        "comparison_id": "C2",
        "test_id": "R2",
        "baseline_id": "R1",
        "comparison_name": "Stable frontal setup vs home baseline",
        "question_answered": "Does the best frontal setup remain close to the home baseline?",
    },
    {
        "comparison_id": "C3",
        "test_id": "R3",
        "baseline_id": "R2",
        "comparison_name": "Upper-corner view vs stable frontal setup",
        "question_answered": "Does camera viewpoint affect recovered motion?",
    },
    {
        "comparison_id": "C4",
        "test_id": "R4",
        "baseline_id": "R1",
        "comparison_name": "Strong occlusion vs home baseline",
        "question_answered": "How much degradation is introduced by severe lower-body occlusion?",
    },
    {
        "comparison_id": "C5",
        "test_id": "R5",
        "baseline_id": "R1",
        "comparison_name": "Partial frontal occlusion vs home baseline",
        "question_answered": "Does partial one-leg occlusion remain recoverable?",
    },
    {
        "comparison_id": "C6",
        "test_id": "R5",
        "baseline_id": "R4",
        "comparison_name": "Partial occlusion vs strong occlusion",
        "question_answered": "Is partial occlusion less harmful than full lower-body occlusion?",
    },
]

KEY_ERROR_METRICS = [
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
]

THESIS_METRICS = [
    "pa_mpjpe_mean_mm",
    "step_contact_timing_mae_s",
    "cadence_abs_error_steps_per_min",
    "foot_clearance_mean_abs_error_mm",
    "step_length_mean_abs_error_mm",
    "stride_length_mean_abs_error_mm",
    "knee_rom_mean_abs_error_deg",
    "pelvis_trajectory_error_mean_mm",
]

CLINICAL_SCORE_METRICS = [
    "pa_mpjpe_mean_mm",
    "knee_rom_mean_abs_error_deg",
    "step_contact_timing_mae_s",
    "cadence_abs_error_steps_per_min",
    "foot_clearance_mean_abs_error_mm",
    "step_length_mean_abs_error_mm",
    "stride_length_mean_abs_error_mm",
]

PER_FRAME_METRICS = [
    "root_aligned_mpjpe_mm",
    "pa_mpjpe_mm",
    "pelvis_trajectory_error_mm",
    "pelvis_forward_error_mm",
    "pelvis_lateral_error_mm",
    "pelvis_vertical_error_mm",
]


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def id_number(exp: str) -> int:
    exp = str(exp)
    if len(exp) >= 2 and exp[0].upper() in {"R", "E"}:
        try:
            return int(exp[1:])
        except ValueError:
            pass
    return 10_000


def sort_by_reduced_id(df: pd.DataFrame) -> pd.DataFrame:
    if "reduced_id" not in df.columns:
        return df
    df = df.copy()
    df["reduced_number"] = df["reduced_id"].apply(id_number)
    return df.sort_values(["reduced_number", "reduced_id"]).reset_index(drop=True)


def to_numeric_columns(df: pd.DataFrame, exclude: Sequence[str] = ("experiment", "reduced_id", "subject", "sequence")) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in exclude:
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def find_summary_csv(repo_root: Path, subject: str, sequence: str, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Summary CSV not found:\n{explicit}")
        return explicit

    default = (
        repo_root
        / "results"
        / "clinically_relevant_metrics_v1"
        / "batch_wham_experiments"
        / subject
        / sequence
        / f"{subject}_{sequence}_all_experiments_metric_summary.csv"
    )
    if default.exists():
        return default

    search_root = repo_root / "results" / "clinically_relevant_metrics_v1"
    candidates = list(search_root.rglob(f"*{subject}*{sequence}*all_experiments_metric_summary.csv"))
    if candidates:
        candidates = sorted(candidates, key=lambda p: len(str(p)))
        return candidates[0]

    raise FileNotFoundError(
        "Could not find the combined experiment summary CSV. Expected for example:\n"
        f"{default}\n\n"
        "Run script 06 batch first, or pass --summary-csv explicitly."
    )


def load_condition_map(condition_map_csv: Optional[Path], output_dir: Path) -> pd.DataFrame:
    template = pd.DataFrame(DEFAULT_CONDITION_ROWS)
    template.to_csv(output_dir / "reduced_matrix_condition_map_template.csv", index=False)

    if condition_map_csv is None:
        return template

    if not condition_map_csv.exists():
        raise FileNotFoundError(f"Condition map CSV not found:\n{condition_map_csv}")

    df = pd.read_csv(condition_map_csv)

    # Allow either reduced_id or experiment as the identifying column.
    if "reduced_id" not in df.columns:
        if "experiment" in df.columns:
            df["reduced_id"] = df["experiment"].astype(str).map(ORIGINAL_TO_REDUCED).fillna(df["experiment"].astype(str))
        else:
            raise ValueError("Condition map must contain either 'reduced_id' or 'experiment'.")

    if "original_id" not in df.columns:
        df["original_id"] = df["reduced_id"].astype(str).map(REDUCED_TO_ORIGINAL).fillna("unknown")

    for col in ["purpose", "environment", "lighting", "camera_view", "occlusion", "why_include"]:
        if col not in df.columns:
            df[col] = "unknown"

    return df


def add_reduced_ids(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["experiment"] = df["experiment"].astype(str)

    # If already R0-R5, keep them. If original IDs are used, map to reduced IDs.
    df["reduced_id"] = df["experiment"].apply(lambda x: x if str(x).upper().startswith("R") else ORIGINAL_TO_REDUCED.get(str(x), str(x)))
    df["original_id"] = df["experiment"].apply(lambda x: REDUCED_TO_ORIGINAL.get(str(x), str(x)))

    # Keep only rows that belong to the reduced matrix.
    valid_reduced = {row["reduced_id"] for row in DEFAULT_CONDITION_ROWS}
    reduced = df[df["reduced_id"].isin(valid_reduced)].copy()
    if reduced.empty:
        raise ValueError(
            "No reduced-matrix experiments found in summary. Expected R0-R5 or original IDs E0,E1,E3,E4,E8,E12."
        )
    return sort_by_reduced_id(reduced)


def merge_condition_map(summary: pd.DataFrame, condition_map: pd.DataFrame) -> pd.DataFrame:
    out = summary.merge(condition_map, on="reduced_id", how="left", suffixes=("", "_condition"))

    # Prefer original_id from summary when available.
    if "original_id_condition" in out.columns:
        out["original_id"] = out["original_id"].fillna(out["original_id_condition"])

    for col in ["purpose", "environment", "lighting", "camera_view", "occlusion", "why_include"]:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown")

    return sort_by_reduced_id(out)


def metric_value(row: pd.Series, metric: str) -> float:
    if metric not in row.index:
        return np.nan
    return pd.to_numeric(pd.Series([row[metric]]), errors="coerce").iloc[0]


# =============================================================================
# Summary comparisons
# =============================================================================

def make_key_results_table(summary: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "reduced_id", "original_id", "purpose", "environment", "lighting", "camera_view", "occlusion",
        "frames_used", "gt_num_gait_events", "wham_num_gait_events", "gait_event_count_difference",
    ]
    for metric in THESIS_METRICS:
        if metric in summary.columns:
            keep.append(metric)
    keep = [c for c in keep if c in summary.columns]
    return sort_by_reduced_id(summary[keep].copy())


def compute_pairwise_comparisons(summary: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_id = {str(row["reduced_id"]): row for _, row in summary.iterrows()}

    for _, comp in comparisons.iterrows():
        test_id = str(comp["test_id"])
        base_id = str(comp["baseline_id"])

        if test_id not in by_id or base_id not in by_id:
            rows.append({
                "comparison_id": comp["comparison_id"],
                "comparison_name": comp["comparison_name"],
                "test_id": test_id,
                "baseline_id": base_id,
                "status": "missing_test_or_baseline",
            })
            continue

        test = by_id[test_id]
        base = by_id[base_id]

        row = {
            "comparison_id": comp["comparison_id"],
            "comparison_name": comp["comparison_name"],
            "question_answered": comp["question_answered"],
            "test_id": test_id,
            "baseline_id": base_id,
            "test_purpose": test.get("purpose", ""),
            "baseline_purpose": base.get("purpose", ""),
            "status": "ok",
        }

        test_frames = metric_value(test, "frames_used")
        base_frames = metric_value(base, "frames_used")
        row["test_frames_used"] = test_frames
        row["baseline_frames_used"] = base_frames
        if not pd.isna(test_frames) and not pd.isna(base_frames) and base_frames > 0:
            row["frames_ratio_test_vs_baseline"] = test_frames / base_frames
            row["comparable_frame_count"] = 0.95 <= row["frames_ratio_test_vs_baseline"] <= 1.05
        else:
            row["frames_ratio_test_vs_baseline"] = np.nan
            row["comparable_frame_count"] = True

        for metric in KEY_ERROR_METRICS:
            if metric not in summary.columns:
                continue
            test_val = metric_value(test, metric)
            base_val = metric_value(base, metric)
            row[f"test_{metric}"] = test_val
            row[f"baseline_{metric}"] = base_val
            row[f"delta_{metric}"] = test_val - base_val

        rows.append(row)

    out = pd.DataFrame(rows)
    return out


def compute_reliability_classification(summary: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight rule-based classification for thesis interpretation.
    It is deliberately transparent and should be treated as descriptive, not clinical scoring.
    """
    rows = []

    for _, row in summary.iterrows():
        rid = str(row["reduced_id"])
        purpose = str(row.get("purpose", ""))
        frames = metric_value(row, "frames_used")
        gt_events = metric_value(row, "gt_num_gait_events")
        wham_events = metric_value(row, "wham_num_gait_events")
        pa = metric_value(row, "pa_mpjpe_mean_mm")
        fc = metric_value(row, "foot_clearance_mean_abs_error_mm")
        stride = metric_value(row, "stride_length_mean_abs_error_mm")
        cadence = metric_value(row, "cadence_abs_error_steps_per_min")

        event_ok = (not pd.isna(gt_events) and not pd.isna(wham_events) and int(gt_events) == int(wham_events))
        frame_ok = True
        if "source_num_frames" in row.index and not pd.isna(row["source_num_frames"]):
            source_frames = metric_value(row, "source_num_frames")
            frame_ok = pd.isna(source_frames) or source_frames <= 0 or frames >= 0.95 * source_frames

        # Pose reliability based on PA-MPJPE only, using broad descriptive thresholds.
        if pd.isna(pa):
            pose_rel = "unknown"
        elif pa < 30:
            pose_rel = "good by PA-MPJPE"
        elif pa < 40:
            pose_rel = "moderate"
        else:
            pose_rel = "poor/moderate"

        # Gait-feature reliability using key gait features.
        clinical_flags = []
        if event_ok:
            clinical_flags.append("correct event count")
        else:
            clinical_flags.append("event-count mismatch")
        if not pd.isna(fc) and fc <= 30:
            clinical_flags.append("low/moderate foot-clearance error")
        if not pd.isna(stride) and stride <= 60:
            clinical_flags.append("low/moderate stride error")
        if not pd.isna(cadence) and cadence <= 1:
            clinical_flags.append("no cadence error")
        if not frame_ok:
            clinical_flags.append("short/incomplete track")

        if not frame_ok:
            clinical_rel = "not reliable"
            recommendation = "treat as tracking failure"
        elif event_ok and (not pd.isna(fc) and fc <= 30) and (not pd.isna(stride) and stride <= 60):
            clinical_rel = "good"
            recommendation = "recommended/acceptable"
        elif not event_ok or (not pd.isna(fc) and fc > 70) or (not pd.isna(stride) and stride > 150):
            clinical_rel = "poor"
            recommendation = "avoid for clinical gait metrics"
        else:
            clinical_rel = "moderate"
            recommendation = "interpret cautiously"

        if rid in {"R0", "R1"}:
            recommendation = "baseline reference"
        if rid == "R3":
            recommendation = "avoid if lower-limb gait metrics are primary"
        if rid == "R4":
            recommendation = "avoid; tests severe visibility loss"
        if rid == "R5" and clinical_rel == "good":
            recommendation = "acceptable if full visibility is not possible"

        rows.append({
            "reduced_id": rid,
            "original_id": row.get("original_id", ""),
            "purpose": purpose,
            "technical_pose_reliability": pose_rel,
            "clinical_gait_feature_reliability": clinical_rel,
            "supporting_observations": "; ".join(clinical_flags),
            "recommendation": recommendation,
        })

    return sort_by_reduced_id(pd.DataFrame(rows))


# =============================================================================
# Per-frame comparisons
# =============================================================================

def existing_path(value) -> Optional[Path]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    p = Path(str(value))
    if p.exists():
        return p
    return None


def find_per_frame_csv(row: pd.Series, subject_out_root: Path) -> Optional[Path]:
    # Prefer explicit path from summary if present.
    for key in ["per_frame_csv", "per_frame_path"]:
        if key in row:
            p = existing_path(row[key])
            if p is not None:
                return p

    # Try actual experiment ID folder and reduced ID folder.
    candidates_roots = []
    for key in ["experiment", "reduced_id", "original_id"]:
        if key in row:
            candidates_roots.append(subject_out_root / str(row[key]))

    for root in candidates_roots:
        if root.exists():
            candidates = sorted(root.glob("*per_frame*.csv"))
            if candidates:
                return candidates[0]

    # General recursive search.
    search_tokens = [str(row.get("experiment", "")), str(row.get("reduced_id", "")), str(row.get("original_id", ""))]
    for token in search_tokens:
        if not token:
            continue
        candidates = sorted(subject_out_root.rglob(f"*{token}*per_frame*.csv"))
        if candidates:
            return candidates[0]

    return None


def load_per_frame_tables(summary: pd.DataFrame, subject_out_root: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for _, row in summary.iterrows():
        rid = str(row["reduced_id"])
        p = find_per_frame_csv(row, subject_out_root)
        if p is None:
            print(f"WARNING: per-frame CSV not found for {rid}")
            continue
        df = pd.read_csv(p)
        if "frame" not in df.columns:
            print(f"WARNING: per-frame CSV for {rid} has no frame column: {p}")
            continue
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
        tables[rid] = df
    return tables


def compute_frame_pairwise_deltas(per_frame_tables: Dict[str, pd.DataFrame], comparisons: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_frames = []
    summary_rows = []

    for _, comp in comparisons.iterrows():
        test_id = str(comp["test_id"])
        base_id = str(comp["baseline_id"])
        comp_id = str(comp["comparison_id"])

        if test_id not in per_frame_tables or base_id not in per_frame_tables:
            summary_rows.append({
                "comparison_id": comp_id,
                "comparison_name": comp["comparison_name"],
                "test_id": test_id,
                "baseline_id": base_id,
                "status": "missing_per_frame_table",
            })
            continue

        test = per_frame_tables[test_id].copy()
        base = per_frame_tables[base_id].copy()
        common_metrics = [m for m in PER_FRAME_METRICS if m in test.columns and m in base.columns]
        if not common_metrics:
            summary_rows.append({
                "comparison_id": comp_id,
                "comparison_name": comp["comparison_name"],
                "test_id": test_id,
                "baseline_id": base_id,
                "status": "no_common_per_frame_metrics",
            })
            continue

        test = test[["frame"] + common_metrics]
        base = base[["frame"] + common_metrics]
        merged = test.merge(base, on="frame", how="inner", suffixes=("_test", "_baseline"))
        if merged.empty:
            summary_rows.append({
                "comparison_id": comp_id,
                "comparison_name": comp["comparison_name"],
                "test_id": test_id,
                "baseline_id": base_id,
                "status": "no_overlapping_frames",
            })
            continue

        merged.insert(0, "comparison_id", comp_id)
        merged.insert(1, "comparison_name", comp["comparison_name"])
        merged.insert(2, "test_id", test_id)
        merged.insert(3, "baseline_id", base_id)

        summary = {
            "comparison_id": comp_id,
            "comparison_name": comp["comparison_name"],
            "test_id": test_id,
            "baseline_id": base_id,
            "status": "ok",
            "frames_compared": int(len(merged)),
        }

        for metric in common_metrics:
            delta_col = f"delta_{metric}"
            merged[delta_col] = pd.to_numeric(merged[f"{metric}_test"], errors="coerce") - pd.to_numeric(merged[f"{metric}_baseline"], errors="coerce")
            values = pd.to_numeric(merged[delta_col], errors="coerce")
            abs_values = values.abs()
            summary[f"mean_delta_{metric}"] = float(values.mean())
            summary[f"median_delta_{metric}"] = float(values.median())
            summary[f"max_delta_{metric}"] = float(values.max())
            summary[f"min_delta_{metric}"] = float(values.min())
            summary[f"mean_abs_delta_{metric}"] = float(abs_values.mean())
            if values.notna().any():
                summary[f"frame_of_max_delta_{metric}"] = int(merged.loc[values.idxmax(), "frame"])
            else:
                summary[f"frame_of_max_delta_{metric}"] = None

        summary_rows.append(summary)
        all_frames.append(merged)

    frame_summary = pd.DataFrame(summary_rows)
    frame_deltas = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    return frame_summary, frame_deltas


# =============================================================================
# Plots
# =============================================================================

def save_pairwise_bar(pairwise: pd.DataFrame, delta_col: str, title: str, ylabel: str, output_path: Path) -> None:
    if delta_col not in pairwise.columns:
        print(f"WARNING: Cannot plot {delta_col}; missing column.")
        return
    df = pairwise[pairwise["status"] == "ok"].copy()
    if df.empty:
        return
    labels = df["comparison_id"].astype(str) + ": " + df["test_id"].astype(str) + "-" + df["baseline_id"].astype(str)
    values = pd.to_numeric(df[delta_col], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Comparison")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_summary_scatter(summary: pd.DataFrame, x_col: str, y_col: str, output_path: Path) -> None:
    if x_col not in summary.columns or y_col not in summary.columns:
        return
    x = pd.to_numeric(summary[x_col], errors="coerce")
    y = pd.to_numeric(summary[y_col], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return
    df = summary[valid].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pd.to_numeric(df[x_col]), pd.to_numeric(df[y_col]))
    for _, row in df.iterrows():
        ax.annotate(str(row["reduced_id"]), (row[x_col], row[y_col]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("PA-MPJPE (mm)")
    ax.set_ylabel("Stride-length error (mm)")
    ax.set_title("PA-MPJPE versus stride-length error, reduced matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_frame_metric_plot(per_frame_tables: Dict[str, pd.DataFrame], metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    plotted = False
    for rid in sorted(per_frame_tables.keys(), key=id_number):
        df = per_frame_tables[rid]
        if metric not in df.columns:
            continue
        ax.plot(pd.to_numeric(df["frame"], errors="coerce"), pd.to_numeric(df[metric], errors="coerce"), label=rid, linewidth=1.3)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("Frame-level PA-MPJPE for reduced-matrix conditions")
    ax.set_xlabel("Frame")
    ax.set_ylabel("PA-MPJPE (mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_frame_delta_plot(frame_deltas: pd.DataFrame, metric: str, output_path: Path) -> None:
    delta_col = f"delta_{metric}"
    if frame_deltas.empty or delta_col not in frame_deltas.columns:
        return
    # Plot the most important frame-level comparisons only.
    keep_ids = {"C3", "C4", "C5", "C6"}
    df = frame_deltas[frame_deltas["comparison_id"].isin(keep_ids)].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for comp_id, group in df.groupby("comparison_id"):
        label = f"{comp_id}: {group['test_id'].iloc[0]}-{group['baseline_id'].iloc[0]}"
        ax.plot(pd.to_numeric(group["frame"], errors="coerce"), pd.to_numeric(group[delta_col], errors="coerce"), label=label, linewidth=1.3)
    ax.axhline(0, linewidth=1)
    ax.set_title("Frame-level PA-MPJPE deltas for key reduced-matrix comparisons")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Delta PA-MPJPE (mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    default_repo_root = Path(__file__).resolve().parents[1]

    parser.add_argument("--repo-root", type=str, default=str(default_repo_root))
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--trial", "--sequence", dest="sequence", type=str, required=True, help="Trial/sequence name, e.g. SUB01_off_walk_1")
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--condition-map-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--save-frame-deltas", action="store_true", help="Save full pairwise per-frame delta table.")

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    summary_csv = find_summary_csv(repo_root, args.subject, args.sequence, as_path(args.summary_csv))
    subject_out_root = summary_csv.parent

    if args.out_dir is None:
        out_dir = subject_out_root / "reduced_matrix_analysis"
    else:
        out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    print("\nConfiguration")
    print("-------------")
    print(f"Repo root:   {repo_root}")
    print(f"Subject:     {args.subject}")
    print(f"Trial:       {args.sequence}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Output dir:  {out_dir}")

    summary = pd.read_csv(summary_csv)
    summary = to_numeric_columns(summary)
    summary = add_reduced_ids(summary)

    condition_map = load_condition_map(as_path(args.condition_map_csv), out_dir)
    summary = merge_condition_map(summary, condition_map)

    # Save the reduced key result table.
    key_results = make_key_results_table(summary)
    key_results.to_csv(out_dir / "reduced_matrix_key_results_table.csv", index=False)

    comparisons = pd.DataFrame(DEFAULT_COMPARISON_ROWS)
    comparisons.to_csv(out_dir / "reduced_matrix_comparison_plan.csv", index=False)

    pairwise = compute_pairwise_comparisons(summary, comparisons)
    pairwise.to_csv(out_dir / "reduced_matrix_pairwise_comparisons.csv", index=False)

    classification = compute_reliability_classification(summary, pairwise)
    classification.to_csv(out_dir / "reduced_matrix_reliability_classification.csv", index=False)

    # Per-frame analysis when per-frame CSVs can be found.
    per_frame_tables = load_per_frame_tables(summary, subject_out_root)
    if per_frame_tables:
        frame_summary, frame_deltas = compute_frame_pairwise_deltas(per_frame_tables, comparisons)
        frame_summary.to_csv(out_dir / "reduced_matrix_frame_level_pairwise_summary.csv", index=False)
        if args.save_frame_deltas and not frame_deltas.empty:
            frame_deltas.to_csv(out_dir / "reduced_matrix_frame_level_pairwise_deltas_selected.csv", index=False)
    else:
        frame_summary = pd.DataFrame()
        frame_deltas = pd.DataFrame()
        print("WARNING: No per-frame tables found. Skipping frame-level outputs.")

    # Plots.
    save_pairwise_bar(
        pairwise,
        "delta_pa_mpjpe_mean_mm",
        "Pairwise PA-MPJPE degradation, reduced matrix",
        "Delta PA-MPJPE (mm)",
        out_dir / "reduced_pairwise_delta_pa_mpjpe.png",
    )
    save_pairwise_bar(
        pairwise,
        "delta_foot_clearance_mean_abs_error_mm",
        "Pairwise foot-clearance degradation, reduced matrix",
        "Delta foot-clearance error (mm)",
        out_dir / "reduced_pairwise_delta_foot_clearance.png",
    )
    save_pairwise_bar(
        pairwise,
        "delta_stride_length_mean_abs_error_mm",
        "Pairwise stride-length degradation, reduced matrix",
        "Delta stride-length error (mm)",
        out_dir / "reduced_pairwise_delta_stride_length.png",
    )
    save_pairwise_bar(
        pairwise,
        "delta_step_contact_timing_mae_s",
        "Pairwise contact-timing degradation, reduced matrix",
        "Delta contact-timing MAE (s)",
        out_dir / "reduced_pairwise_delta_contact_timing.png",
    )
    save_summary_scatter(summary, "pa_mpjpe_mean_mm", "stride_length_mean_abs_error_mm", out_dir / "reduced_scatter_pa_mpjpe_vs_stride_length.png")
    save_frame_metric_plot(per_frame_tables, "pa_mpjpe_mm", out_dir / "reduced_frame_pa_mpjpe_selected.png")
    save_frame_delta_plot(frame_deltas, "pa_mpjpe_mm", out_dir / "reduced_frame_delta_pa_mpjpe_key_comparisons.png")

    print("\nDone.")
    print("-----")
    print(f"Key results:              {out_dir / 'reduced_matrix_key_results_table.csv'}")
    print(f"Comparison plan:          {out_dir / 'reduced_matrix_comparison_plan.csv'}")
    print(f"Pairwise comparisons:     {out_dir / 'reduced_matrix_pairwise_comparisons.csv'}")
    print(f"Reliability table:        {out_dir / 'reduced_matrix_reliability_classification.csv'}")
    print(f"Frame-level summary:      {out_dir / 'reduced_matrix_frame_level_pairwise_summary.csv'}")
    print(f"Plots saved in:           {out_dir}")

    print("\nPairwise comparison overview:")
    short_cols = [
        "comparison_id", "comparison_name", "test_id", "baseline_id", "status",
        "delta_pa_mpjpe_mean_mm",
        "delta_foot_clearance_mean_abs_error_mm",
        "delta_stride_length_mean_abs_error_mm",
        "delta_step_contact_timing_mae_s",
    ]
    short_cols = [c for c in short_cols if c in pairwise.columns]
    print(pairwise[short_cols].to_string(index=False))


if __name__ == "__main__":
    main()
