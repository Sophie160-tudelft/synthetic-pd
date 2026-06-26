r"""
Analyze additional degradation of WHAM experiment results relative to baselines.

Purpose
-------
This script reads the outputs created by:
    06_compare_bmclab_wham_all_experiments_auto_subject.py

It does NOT rerun SMPL, WHAM, or BMCLab reconstruction. Instead, it compares the
already-computed metric CSV/JSON outputs across experiments.

Main questions answered
-----------------------
1. How much additional error does each rendered condition introduce compared with
   the neutral baseline E0?
2. How much additional error does each condition introduce compared with the
   appropriate home baseline for its experimental block?
3. Which experiments degrade most on clinical gait features?
4. At which frames does a perturbed experiment become worse than the selected
   home baseline?

Recommended interpretation
--------------------------
- E0 is the neutral technical baseline.
- E1 is the home/living-room baseline with table.
- E2 is the bright living-room-without-table baseline for the camera-view block.
- E3 is the stable frontal/table setup and is used as R2 in the reduced matrix.
- Block-specific comparisons should be preferred for interpretation.

Example run
-----------
From project root:

    conda activate metrics
    cd C:\Users\sopha\synthetic-pd

    python scripts\07_analyze_experiment_degradation_relative_to_baselines.py ^
        --subject SUB01 ^
        --sequence SUB01_off_walk_1

Optional with condition map:

    python scripts\07_analyze_experiment_degradation_relative_to_baselines.py ^
        --subject SUB01 ^
        --sequence SUB01_off_walk_1 ^
        --condition-map-csv data\experiment_condition_map.csv

Condition map CSV columns, optional:
    experiment,environment,camera_angle,lighting,occlusion,perturbation_group,description

Outputs
-------
Saved to:
    results/clinically_relevant_metrics_v1/batch_wham_experiments/<SUBJECT>/<SEQUENCE>/degradation_analysis/

Core tables:
    degradation_vs_E0_summary.csv
    degradation_vs_E1_summary.csv
    block_specific_pairwise_comparisons.csv
    experiment_ranking_by_clinical_degradation_vs_E1.csv
    frame_level_reliability_summary_vs_E1.csv
    frame_level_delta_vs_E1_selected.csv
    thesis_key_results_table.csv
    condition_map_template.csv

Plots:
    bar_delta_foot_clearance_vs_E1.png
    bar_delta_stride_length_vs_E1.png
    bar_delta_cadence_vs_E1.png
    block_pairwise_delta_pa_mpjpe.png
    block_pairwise_delta_foot_clearance.png
    block_pairwise_delta_stride_length.png
    scatter_pa_mpjpe_vs_foot_clearance_error.png
    line_frame_delta_pa_mpjpe_vs_E1_selected.png
    line_frame_pa_mpjpe_selected.png

Notes
-----
Positive delta means worse than baseline for error metrics.
Negative delta means better than baseline for error metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend so plots save correctly from PowerShell/batch runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASELINE_NEUTRAL = "E0"
DEFAULT_BASELINE_HOME = "E1"
DEFAULT_BASELINE_CAMERA = "E2"


# Metrics where lower is better and where baseline subtraction is meaningful.
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

# Smaller thesis table, focused on the metrics you actually discuss.
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

# Metrics used for the summary degradation score. These are all error metrics, lower is better.
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

DEFAULT_CONDITION_ROWS = [
    {
        "experiment": "E0",
        "environment": "neutral",
        "camera_angle": "frontal",
        "lighting": "good",
        "occlusion": "none",
        "perturbation_group": "neutral_baseline",
        "description": "Neutral technical baseline",
    },
    {
        "experiment": "E1",
        "environment": "living_room",
        "camera_angle": "frontal",
        "lighting": "good",
        "occlusion": "none",
        "perturbation_group": "home_baseline",
        "description": "Ideal home/living-room baseline",
    },
]


# Block-specific comparisons used for thesis interpretation.
# Positive deltas mean the test condition has a higher error than the baseline.
DEFAULT_BLOCK_COMPARISON_ROWS = [
    {
        "comparison_id": "C1",
        "test_id": "E1",
        "baseline_id": "E0",
        "comparison_group": "Room complexity",
        "comparison_name": "Living-room baseline vs neutral baseline",
        "question_answered": "Does room complexity affect recovered motion?",
    },
    {
        "comparison_id": "C2",
        "test_id": "E3",
        "baseline_id": "E2",
        "comparison_group": "Camera view",
        "comparison_name": "Stable frontal/table setup vs frontal tripod baseline",
        "question_answered": "Does a lower frontal/table camera differ from the frontal tripod baseline?",
    },
    {
        "comparison_id": "C3",
        "test_id": "E4",
        "baseline_id": "E2",
        "comparison_group": "Camera view",
        "comparison_name": "Upper-corner viewpoint vs frontal tripod baseline",
        "question_answered": "Does the upper-corner viewpoint affect recovered motion?",
    },
    {
        "comparison_id": "C4",
        "test_id": "E5",
        "baseline_id": "E2",
        "comparison_group": "Camera view",
        "comparison_name": "Centred frontal setup vs frontal tripod baseline",
        "question_answered": "Does subject centring affect recovered motion?",
    },
    {
        "comparison_id": "C5",
        "test_id": "E6",
        "baseline_id": "E0",
        "comparison_group": "Lighting",
        "comparison_name": "Dim neutral condition vs neutral bright baseline",
        "question_answered": "Does dim lighting affect recovered motion in the neutral room?",
    },
    {
        "comparison_id": "C6",
        "test_id": "E7",
        "baseline_id": "E2",
        "comparison_group": "Lighting",
        "comparison_name": "Dim living-room condition vs bright living-room baseline",
        "question_answered": "Does dim lighting affect recovered motion in the living-room without table?",
    },
    {
        "comparison_id": "C7",
        "test_id": "E8",
        "baseline_id": "E1",
        "comparison_group": "Occlusion",
        "comparison_name": "Strong lower-body occlusion vs home baseline",
        "question_answered": "How much degradation is introduced by severe lower-body occlusion?",
    },
    {
        "comparison_id": "C8",
        "test_id": "E11",
        "baseline_id": "E1",
        "comparison_group": "Occlusion",
        "comparison_name": "Frontal partial one-leg occlusion vs home baseline",
        "question_answered": "Does partial frontal one-leg occlusion affect recovered motion?",
    },
    {
        "comparison_id": "C9",
        "test_id": "E10",
        "baseline_id": "E4",
        "comparison_group": "Occlusion",
        "comparison_name": "Upper-corner partial occlusion vs upper-corner baseline",
        "question_answered": "Does partial occlusion add degradation to the upper-corner viewpoint?",
    },
    {
        "comparison_id": "C10",
        "test_id": "E13",
        "baseline_id": "E4",
        "comparison_group": "Occlusion",
        "comparison_name": "Upper-corner one-leg occlusion vs upper-corner baseline",
        "question_answered": "Does one-leg occlusion add degradation to the upper-corner viewpoint?",
    },
]


# =============================================================================
# General helpers
# =============================================================================

def as_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiment_number(exp: str) -> int:
    exp = str(exp)
    if exp.upper().startswith("E"):
        try:
            return int(exp[1:])
        except ValueError:
            pass
    return 10_000


def sort_by_experiment(df: pd.DataFrame) -> pd.DataFrame:
    if "experiment" not in df.columns:
        return df
    df = df.copy()
    if "experiment_number" not in df.columns:
        df["experiment_number"] = df["experiment"].apply(experiment_number)
    return df.sort_values(["experiment_number", "experiment"]).reset_index(drop=True)


def to_numeric_columns(df: pd.DataFrame, exclude: Sequence[str] = ("experiment", "subject", "sequence")) -> pd.DataFrame:
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
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates = sorted(candidates, key=lambda p: len(str(p)))
        return candidates[0]

    raise FileNotFoundError(
        "Could not find the combined experiment summary CSV. Expected for example:\n"
        f"{default}\n\n"
        "Run script 06 batch first, or pass --summary-csv explicitly."
    )


def load_condition_map(condition_map_csv: Optional[Path], output_dir: Path) -> pd.DataFrame:
    template_path = output_dir / "condition_map_template.csv"
    template = pd.DataFrame(DEFAULT_CONDITION_ROWS)
    template.to_csv(template_path, index=False)

    if condition_map_csv is None:
        return template

    if not condition_map_csv.exists():
        raise FileNotFoundError(f"Condition map CSV not found:\n{condition_map_csv}")

    df = pd.read_csv(condition_map_csv)
    required = ["experiment"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Condition map is missing columns: {missing}")

    for col in ["environment", "camera_angle", "lighting", "occlusion", "perturbation_group", "description"]:
        if col not in df.columns:
            df[col] = "unknown"

    return df


def merge_condition_map(summary: pd.DataFrame, condition_map: pd.DataFrame) -> pd.DataFrame:
    out = summary.merge(condition_map, on="experiment", how="left", suffixes=("", "_condition"))
    for col in ["environment", "camera_angle", "lighting", "occlusion", "perturbation_group", "description"]:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown")
    return out


# =============================================================================
# Metric degradation summaries
# =============================================================================

def compute_degradation(summary: pd.DataFrame, baseline_exp: str, metrics: Sequence[str]) -> pd.DataFrame:
    if baseline_exp not in set(summary["experiment"].astype(str)):
        raise ValueError(f"Baseline {baseline_exp} not found in summary CSV.")

    df = summary.copy()
    baseline_row = df.loc[df["experiment"].astype(str) == baseline_exp].iloc[0]

    baseline_frames = float(baseline_row.get("frames_used", np.nan))
    if math.isnan(baseline_frames) or baseline_frames <= 0:
        baseline_frames = np.nan

    for metric in metrics:
        if metric not in df.columns:
            continue
        base_value = pd.to_numeric(pd.Series([baseline_row[metric]]), errors="coerce").iloc[0]
        df[f"baseline_{baseline_exp}_{metric}"] = base_value
        df[f"delta_vs_{baseline_exp}_{metric}"] = pd.to_numeric(df[metric], errors="coerce") - base_value

    if "frames_used" in df.columns:
        df["frames_ratio_vs_baseline"] = pd.to_numeric(df["frames_used"], errors="coerce") / baseline_frames
        df["comparable_frame_count"] = df["frames_ratio_vs_baseline"].between(0.95, 1.05)
    else:
        df["frames_ratio_vs_baseline"] = np.nan
        df["comparable_frame_count"] = True

    return sort_by_experiment(df)


def compute_clinical_degradation_score(degradation_df: pd.DataFrame, baseline_exp: str) -> pd.DataFrame:
    """
    Computes a simple 0-1 score from positive deltas versus baseline.

    For each metric:
        positive_delta = max(delta, 0)
        normalized_delta = positive_delta / max_positive_delta_for_that_metric

    Final score = mean normalized positive delta over available score metrics.

    Higher score = more additional degradation relative to baseline.
    This is only a ranking aid, not a clinical score.
    """
    df = degradation_df.copy()
    norm_cols = []

    for metric in CLINICAL_SCORE_METRICS:
        delta_col = f"delta_vs_{baseline_exp}_{metric}"
        if delta_col not in df.columns:
            continue

        values = pd.to_numeric(df[delta_col], errors="coerce")
        positive = values.clip(lower=0)
        max_positive = positive.max(skipna=True)

        norm_col = f"norm_positive_{delta_col}"
        if pd.isna(max_positive) or max_positive <= 0:
            df[norm_col] = 0.0
        else:
            df[norm_col] = positive / max_positive

        norm_cols.append(norm_col)

    if norm_cols:
        df["clinical_degradation_score_vs_baseline"] = df[norm_cols].mean(axis=1, skipna=True)
    else:
        df["clinical_degradation_score_vs_baseline"] = np.nan

    # Baseline itself should be exactly zero.
    df.loc[df["experiment"].astype(str) == baseline_exp, "clinical_degradation_score_vs_baseline"] = 0.0

    return df.sort_values("clinical_degradation_score_vs_baseline", ascending=False).reset_index(drop=True)


def metric_value(row: pd.Series, metric: str) -> float:
    if metric not in row.index:
        return np.nan
    return pd.to_numeric(pd.Series([row[metric]]), errors="coerce").iloc[0]


def compute_pairwise_comparisons(summary: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_id = {str(row["experiment"]): row for _, row in summary.iterrows()}

    for _, comp in comparisons.iterrows():
        test_id = str(comp["test_id"])
        baseline_id = str(comp["baseline_id"])

        base_row = {
            "comparison_id": comp.get("comparison_id", ""),
            "comparison_group": comp.get("comparison_group", ""),
            "comparison_name": comp.get("comparison_name", ""),
            "question_answered": comp.get("question_answered", ""),
            "test_id": test_id,
            "baseline_id": baseline_id,
        }

        if test_id not in by_id or baseline_id not in by_id:
            base_row["status"] = "missing_test_or_baseline"
            rows.append(base_row)
            continue

        test = by_id[test_id]
        baseline = by_id[baseline_id]
        base_row["status"] = "ok"
        base_row["test_description"] = test.get("description", "")
        base_row["baseline_description"] = baseline.get("description", "")

        test_frames = metric_value(test, "frames_used")
        baseline_frames = metric_value(baseline, "frames_used")
        base_row["test_frames_used"] = test_frames
        base_row["baseline_frames_used"] = baseline_frames
        if not pd.isna(test_frames) and not pd.isna(baseline_frames) and baseline_frames > 0:
            base_row["frames_ratio_test_vs_baseline"] = test_frames / baseline_frames
            base_row["comparable_frame_count"] = 0.95 <= base_row["frames_ratio_test_vs_baseline"] <= 1.05
        else:
            base_row["frames_ratio_test_vs_baseline"] = np.nan
            base_row["comparable_frame_count"] = True

        for metric in KEY_ERROR_METRICS:
            if metric not in summary.columns:
                continue
            test_value = metric_value(test, metric)
            baseline_value = metric_value(baseline, metric)
            base_row[f"test_{metric}"] = test_value
            base_row[f"baseline_{metric}"] = baseline_value
            base_row[f"delta_{metric}"] = test_value - baseline_value

        rows.append(base_row)

    return pd.DataFrame(rows)


def make_thesis_table(summary: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "experiment",
        "experiment_number",
        "environment",
        "camera_angle",
        "lighting",
        "occlusion",
        "perturbation_group",
        "frames_used",
        "gt_num_gait_events",
        "wham_num_gait_events",
        "gait_event_count_difference",
    ]

    for metric in THESIS_METRICS:
        if metric in summary.columns:
            keep.append(metric)

    keep = [c for c in keep if c in summary.columns]
    return sort_by_experiment(summary[keep].copy())


def write_metric_notes(path: Path) -> None:
    notes = [
        ["term", "meaning"],
        ["delta_vs_E0", "Experiment error minus neutral baseline E0 error. Positive means worse than E0."],
        ["delta_vs_E1", "Experiment error minus ideal home baseline E1 error. Positive means additional degradation due to perturbation."],
        ["clinical_degradation_score_vs_baseline", "Average normalized positive degradation over selected pose and gait-feature error metrics. Use only as a ranking aid."],
        ["comparable_frame_count", "False means frame count differs more than 5% from the baseline; interpret timing and event metrics cautiously."],
        ["PA-MPJPE", "Local 3D pose error after frame-wise similarity alignment. Useful but insufficient alone for clinical gait reliability."],
        ["root-aligned MPJPE", "Diagnostic metric. Large values may reflect global orientation or coordinate-frame mismatch."],
        ["foot clearance / stride / contact timing", "Clinically important lower-limb and gait-cycle metrics. These are often more sensitive to setup degradation than PA-MPJPE."],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(notes)


# =============================================================================
# Per-frame analysis
# =============================================================================

def existing_path(value) -> Optional[Path]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    p = Path(str(value))
    if p.exists():
        return p
    return None


def find_per_frame_csv(row: pd.Series, subject_out_root: Path) -> Optional[Path]:
    # Prefer explicit path from the combined summary.
    for key in ["per_frame_csv", "per_frame_path"]:
        if key in row:
            p = existing_path(row[key])
            if p is not None:
                return p

    exp = str(row["experiment"])
    exp_dir = subject_out_root / exp
    if exp_dir.exists():
        candidates = sorted(exp_dir.glob("*per_frame_metrics*.csv"))
        if candidates:
            return candidates[0]

    candidates = sorted(subject_out_root.rglob(f"*{exp}*per_frame_metrics*.csv"))
    return candidates[0] if candidates else None


def load_per_frame_tables(summary: pd.DataFrame, subject_out_root: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for _, row in summary.iterrows():
        exp = str(row["experiment"])
        p = find_per_frame_csv(row, subject_out_root)
        if p is None:
            print(f"WARNING: per-frame CSV not found for {exp}")
            continue
        df = pd.read_csv(p)
        if "frame" not in df.columns:
            print(f"WARNING: per-frame CSV for {exp} has no 'frame' column: {p}")
            continue
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
        tables[exp] = df
    return tables


def consecutive_ranges(frames: Sequence[int]) -> List[Tuple[int, int]]:
    frames = sorted(int(f) for f in frames)
    if not frames:
        return []

    ranges = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            ranges.append((start, prev))
            start = prev = f
    ranges.append((start, prev))
    return ranges


def top_ranges_from_delta(df: pd.DataFrame, delta_col: str, percentile: float = 95.0, max_ranges: int = 5) -> str:
    if delta_col not in df.columns:
        return ""
    values = pd.to_numeric(df[delta_col], errors="coerce")
    valid = df.loc[values.notna()].copy()
    if valid.empty:
        return ""

    values = pd.to_numeric(valid[delta_col], errors="coerce")
    threshold = np.nanpercentile(values, percentile)
    peak = valid.loc[values >= threshold].copy()
    if peak.empty:
        return ""

    ranges = consecutive_ranges(peak["frame"].astype(int).tolist())

    # Score each range by mean delta and keep the most severe ones.
    scored = []
    for start, end in ranges:
        segment = valid[(valid["frame"] >= start) & (valid["frame"] <= end)]
        scored.append((float(segment[delta_col].mean()), start, end, int(len(segment))))
    scored = sorted(scored, reverse=True)[:max_ranges]

    return "; ".join(
        f"{start}-{end} (n={n}, mean={mean_val:.2f})"
        for mean_val, start, end, n in scored
    )


def compute_frame_delta_summary(
    per_frame_tables: Dict[str, pd.DataFrame],
    baseline_exp: str,
    selected_experiments: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if baseline_exp not in per_frame_tables:
        raise ValueError(f"No per-frame table found for baseline {baseline_exp}")

    baseline = per_frame_tables[baseline_exp].copy()
    baseline_cols = ["frame"] + [c for c in PER_FRAME_METRICS if c in baseline.columns]
    baseline = baseline[baseline_cols].copy()

    if selected_experiments is None:
        experiments = sorted(per_frame_tables.keys(), key=experiment_number)
    else:
        experiments = [e for e in selected_experiments if e in per_frame_tables]

    summary_rows = []
    selected_delta_frames = []

    for exp in experiments:
        if exp == baseline_exp:
            continue

        df = per_frame_tables[exp].copy()
        available_cols = ["frame"] + [c for c in PER_FRAME_METRICS if c in df.columns and c in baseline.columns]
        df = df[available_cols].copy()

        merged = df.merge(baseline, on="frame", suffixes=("", f"_{baseline_exp}"), how="inner")
        if merged.empty:
            continue

        for metric in [c for c in PER_FRAME_METRICS if c in df.columns and c in baseline.columns]:
            merged[f"delta_vs_{baseline_exp}_{metric}"] = (
                pd.to_numeric(merged[metric], errors="coerce")
                - pd.to_numeric(merged[f"{metric}_{baseline_exp}"], errors="coerce")
            )

        merged["experiment"] = exp
        selected_delta_frames.append(merged)

        row = {
            "experiment": exp,
            "baseline": baseline_exp,
            "frames_compared": int(len(merged)),
        }

        for metric in [c for c in PER_FRAME_METRICS if c in df.columns and c in baseline.columns]:
            delta_col = f"delta_vs_{baseline_exp}_{metric}"
            values = pd.to_numeric(merged[delta_col], errors="coerce")
            abs_values = values.abs()
            row[f"mean_delta_{metric}"] = float(values.mean())
            row[f"median_delta_{metric}"] = float(values.median())
            row[f"max_delta_{metric}"] = float(values.max())
            row[f"min_delta_{metric}"] = float(values.min())
            row[f"mean_abs_delta_{metric}"] = float(abs_values.mean())
            row[f"frame_of_max_delta_{metric}"] = int(merged.loc[values.idxmax(), "frame"]) if values.notna().any() else None
            row[f"top_95pct_ranges_{metric}"] = top_ranges_from_delta(merged, delta_col, percentile=95.0)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["experiment_number"] = summary_df["experiment"].apply(experiment_number)
        summary_df = sort_by_experiment(summary_df)

    if selected_delta_frames:
        deltas_df = pd.concat(selected_delta_frames, ignore_index=True)
        deltas_df["experiment_number"] = deltas_df["experiment"].apply(experiment_number)
        deltas_df = sort_by_experiment(deltas_df)
    else:
        deltas_df = pd.DataFrame()

    return summary_df, deltas_df


# =============================================================================
# Plots
# =============================================================================

def save_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    comparable_only: bool = False,
) -> None:
    plot_df = df.copy()
    if comparable_only and "comparable_frame_count" in plot_df.columns:
        plot_df = plot_df[plot_df["comparable_frame_count"]]
    plot_df = sort_by_experiment(plot_df)

    if y_col not in plot_df.columns or plot_df.empty:
        print(f"WARNING: Cannot plot {y_col}; column missing or no rows.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df[x_col].astype(str), pd.to_numeric(plot_df[y_col], errors="coerce"))
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_pairwise_bar_plot(
    pairwise: pd.DataFrame,
    delta_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if pairwise.empty or delta_col not in pairwise.columns:
        print(f"WARNING: Cannot plot {delta_col}; column missing or no rows.")
        return
    plot_df = pairwise[pairwise["status"] == "ok"].copy()
    if plot_df.empty:
        return
    labels = plot_df["comparison_id"].astype(str) + ": " + plot_df["test_id"].astype(str) + "-" + plot_df["baseline_id"].astype(str)
    values = pd.to_numeric(plot_df[delta_col], errors="coerce")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values)
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Block-specific comparison")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        print(f"WARNING: Cannot plot scatter {x_col} vs {y_col}; column missing.")
        return

    plot_df = df.copy()
    x = pd.to_numeric(plot_df[x_col], errors="coerce")
    y = pd.to_numeric(plot_df[y_col], errors="coerce")
    valid = x.notna() & y.notna()
    plot_df = plot_df[valid]
    x = x[valid]
    y = y[valid]

    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y)
    for _, row in plot_df.iterrows():
        ax.annotate(str(row["experiment"]), (row[x_col], row[y_col]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_line_plot_frame_metric(
    per_frame_tables: Dict[str, pd.DataFrame],
    experiments: Sequence[str],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    plotted = False
    for exp in experiments:
        if exp not in per_frame_tables:
            continue
        df = per_frame_tables[exp]
        if "frame" not in df.columns or metric not in df.columns:
            continue
        ax.plot(pd.to_numeric(df["frame"], errors="coerce"), pd.to_numeric(df[metric], errors="coerce"), label=exp, linewidth=1.4)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_line_plot_frame_delta(
    frame_delta_df: pd.DataFrame,
    delta_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if frame_delta_df.empty or delta_col not in frame_delta_df.columns:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for exp, group in frame_delta_df.groupby("experiment"):
        ax.plot(pd.to_numeric(group["frame"], errors="coerce"), pd.to_numeric(group[delta_col], errors="coerce"), label=exp, linewidth=1.4)

    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
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
    parser.add_argument("--subject", type=str, default="SUB01")
    parser.add_argument("--sequence", type=str, default="SUB01_off_walk_1")
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--condition-map-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument("--neutral-baseline", type=str, default=DEFAULT_BASELINE_NEUTRAL)
    parser.add_argument("--home-baseline", type=str, default=DEFAULT_BASELINE_HOME)

    parser.add_argument(
        "--selected-frame-experiments",
        type=str,
        default="E0,E1,E3,E8,E10,E11",
        help="Comma-separated experiment IDs for frame-level plots and selected frame-delta CSV.",
    )
    parser.add_argument(
        "--save-full-frame-deltas",
        action="store_true",
        help="If set, save frame-level deltas for all experiments instead of only selected experiments.",
    )

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    summary_csv = find_summary_csv(repo_root, args.subject, args.sequence, as_path(args.summary_csv))

    subject_out_root = summary_csv.parent
    if args.out_dir is None:
        out_dir = subject_out_root / "degradation_analysis"
    else:
        out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    print("\nConfiguration")
    print("-------------")
    print(f"Repo root:        {repo_root}")
    print(f"Subject:          {args.subject}")
    print(f"Sequence:         {args.sequence}")
    print(f"Summary CSV:      {summary_csv}")
    print(f"Output dir:       {out_dir}")
    print(f"Neutral baseline: {args.neutral_baseline}")
    print(f"Home baseline:    {args.home_baseline}")

    summary = pd.read_csv(summary_csv)
    summary = to_numeric_columns(summary)
    summary["experiment"] = summary["experiment"].astype(str)
    if "experiment_number" not in summary.columns:
        summary["experiment_number"] = summary["experiment"].apply(experiment_number)
    summary = sort_by_experiment(summary)

    condition_map = load_condition_map(as_path(args.condition_map_csv), out_dir)
    summary = merge_condition_map(summary, condition_map)

    # Warn about non-comparable frame counts.
    if "frames_used" in summary.columns:
        frame_counts = summary[["experiment", "frames_used"]].copy()
        mode_frames = frame_counts["frames_used"].mode(dropna=True)
        if len(mode_frames):
            typical = float(mode_frames.iloc[0])
            unusual = frame_counts[pd.to_numeric(frame_counts["frames_used"], errors="coerce") < 0.95 * typical]
            if not unusual.empty:
                print("\nWARNING: Some experiments have much shorter frame counts and should be marked cautiously:")
                print(unusual.to_string(index=False))

    thesis_table = make_thesis_table(summary)
    thesis_table.to_csv(out_dir / "thesis_key_results_table.csv", index=False)

    degradation_e0 = compute_degradation(summary, args.neutral_baseline, KEY_ERROR_METRICS)
    degradation_e1 = compute_degradation(summary, args.home_baseline, KEY_ERROR_METRICS)

    degradation_e0.to_csv(out_dir / f"degradation_vs_{args.neutral_baseline}_summary.csv", index=False)
    degradation_e1.to_csv(out_dir / f"degradation_vs_{args.home_baseline}_summary.csv", index=False)

    block_comparisons = pd.DataFrame(DEFAULT_BLOCK_COMPARISON_ROWS)
    block_comparisons.to_csv(out_dir / "block_specific_comparison_plan.csv", index=False)
    block_pairwise = compute_pairwise_comparisons(summary, block_comparisons)
    block_pairwise.to_csv(out_dir / "block_specific_pairwise_comparisons.csv", index=False)

    ranking_e1 = compute_clinical_degradation_score(degradation_e1, args.home_baseline)
    ranking_e1.to_csv(out_dir / f"experiment_ranking_by_clinical_degradation_vs_{args.home_baseline}.csv", index=False)

    write_metric_notes(out_dir / "metric_interpretation_notes.csv")

    # Per-frame deltas vs home baseline.
    per_frame_tables = load_per_frame_tables(summary, subject_out_root)
    selected = [e.strip() for e in args.selected_frame_experiments.split(",") if e.strip()]
    if args.save_full_frame_deltas:
        selected_for_deltas = sorted(per_frame_tables.keys(), key=experiment_number)
        delta_name = f"frame_level_delta_vs_{args.home_baseline}_all_experiments.csv"
    else:
        selected_for_deltas = selected
        delta_name = f"frame_level_delta_vs_{args.home_baseline}_selected.csv"

    try:
        frame_summary, frame_deltas = compute_frame_delta_summary(
            per_frame_tables=per_frame_tables,
            baseline_exp=args.home_baseline,
            selected_experiments=selected_for_deltas,
        )
        frame_summary.to_csv(out_dir / f"frame_level_reliability_summary_vs_{args.home_baseline}.csv", index=False)
        if not frame_deltas.empty:
            frame_deltas.to_csv(out_dir / delta_name, index=False)
    except Exception as error:
        print(f"WARNING: Could not compute frame-level deltas: {error}")
        frame_summary = pd.DataFrame()
        frame_deltas = pd.DataFrame()

    # Plots from metric summary.
    save_bar_plot(
        degradation_e1,
        x_col="experiment",
        y_col=f"delta_vs_{args.home_baseline}_foot_clearance_mean_abs_error_mm",
        title=f"Additional foot-clearance error versus {args.home_baseline}",
        ylabel="Delta foot-clearance error (mm)",
        output_path=out_dir / f"bar_delta_foot_clearance_vs_{args.home_baseline}.png",
    )
    save_bar_plot(
        degradation_e1,
        x_col="experiment",
        y_col=f"delta_vs_{args.home_baseline}_stride_length_mean_abs_error_mm",
        title=f"Additional stride-length error versus {args.home_baseline}",
        ylabel="Delta stride-length error (mm)",
        output_path=out_dir / f"bar_delta_stride_length_vs_{args.home_baseline}.png",
    )
    save_bar_plot(
        degradation_e1,
        x_col="experiment",
        y_col=f"delta_vs_{args.home_baseline}_cadence_abs_error_steps_per_min",
        title=f"Additional cadence error versus {args.home_baseline}",
        ylabel="Delta cadence error (steps/min)",
        output_path=out_dir / f"bar_delta_cadence_vs_{args.home_baseline}.png",
    )
    save_pairwise_bar_plot(
        block_pairwise,
        delta_col="delta_pa_mpjpe_mean_mm",
        title="Block-specific PA-MPJPE degradation",
        ylabel="Delta PA-MPJPE (mm)",
        output_path=out_dir / "block_pairwise_delta_pa_mpjpe.png",
    )
    save_pairwise_bar_plot(
        block_pairwise,
        delta_col="delta_foot_clearance_mean_abs_error_mm",
        title="Block-specific foot-clearance degradation",
        ylabel="Delta foot-clearance error (mm)",
        output_path=out_dir / "block_pairwise_delta_foot_clearance.png",
    )
    save_pairwise_bar_plot(
        block_pairwise,
        delta_col="delta_stride_length_mean_abs_error_mm",
        title="Block-specific stride-length degradation",
        ylabel="Delta stride-length error (mm)",
        output_path=out_dir / "block_pairwise_delta_stride_length.png",
    )

    save_scatter_plot(
        summary,
        x_col="pa_mpjpe_mean_mm",
        y_col="foot_clearance_mean_abs_error_mm",
        title="PA-MPJPE versus foot-clearance error",
        xlabel="PA-MPJPE (mm)",
        ylabel="Foot-clearance error (mm)",
        output_path=out_dir / "scatter_pa_mpjpe_vs_foot_clearance_error.png",
    )
    save_pairwise_bar_plot(
        block_pairwise,
        delta_col="delta_pa_mpjpe_mean_mm",
        title="Block-specific PA-MPJPE degradation",
        ylabel="Delta PA-MPJPE (mm)",
        output_path=out_dir / "block_pairwise_delta_pa_mpjpe.png",
    )
    save_pairwise_bar_plot(
        block_pairwise,
        delta_col="delta_foot_clearance_mean_abs_error_mm",
        title="Block-specific foot-clearance degradation",
        ylabel="Delta foot-clearance error (mm)",
        output_path=out_dir / "block_pairwise_delta_foot_clearance.png",
    )
    save_pairwise_bar_plot(
        block_pairwise,
        delta_col="delta_stride_length_mean_abs_error_mm",
        title="Block-specific stride-length degradation",
        ylabel="Delta stride-length error (mm)",
        output_path=out_dir / "block_pairwise_delta_stride_length.png",
    )

    save_scatter_plot(
        summary,
        x_col="pa_mpjpe_mean_mm",
        y_col="stride_length_mean_abs_error_mm",
        title="PA-MPJPE versus stride-length error",
        xlabel="PA-MPJPE (mm)",
        ylabel="Stride-length error (mm)",
        output_path=out_dir / "scatter_pa_mpjpe_vs_stride_length_error.png",
    )

    # Frame-level plots.
    selected_for_plot = [e for e in selected if e in per_frame_tables]
    save_line_plot_frame_metric(
        per_frame_tables,
        experiments=selected_for_plot,
        metric="pa_mpjpe_mm",
        title="Frame-level PA-MPJPE for selected experiments",
        ylabel="PA-MPJPE (mm)",
        output_path=out_dir / "line_frame_pa_mpjpe_selected.png",
    )
    save_line_plot_frame_metric(
        per_frame_tables,
        experiments=selected_for_plot,
        metric="pelvis_trajectory_error_mm",
        title="Frame-level pelvis trajectory error for selected experiments",
        ylabel="Pelvis trajectory error (mm)",
        output_path=out_dir / "line_frame_pelvis_error_selected.png",
    )
    save_line_plot_frame_delta(
        frame_deltas,
        delta_col=f"delta_vs_{args.home_baseline}_pa_mpjpe_mm",
        title=f"Frame-level additional PA-MPJPE versus {args.home_baseline}",
        ylabel="Delta PA-MPJPE (mm)",
        output_path=out_dir / f"line_frame_delta_pa_mpjpe_vs_{args.home_baseline}_selected.png",
    )

    print("\nDone.")
    print("-----")
    print(f"Thesis key table:                  {out_dir / 'thesis_key_results_table.csv'}")
    print(f"Degradation vs {args.neutral_baseline}:              {out_dir / ('degradation_vs_' + args.neutral_baseline + '_summary.csv')}")
    print(f"Degradation vs {args.home_baseline}:              {out_dir / ('degradation_vs_' + args.home_baseline + '_summary.csv')}")
    print(f"Ranking vs {args.home_baseline}:                  {out_dir / ('experiment_ranking_by_clinical_degradation_vs_' + args.home_baseline + '.csv')}")
    print(f"Block-specific pairwise comparisons: {out_dir / 'block_specific_pairwise_comparisons.csv'}")
    print(f"Frame-level summary vs {args.home_baseline}:      {out_dir / ('frame_level_reliability_summary_vs_' + args.home_baseline + '.csv')}")
    print(f"Condition map template:            {out_dir / 'condition_map_template.csv'}")
    print(f"Plots saved in:                    {out_dir}")

    print("\nMost degraded experiments versus home baseline, by composite score:")
    cols = ["experiment", "clinical_degradation_score_vs_baseline", "frames_used", "comparable_frame_count"]
    cols = [c for c in cols if c in ranking_e1.columns]
    print(ranking_e1[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
