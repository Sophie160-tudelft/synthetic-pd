"""
10_make_results_tables_figures_complete.py

Reproduce all tables and figures used in Chapter 4 (Results) of the thesis.

This script reads the metric outputs produced by the earlier validation and WHAM
comparison scripts, then creates thesis-ready CSV tables, LaTeX tables, and PNG/JPG
figures.

Expected input files by default:
    clinically_relevant_metrics_v1.zip
    render_input_validation.zip
    selected_first_four_bmclab_sequences.csv

The script can also work with already-extracted folders. It creates the exact figure
filenames referenced in 5-Results.tex:
    figures/fig_1_pipeline_validation_pa_mpjpe.png
    figures/fig_2_sub01_pa_mpjpe_by_experiment.png
    figures/fig_3_sub01_spatial_gait_errors.png
    figures/fig_4_sub01_pa_vs_stride_scatter.png
    figures/fig_5_sub01_frame_level_pa_mpjpe_selected.png
    figures/fig_frame_delta_pa_mpjpe_vs_home_baseline.png
    figures/fig_peak_frame_visual_examples.jpg
    figures/fig_6_aligned_pa_mpjpe_by_condition_subject.png
    figures/fig_7_aligned_foot_clearance_by_condition_subject.png
    figures/fig_8_aligned_stride_length_by_condition_subject.png

The visual snapshot panel cannot be reconstructed from metric CSV files alone. To
reproduce it from source renders, pass --render-frames-root pointing to a folder that
contains PNG/JPG renders in one of these layouts:
    <root>/<SUBJECT>/<EXPERIMENT>/*.png
    <root>/<SUBJECT>/<TRIAL>/<EXPERIMENT>/*.png
    <root>/<TRIAL>/<EXPERIMENT>/*.png
    <root>/<EXPERIMENT>/*.png
If no render frames are supplied, the script creates a clearly marked placeholder
image and writes a note in README_reproducibility.txt.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# Thesis condition definitions
# -----------------------------

FULL_CONDITION_MAP: Dict[str, Tuple[str, str, str, str, str, str]] = {
    "E0": ("Room complexity", "Neutral", "Bright/day", "Frontal/table", "None", "Neutral technical baseline"),
    "E1": ("Room complexity", "Living room with table", "Bright/day", "Frontal/table", "None", "Living-room baseline"),
    "E2": ("Camera view", "Living room without table", "Bright/day", "Frontal tripod", "None", "High frontal static camera"),
    "E3": ("Camera view", "Living room without table", "Bright/day", "Frontal/table 0.95 m", "None", "Stable frontal/table setup"),
    "E4": ("Camera view", "Living room without table", "Bright/day", "Upper corner", "None", "Upper-corner viewpoint"),
    "E5": ("Camera view", "Living room without table", "Bright/day", "Frontal centered", "None", "Centred frontal setup"),
    "E6": ("Lighting", "Neutral", "Dim/night", "Frontal best", "None", "Dim neutral condition"),
    "E7": ("Lighting", "Living room without table", "Dim/night", "Frontal best", "None", "Dim living-room condition"),
    "E8": ("Occlusion", "Living room with table", "Bright/day", "Frontal/table", "Both-leg occlusion, all frames", "Strong lower-body occlusion"),
    "E9": ("Occlusion", "Living room with table", "Bright/day", "Upper-frontal", "No occlusion or partial both-leg occlusion", "Tracking-instability case"),
    "E10": ("Occlusion", "Living room with table", "Bright/day", "Upper corner", "Partial-frame occlusion", "Upper-corner partial occlusion"),
    "E11": ("Occlusion", "Living room with table", "Bright/day", "Frontal/table", "One-leg occlusion, partial frames", "Frontal partial one-leg occlusion"),
    "E12": ("Occlusion", "Living room with table", "Bright/day", "Frontal tripod", "One-leg occlusion, partial frames", "Stable partial-occlusion setup"),
    "E13": ("Occlusion", "Living room with table", "Bright/day", "Upper corner", "One-leg occlusion, partial frames", "Upper-corner one-leg occlusion"),
}

# The reduced-condition labels used in the thesis interpretation.
# "Full-matrix ID" is the condition label used for the matched setup in SUB01.
REDUCED_CONDITIONS: Dict[str, Dict[str, str]] = {
    "R0": {"source_id": "E0", "condition": "Neutral baseline", "environment": "Neutral", "camera": "Frontal/table", "occlusion": "None"},
    "R1": {"source_id": "E1", "condition": "Home baseline", "environment": "LR + table", "camera": "Frontal/table", "occlusion": "None"},
    "R2": {"source_id": "E3", "condition": "Stable frontal setup", "environment": "LR no table", "camera": "Frontal/table 0.95 m", "occlusion": "None"},
    "R3": {"source_id": "E4", "condition": "Upper-corner viewpoint", "environment": "LR no table", "camera": "Upper corner", "occlusion": "None"},
    "R4": {"source_id": "E8", "condition": "Strong occlusion", "environment": "LR + table", "camera": "Frontal/table", "occlusion": "Both legs"},
    "R5": {"source_id": "E11", "condition": "Partial frontal occlusion", "environment": "LR + table", "camera": "Frontal/table", "occlusion": "Partial one-leg occlusion"},
}

# Output folders for the extra subjects store the reduced conditions as E0--E5.
# The thesis labels them as matched full-matrix source conditions E0/E1/E3/E4/E8/E11.
PRIMARY_OUTPUT_MAP = {"R0": "E0", "R1": "E1", "R2": "E3", "R3": "E4", "R4": "E8", "R5": "E11"}
REPLICATION_OUTPUT_MAP = {"R0": "E0", "R1": "E1", "R2": "E2", "R3": "E3", "R4": "E4", "R5": "E5"}

SUMMARY_METRICS = [
    "pa_mpjpe_mean_mm",
    "step_contact_timing_mae_s",
    "cadence_abs_error_steps_per_min",
    "foot_clearance_mean_abs_error_mm",
    "step_length_mean_abs_error_mm",
    "stride_length_mean_abs_error_mm",
    "knee_rom_mean_abs_error_deg",
]

PRETTY_METRIC = {
    "pa_mpjpe_mean_mm": "PA-MPJPE",
    "step_contact_timing_mae_s": "Contact timing MAE",
    "cadence_abs_error_steps_per_min": "Cadence error",
    "foot_clearance_mean_abs_error_mm": "Foot clearance error",
    "step_length_mean_abs_error_mm": "Step length error",
    "stride_length_mean_abs_error_mm": "Stride length error",
    "knee_rom_mean_abs_error_deg": "Knee ROM error",
}

EXPECTED_FIGURES = [
    "fig_1_pipeline_validation_pa_mpjpe.png",
    "fig_2_sub01_pa_mpjpe_by_experiment.png",
    "fig_3_sub01_spatial_gait_errors.png",
    "fig_4_sub01_pa_vs_stride_scatter.png",
    "fig_5_sub01_frame_level_pa_mpjpe_selected.png",
    "fig_frame_delta_pa_mpjpe_vs_home_baseline.png",
    "fig_peak_frame_visual_examples.jpg",
    "fig_6_aligned_pa_mpjpe_by_condition_subject.png",
    "fig_7_aligned_foot_clearance_by_condition_subject.png",
    "fig_8_aligned_stride_length_by_condition_subject.png",
]

# -----------------------------
# Generic helpers
# -----------------------------

def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def unpack_or_use(path: Path, dest: Path) -> Path:
    """Return an extracted directory for a ZIP file, or the directory itself."""
    if path.is_dir():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip file or directory, got: {path}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(dest)
    return dest


def find_single(root: Path, pattern: str, required: bool = True) -> Optional[Path]:
    matches = sorted(root.rglob(pattern))
    if not matches:
        if required:
            raise FileNotFoundError(f"Could not find {pattern!r} under {root}")
        return None
    return matches[0]


def latex_escape(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value)
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in s)


def fmt(value, nd: int = 2) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        x = float(value)
        if abs(x - round(x)) < 1e-10:
            return str(int(round(x)))
        return f"{x:.{nd}f}"
    except Exception:
        return latex_escape(value)


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, nd: int = 2, resize: bool = True) -> None:
    align = "l" * len(df.columns)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
    ]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(c) for c in df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(fmt(row[c], nd=nd) for c in df.columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    if resize:
        lines.append("}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csv_and_table(df: pd.DataFrame, csv_path: Path, tex_path: Path, caption: str, label: str) -> None:
    df.to_csv(csv_path, index=False)
    write_latex_table(df, tex_path, caption, label)


def sort_experiment_ids(ids: Iterable[str]) -> List[str]:
    def key(e: str):
        m = re.match(r"E(\d+)$", str(e))
        return int(m.group(1)) if m else 9999
    return sorted(ids, key=key)


def set_common_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

# -----------------------------
# Data loading
# -----------------------------

def load_sequences(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject", "sequence"}
    if not required.issubset(df.columns):
        raise ValueError(f"Sequence CSV must contain columns {sorted(required)}; found {list(df.columns)}")
    return df


def metric_root_from_input(metrics_input: Path, work_dir: Path) -> Path:
    root = unpack_or_use(metrics_input, work_dir / "metrics")
    # Accept either root/clinically_relevant_metrics_v1/... or root/...
    candidate = root / "clinically_relevant_metrics_v1" / "batch_wham_experiments"
    if candidate.exists():
        return candidate
    candidate = root / "batch_wham_experiments"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not locate batch_wham_experiments inside {root}")


def validation_root_from_input(render_input: Path, work_dir: Path) -> Path:
    root = unpack_or_use(render_input, work_dir / "render_validation")
    candidate = root / "render_input_validation"
    if candidate.exists():
        return candidate
    if root.name == "render_input_validation":
        return root
    raise FileNotFoundError(f"Could not locate render_input_validation inside {root}")


def load_summary_tables(metrics_base: Path, seqs: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for _, s in seqs.iterrows():
        subject = str(s["subject"])
        sequence = str(s["sequence"])
        p = metrics_base / subject / sequence / f"{subject}_{sequence}_all_experiments_metric_summary.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing metric summary for {subject} {sequence}: {p}")
        df = pd.read_csv(p)
        if "experiment" not in df.columns:
            raise ValueError(f"Summary file has no 'experiment' column: {p}")
        df["experiment"] = df["experiment"].astype(str)
        df["subject"] = subject
        df["sequence"] = sequence
        out[subject] = df
    return out


def load_validation_table(validation_base: Path, seqs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in seqs.iterrows():
        subject = str(s["subject"])
        sequence = str(s["sequence"])
        stages = [
            ("SMPL to SMPL-X render input", validation_base / f"{subject}_{sequence}" / "render_input_motion_validation_recommended_final_recommended_metrics.csv"),
            ("Coordinate correction", validation_base / f"root_rotation_effect_{sequence}" / "root_rotation_effect_recommended_final_recommended_metrics.csv"),
        ]
        for stage, p in stages:
            if not p.exists():
                raise FileNotFoundError(f"Missing validation file: {p}")
            raw = pd.read_csv(p)
            d = {"subject": subject, "sequence": sequence, "stage": stage}
            # Expected layout: metric,value
            if {"metric", "value"}.issubset(raw.columns):
                for _, r in raw.iterrows():
                    d[str(r["metric"])] = r["value"]
            else:
                raise ValueError(f"Unexpected validation table format: {p}")
            rows.append(d)
    return pd.DataFrame(rows)


def per_frame_file(metrics_base: Path, subject: str, sequence: str, experiment: str) -> Optional[Path]:
    folder = metrics_base / subject / sequence / experiment
    if not folder.exists():
        return None
    matches = sorted(folder.glob("*_per_frame_metrics.csv"))
    return matches[0] if matches else None


def gait_events_file(metrics_base: Path, subject: str, sequence: str, experiment: str) -> Optional[Path]:
    folder = metrics_base / subject / sequence / experiment
    if not folder.exists():
        return None
    matches = sorted(folder.glob("*_gait_events_steps_strides.csv"))
    return matches[0] if matches else None

# -----------------------------
# Derived result tables
# -----------------------------

def build_validation_summary(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, g in validation.groupby("stage", sort=False):
        interpretation = "Motion preserved before WHAM" if "SMPL" in stage else "Coordinate correction did not materially distort gait"
        rows.append({
            "Validation stage": stage,
            "Mean PA-MPJPE (mm)": g["PA-MPJPE"].mean(),
            "Max PA-MPJPE (mm)": g["PA-MPJPE"].max(),
            "Mean cadence error": g["Cadence error"].mean(),
            "Max foot clearance error (mm)": g["Foot clearance error"].max(),
            "Interpretation": interpretation,
        })
    # Match thesis order: coordinate correction first, then SMPL-to-SMPL-X.
    order = ["Coordinate correction", "SMPL to SMPL-X render input"]
    return pd.DataFrame(rows).set_index("Validation stage").loc[order].reset_index()


def build_sub01_full_table(summary_by_subject: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    sub01 = summary_by_subject["SUB01"].copy()
    ids = [eid for eid in sort_experiment_ids(sub01["experiment"].unique()) if eid in FULL_CONDITION_MAP]
    sub01 = sub01.set_index("experiment").loc[ids].reset_index()
    out = pd.DataFrame({
        "ID": sub01["experiment"],
        "Condition": [FULL_CONDITION_MAP[eid][5] for eid in sub01["experiment"]],
        "Frames": sub01["frames_used"],
        "PA-MPJPE": sub01["pa_mpjpe_mean_mm"],
        "GT events": sub01["gt_num_gait_events"],
        "WHAM events": sub01["wham_num_gait_events"],
        "Foot clearance error": sub01["foot_clearance_mean_abs_error_mm"],
        "Step length error": sub01["step_length_mean_abs_error_mm"],
        "Stride length error": sub01["stride_length_mean_abs_error_mm"],
        "Knee ROM error": sub01["knee_rom_mean_abs_error_deg"],
    })
    return out


def build_reduced_matrix_table() -> pd.DataFrame:
    rows = []
    for rid in ["R0", "R1", "R2", "R3", "R4", "R5"]:
        d = REDUCED_CONDITIONS[rid]
        rows.append({
            "R ID": rid,
            "Full-matrix ID": d["source_id"],
            "Environment": d["environment"],
            "Camera view": d["camera"],
            "Occlusion": d["occlusion"],
        })
    return pd.DataFrame(rows)


def build_aligned_metrics(summary_by_subject: Dict[str, pd.DataFrame], seqs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in seqs.iterrows():
        subject = str(s["subject"])
        sequence = str(s["sequence"])
        summary = summary_by_subject[subject].set_index("experiment")
        out_map = PRIMARY_OUTPUT_MAP if subject == "SUB01" else REPLICATION_OUTPUT_MAP
        for rid in ["R0", "R1", "R2", "R3", "R4", "R5"]:
            output_id = out_map[rid]
            if output_id not in summary.index:
                continue
            r = summary.loc[output_id]
            cond = REDUCED_CONDITIONS[rid]
            rows.append({
                "Subject": subject,
                "R ID": rid,
                "Source ID": cond["source_id"],
                "Output ID": output_id,
                "Condition": cond["condition"],
                "Frames": r.get("frames_used", np.nan),
                "PA-MPJPE": r.get("pa_mpjpe_mean_mm", np.nan),
                "GT events": r.get("gt_num_gait_events", np.nan),
                "WHAM events": r.get("wham_num_gait_events", np.nan),
                "Foot clearance error": r.get("foot_clearance_mean_abs_error_mm", np.nan),
                "Step length error": r.get("step_length_mean_abs_error_mm", np.nan),
                "Stride length error": r.get("stride_length_mean_abs_error_mm", np.nan),
                "Knee ROM error": r.get("knee_rom_mean_abs_error_deg", np.nan),
                "sequence": sequence,
            })
    return pd.DataFrame(rows)


def build_reduced_only_table(aligned: pd.DataFrame) -> pd.DataFrame:
    # This is the thesis Table 4.4: only the additional walking sequences.
    return aligned[aligned["Subject"] != "SUB01"].copy()


def table_for_results(df: pd.DataFrame, include_output_id: bool = False) -> pd.DataFrame:
    cols = ["Subject", "R ID", "Source ID"]
    if include_output_id:
        cols.append("Output ID")
    cols += ["Condition", "Frames", "PA-MPJPE", "GT events", "WHAM events", "Foot clearance error", "Step length error", "Stride length error"]
    return df[cols].copy()

# -----------------------------
# Plotting helpers
# -----------------------------

def plot_validation(validation: pd.DataFrame, figs_dir: Path) -> None:
    subjects = list(dict.fromkeys(validation["subject"].astype(str)))
    stages = ["SMPL to SMPL-X render input", "Coordinate correction"]
    x = np.arange(len(subjects))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for i, stage in enumerate(stages):
        vals = []
        for sub in subjects:
            row = validation[(validation["subject"] == sub) & (validation["stage"] == stage)]
            vals.append(float(row["PA-MPJPE"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width, label=stage)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    set_common_axes(ax, "Pre-WHAM validation PA-MPJPE", ylabel="PA-MPJPE (mm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_1_pipeline_validation_pa_mpjpe.png", dpi=220)
    plt.close(fig)


def plot_sub01_pa(summary_by_subject: Dict[str, pd.DataFrame], figs_dir: Path) -> None:
    df = summary_by_subject["SUB01"].copy()
    ids = [eid for eid in sort_experiment_ids(df["experiment"].unique()) if eid in FULL_CONDITION_MAP]
    df = df.set_index("experiment").loc[ids].reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.bar(df["experiment"], df["pa_mpjpe_mean_mm"])
    set_common_axes(ax, "SUB01 full matrix: PA-MPJPE by experiment", ylabel="PA-MPJPE (mm)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_2_sub01_pa_mpjpe_by_experiment.png", dpi=220)
    plt.close(fig)


def plot_sub01_spatial(summary_by_subject: Dict[str, pd.DataFrame], figs_dir: Path) -> None:
    df = summary_by_subject["SUB01"].copy()
    ids = [eid for eid in sort_experiment_ids(df["experiment"].unique()) if eid in FULL_CONDITION_MAP]
    df = df.set_index("experiment").loc[ids].reset_index()
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 4.9))
    metrics = [
        ("foot_clearance_mean_abs_error_mm", "Foot clearance error (mm)"),
        ("step_length_mean_abs_error_mm", "Step length error (mm)"),
        ("stride_length_mean_abs_error_mm", "Stride length error (mm)"),
    ]
    for i, (col, label) in enumerate(metrics):
        ax.bar(x + (i - 1) * width, df[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(df["experiment"], rotation=35)
    set_common_axes(ax, "SUB01 full matrix: spatial gait-feature errors", ylabel="Error (mm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_3_sub01_spatial_gait_errors.png", dpi=220)
    plt.close(fig)


def plot_sub01_pa_vs_stride(summary_by_subject: Dict[str, pd.DataFrame], figs_dir: Path) -> None:
    df = summary_by_subject["SUB01"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(df["pa_mpjpe_mean_mm"], df["stride_length_mean_abs_error_mm"])
    for _, row in df.iterrows():
        ax.annotate(str(row["experiment"]), (row["pa_mpjpe_mean_mm"], row["stride_length_mean_abs_error_mm"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    set_common_axes(ax, "SUB01: PA-MPJPE versus stride-length error", xlabel="PA-MPJPE (mm)", ylabel="Stride length error (mm)")
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_4_sub01_pa_vs_stride_scatter.png", dpi=220)
    plt.close(fig)


def plot_frame_level_pa(metrics_base: Path, figs_dir: Path) -> None:
    subject, sequence = "SUB01", "SUB01_off_walk_1"
    selected = ["E1", "E3", "E8", "E10", "E12"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for exp in selected:
        p = per_frame_file(metrics_base, subject, sequence, exp)
        if p is None:
            continue
        df = pd.read_csv(p)
        ax.plot(df["frame"], df["pa_mpjpe_mm"], linewidth=1.2, label=exp)
    set_common_axes(ax, "Frame-level PA-MPJPE: SUB01", xlabel="Frame", ylabel="PA-MPJPE (mm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_5_sub01_frame_level_pa_mpjpe_selected.png", dpi=220)
    plt.close(fig)


def plot_frame_delta_vs_home(metrics_base: Path, figs_dir: Path) -> None:
    subject, sequence = "SUB01", "SUB01_off_walk_1"
    # Prefer the file generated by the degradation analysis script, because it already aligns selected conditions to E1.
    delta_path = metrics_base / subject / sequence / "degradation_analysis" / "frame_level_delta_vs_E1_selected.csv"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if delta_path.exists():
        delta = pd.read_csv(delta_path)
        for exp in ["E3", "E8", "E12"]:
            g = delta[delta["experiment"].astype(str) == exp]
            if len(g):
                ax.plot(g["frame"], g["delta_vs_E1_pa_mpjpe_mm"], linewidth=1.2, label=f"{exp} vs E1")
    else:
        # Fallback: compute directly from per-frame files.
        base_path = per_frame_file(metrics_base, subject, sequence, "E1")
        if base_path is None:
            raise FileNotFoundError("Missing E1 per-frame metrics needed for frame-delta plot")
        base = pd.read_csv(base_path)[["frame", "pa_mpjpe_mm"]].rename(columns={"pa_mpjpe_mm": "pa_base"})
        for exp in ["E3", "E8", "E12"]:
            p = per_frame_file(metrics_base, subject, sequence, exp)
            if p is None:
                continue
            df = pd.read_csv(p)[["frame", "pa_mpjpe_mm"]].merge(base, on="frame", how="inner")
            ax.plot(df["frame"], df["pa_mpjpe_mm"] - df["pa_base"], linewidth=1.2, label=f"{exp} vs E1")
    # Vertical dotted lines: ground-truth contact events from E1.
    events_path = gait_events_file(metrics_base, subject, sequence, "E1")
    if events_path is not None:
        ev = pd.read_csv(events_path)
        if {"source", "type", "frame"}.issubset(ev.columns):
            gt_contacts = ev[(ev["source"] == "gt") & (ev["type"] == "contact_event")]["frame"].dropna().astype(int).tolist()
            for f in gt_contacts:
                ax.axvline(f, linestyle=":", linewidth=0.8, alpha=0.35)
    ax.axhline(0, linewidth=0.8)
    set_common_axes(ax, "Frame-specific pose degradation relative to the home baseline", xlabel="Frame", ylabel=r"$\Delta$ PA-MPJPE relative to E1 (mm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig_frame_delta_pa_mpjpe_vs_home_baseline.png", dpi=220)
    plt.close(fig)


def plot_aligned_grouped(aligned: pd.DataFrame, figs_dir: Path) -> None:
    # Figures 6--8 in the current Result section.
    specs = [
        ("PA-MPJPE", "fig_6_aligned_pa_mpjpe_by_condition_subject.png", "Aligned comparison set: PA-MPJPE by condition and sequence", "PA-MPJPE (mm)"),
        ("Foot clearance error", "fig_7_aligned_foot_clearance_by_condition_subject.png", "Aligned comparison set: foot-clearance error by condition and sequence", "Foot-clearance error (mm)"),
        ("Stride length error", "fig_8_aligned_stride_length_by_condition_subject.png", "Aligned comparison set: stride-length error by condition and sequence", "Stride-length error (mm)"),
    ]
    subjects = list(dict.fromkeys(aligned["Subject"]))
    rids = ["R0", "R1", "R2", "R3", "R4", "R5"]
    x = np.arange(len(rids))
    width = 0.82 / max(1, len(subjects))
    for col, filename, title, ylabel in specs:
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        for i, subject in enumerate(subjects):
            vals = []
            for rid in rids:
                sub = aligned[(aligned["Subject"] == subject) & (aligned["R ID"] == rid)]
                vals.append(float(sub[col].iloc[0]) if len(sub) else np.nan)
            ax.bar(x + (i - (len(subjects) - 1) / 2) * width, vals, width, label=subject)
        ax.set_xticks(x)
        ax.set_xticklabels(rids)
        set_common_axes(ax, title, xlabel="Aligned condition", ylabel=ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs_dir / filename, dpi=220)
        plt.close(fig)

# -----------------------------
# Visual snapshot panel
# -----------------------------

def frame_number_from_name(path: Path) -> Optional[int]:
    nums = re.findall(r"(\d+)", path.stem)
    if not nums:
        return None
    return int(nums[-1])


def find_render_frame(render_root: Path, subject: str, sequence: str, experiment: str, frame: int) -> Optional[Path]:
    candidates_dirs = [
        render_root / subject / experiment,
        render_root / subject / sequence / experiment,
        render_root / sequence / experiment,
        render_root / experiment,
    ]
    image_exts = ["*.png", "*.jpg", "*.jpeg"]
    for d in candidates_dirs:
        if not d.exists():
            continue
        imgs: List[Path] = []
        for ext in image_exts:
            imgs.extend(sorted(d.glob(ext)))
        if not imgs:
            continue
        # Prefer exact frame-number match. Accept zero- or one-based names by trying both frame and frame+1.
        by_num = {frame_number_from_name(p): p for p in imgs if frame_number_from_name(p) is not None}
        for k in [frame, frame + 1, frame - 1]:
            if k in by_num:
                return by_num[k]
        # Fallback to positional index.
        if 0 <= frame < len(imgs):
            return imgs[frame]
    return None


def make_visual_placeholder(out_path: Path, note: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.text(
        0.5, 0.55,
        "Visual frame examples not generated from metric CSV files",
        ha="center", va="center", fontsize=16, fontweight="bold",
    )
    ax.text(
        0.5, 0.40,
        textwrap.fill(note, width=100),
        ha="center", va="center", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_visual_examples(figs_dir: Path, visual_examples: Optional[Path], render_frames_root: Optional[Path]) -> str:
    out_path = figs_dir / "fig_peak_frame_visual_examples.jpg"
    if visual_examples and visual_examples.exists():
        shutil.copy2(visual_examples, out_path)
        return f"Copied visual example panel from {visual_examples}."

    note = (
        "This snapshot panel requires rendered RGB frames, not only metric CSV files. "
        "To reproduce it, rerun this script with --render-frames-root pointing to the Unreal PNG/JPG render folders. "
        "The script will sample the peak/error frames for E3, E8, E12, E4, and E10."
    )
    if not render_frames_root:
        make_visual_placeholder(out_path, note)
        return "Created placeholder visual example panel because no --render-frames-root or --visual-examples was provided."

    examples = [
        ("E3 frontal/table", "SUB01", "SUB01_off_walk_1", "E3", 301),
        ("E8 strong occlusion", "SUB01", "SUB01_off_walk_1", "E8", 476),
        ("E12 partial occlusion", "SUB01", "SUB01_off_walk_1", "E12", 323),
        ("E4 upper corner", "SUB01", "SUB01_off_walk_1", "E4", 301),
        ("E10 upper-corner occlusion", "SUB01", "SUB01_off_walk_1", "E10", 301),
    ]
    image_paths = []
    labels = []
    missing = []
    for label, sub, seq, exp, frame in examples:
        p = find_render_frame(render_frames_root, sub, seq, exp, frame)
        if p is None:
            missing.append(f"{label} frame {frame}")
        else:
            image_paths.append(p)
            labels.append(f"{label}, frame {frame}")
    if not image_paths:
        make_visual_placeholder(out_path, note + " No matching rendered frames were found in the supplied folder.")
        return "Created placeholder visual example panel because no matching rendered frames were found."

    n = len(image_paths)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.4 * rows))
    axes_arr = np.array(axes).reshape(-1)
    for ax, p, label in zip(axes_arr, image_paths, labels):
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    for ax in axes_arr[len(image_paths):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    if missing:
        return "Created visual example panel, but missing: " + "; ".join(missing)
    return "Created visual example panel from rendered frames."

# -----------------------------
# LaTeX/README generation
# -----------------------------

def write_results_figure_manifest(figs_dir: Path, out_dir: Path) -> None:
    rows = []
    for name in EXPECTED_FIGURES:
        p = figs_dir / name
        rows.append({"Figure file": f"figures/{name}", "Created": p.exists(), "Size bytes": p.stat().st_size if p.exists() else 0})
    pd.DataFrame(rows).to_csv(out_dir / "figure_reproduction_manifest.csv", index=False)


def write_readme(out_dir: Path, visual_note: str) -> None:
    text = f"""
    Results table/figure reproduction pack
    ======================================

    This folder was generated by 10_make_results_tables_figures_complete.py.

    Main outputs:
    - figures/: all figure files referenced in the current Results section.
    - tables_csv/: CSV versions of the result tables.
    - tables_latex/: LaTeX table versions of the result tables.
    - figure_reproduction_manifest.csv: checklist of expected figure files.
    - source_data/: compact CSV files derived from the metric outputs.

    Reproduction command used for the default thesis data:
        python scripts\\10_make_results_tables_figures_complete.py ^
            --metrics-input clinically_relevant_metrics_v1.zip ^
            --render-validation-input render_input_validation.zip ^
            --sequences-csv results\\selected_first_four_bmclab_sequences.csv ^
            --out-dir results\\final_results_tables_figures

    Notes about the visual snapshot panel:
    {visual_note}

    LaTeX figure paths in 5-Results.tex assume the PNG/JPG files are placed in a
    folder called figures/ relative to the Results .tex file.
    """
    (out_dir / "README_reproducibility.txt").write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate all Results chapter tables and figures from metric outputs.")
    ap.add_argument("--metrics-input", default="/mnt/data/clinically_relevant_metrics_v1.zip", help="ZIP or folder containing clinically_relevant_metrics_v1/batch_wham_experiments.")
    ap.add_argument("--render-validation-input", default="/mnt/data/render_input_validation.zip", help="ZIP or folder containing render_input_validation.")
    ap.add_argument("--sequences-csv", default="/mnt/data/selected_first_four_bmclab_sequences.csv", help="CSV with subject and sequence columns.")
    ap.add_argument("--out-dir", default="/mnt/data/final_results_tables_figures_repro", help="Output folder.")
    ap.add_argument("--visual-examples", default=None, help="Optional existing JPG/PNG panel to copy as fig_peak_frame_visual_examples.jpg.")
    ap.add_argument("--render-frames-root", default=None, help="Optional root folder containing rendered Unreal PNG/JPG frames for visual panel reproduction.")
    ap.add_argument("--zip-output", action="store_true", help="Also create a ZIP archive next to --out-dir.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_clean_dir(out_dir)
    work_dir = out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    tables_csv_dir = out_dir / "tables_csv"
    tables_tex_dir = out_dir / "tables_latex"
    source_dir = out_dir / "source_data"
    for d in [figs_dir, tables_csv_dir, tables_tex_dir, source_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metrics_base = metric_root_from_input(Path(args.metrics_input), work_dir)
    validation_base = validation_root_from_input(Path(args.render_validation_input), work_dir)
    seqs = load_sequences(Path(args.sequences_csv))
    summary_by_subject = load_summary_tables(metrics_base, seqs)
    validation = load_validation_table(validation_base, seqs)

    validation_summary = build_validation_summary(validation)
    sub01_full = build_sub01_full_table(summary_by_subject)
    reduced_matrix = build_reduced_matrix_table()
    aligned = build_aligned_metrics(summary_by_subject, seqs)
    reduced_only = build_reduced_only_table(aligned)

    # Tables used in the current Results chapter.
    save_csv_and_table(
        validation_summary,
        tables_csv_dir / "table_1_pipeline_validation_summary.csv",
        tables_tex_dir / "table_1_pipeline_validation_summary.tex",
        "Pipeline-validation summary across the selected walking sequences.",
        "tab:pipeline_validation_updated",
    )
    save_csv_and_table(
        sub01_full,
        tables_csv_dir / "table_2_sub01_full_metrics.csv",
        tables_tex_dir / "table_2_sub01_full_metrics.tex",
        "Main WHAM recovery metrics for the full primary experiment matrix.",
        "tab:sub01_full_metrics",
    )
    save_csv_and_table(
        reduced_matrix,
        tables_csv_dir / "table_3_reduced_matrix.csv",
        tables_tex_dir / "table_3_reduced_matrix.tex",
        "Reduced experiment matrix used for the additional walking sequences.",
        "tab:reduced_matrix",
    )
    save_csv_and_table(
        table_for_results(reduced_only, include_output_id=False),
        tables_csv_dir / "table_4_reduced_metrics_additional_sequences.csv",
        tables_tex_dir / "table_4_reduced_metrics_additional_sequences.tex",
        "Reduced-matrix WHAM recovery metrics for the additional walking sequences.",
        "tab:reduced_metrics_all_sequences",
    )
    save_csv_and_table(
        table_for_results(aligned, include_output_id=False),
        tables_csv_dir / "table_5_aligned_metrics_all_sequences.csv",
        tables_tex_dir / "table_5_aligned_metrics_all_sequences.tex",
        "Aligned WHAM recovery metrics across the primary and additional walking sequences.",
        "tab:aligned_metrics_all_sequences",
    )

    # Extra source/diagnostic CSVs.
    validation.to_csv(source_dir / "pipeline_validation_all_sequences_long.csv", index=False)
    aligned.to_csv(source_dir / "aligned_metrics_with_output_ids.csv", index=False)
    for subject, df in summary_by_subject.items():
        df.to_csv(source_dir / f"{subject}_metric_summary_raw.csv", index=False)

    # Figures used in the current Results chapter.
    plot_validation(validation, figs_dir)
    plot_sub01_pa(summary_by_subject, figs_dir)
    plot_sub01_spatial(summary_by_subject, figs_dir)
    plot_sub01_pa_vs_stride(summary_by_subject, figs_dir)
    plot_frame_level_pa(metrics_base, figs_dir)
    plot_frame_delta_vs_home(metrics_base, figs_dir)
    visual_note = make_visual_examples(
        figs_dir,
        Path(args.visual_examples) if args.visual_examples else None,
        Path(args.render_frames_root) if args.render_frames_root else None,
    )
    plot_aligned_grouped(aligned, figs_dir)

    write_results_figure_manifest(figs_dir, out_dir)
    write_readme(out_dir, visual_note)

    # Copy this script into the output folder for exact reproducibility.
    try:
        shutil.copy2(Path(__file__), out_dir / "10_make_results_tables_figures_complete.py")
    except Exception:
        pass

    # Verify that every Results-section figure exists.
    missing = [name for name in EXPECTED_FIGURES if not (figs_dir / name).exists()]
    if missing:
        raise RuntimeError("Missing expected figure files: " + ", ".join(missing))

    if args.zip_output:
        zip_path = out_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in out_dir.rglob("*"):
                if "_work" in p.parts:
                    continue
                z.write(p, p.relative_to(out_dir))
        print(f"Wrote ZIP: {zip_path}")
    print(f"Wrote results tables and figures to: {out_dir}")


if __name__ == "__main__":
    main()
