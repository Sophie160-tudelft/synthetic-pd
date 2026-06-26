"""
Select the first four BMCLab sequences from the current experiment table.

Included sequences:
1. SUB01 / SUB01_off_walk_1   - original sequence
2. SUB02 / SUB02_off_walk_2   - non-freezer, UPDRS_GAIT 0
3. SUB08 / SUB08_off_walk_4   - non-freezer, UPDRS_GAIT 1
4. SUB17 / SUB17_off_walk_5   - freezer, UPDRS_GAIT 1

Excluded:
- SUB01 / SUB01_off_walk_13

Run from the project root:
    python scripts\select_first_four_bmclab_sequences.py

Optional:
    python scripts\select_first_four_bmclab_sequences.py --bmclab-pkl data\raw\BMCLab.pkl
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


SELECTED_SEQUENCES = [
    {
        "purpose": "original sequence",
        "subject": "SUB01",
        "sequence": "SUB01_off_walk_1",
    },
    {
        "purpose": "non-freezer, normal/mildest gait",
        "subject": "SUB02",
        "sequence": "SUB02_off_walk_2",
    },
    {
        "purpose": "non-freezer, moderate gait score",
        "subject": "SUB08",
        "sequence": "SUB08_off_walk_4",
    },
    {
        "purpose": "freezer, moderate gait score",
        "subject": "SUB17",
        "sequence": "SUB17_off_walk_5",
    },
]


def scalar_to_python(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def load_bmclab(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def get_sequence_metadata(bmclab, subject, sequence, purpose):
    if subject not in bmclab:
        raise KeyError(f"Subject not found in BMCLab.pkl: {subject}")

    if sequence not in bmclab[subject]:
        available = list(bmclab[subject].keys())[:20]
        raise KeyError(
            f"Sequence not found for {subject}: {sequence}\n"
            f"First available sequences for this subject: {available}"
        )

    seq = bmclab[subject][sequence]

    pose = np.asarray(seq["pose"])
    trans = np.asarray(seq["trans"])
    beta = np.asarray(seq["beta"]) if "beta" in seq else None

    fps = scalar_to_python(seq.get("fps", seq.get("mocap_framerate", None)))
    updrs_gait = scalar_to_python(seq.get("UPDRS_GAIT", None))
    medication = scalar_to_python(seq.get("medication", None))
    freezer_group = scalar_to_python(seq.get("other", None))

    frames = int(pose.shape[0])
    duration_s = float((frames - 1) / float(fps)) if fps else None

    return {
        "purpose": purpose,
        "subject": subject,
        "sequence": sequence,
        "UPDRS_GAIT": int(updrs_gait) if updrs_gait is not None else None,
        "freezer_group": str(freezer_group),
        "medication": str(medication),
        "frames": frames,
        "fps": int(fps) if fps is not None else None,
        "duration_s": duration_s,
        "pose_shape": str(tuple(pose.shape)),
        "trans_shape": str(tuple(trans.shape)),
        "beta_shape": str(tuple(beta.shape)) if beta is not None else "",
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "purpose",
        "subject",
        "sequence",
        "UPDRS_GAIT",
        "freezer_group",
        "medication",
        "frames",
        "fps",
        "duration_s",
        "pose_shape",
        "trans_shape",
        "beta_shape",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print("\nSelected BMCLab sequences")
    print("-------------------------")
    header = (
        f"{'Purpose':<36} {'Subject':<8} {'Sequence':<22} "
        f"{'UPDRS':<6} {'Freezer group':<14} {'Medication':<10} {'Frames':<7}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['purpose']:<36} "
            f"{r['subject']:<8} "
            f"{r['sequence']:<22} "
            f"{r['UPDRS_GAIT']:<6} "
            f"{r['freezer_group']:<14} "
            f"{r['medication']:<10} "
            f"{r['frames']:<7}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bmclab-pkl",
        type=str,
        default=str(Path("data") / "raw" / "BMCLab.pkl"),
        help="Path to BMCLab.pkl",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(Path("results") / "selected_first_four_bmclab_sequences.csv"),
        help="Where to save the selected sequence table",
    )
    args = parser.parse_args()

    bmclab_path = Path(args.bmclab_pkl)
    if not bmclab_path.exists():
        raise FileNotFoundError(f"BMCLab file not found: {bmclab_path}")

    bmclab = load_bmclab(bmclab_path)

    rows = []
    for item in SELECTED_SEQUENCES:
        rows.append(
            get_sequence_metadata(
                bmclab=bmclab,
                subject=item["subject"],
                sequence=item["sequence"],
                purpose=item["purpose"],
            )
        )

    print_table(rows)

    write_csv(args.out_csv, rows)

    print("\nSaved CSV:")
    print(Path(args.out_csv).resolve())

    print("\nPipeline commands:")
    for r in rows:
        print(
            f"python scripts\\00_fit_smpl_to_smplx_one_trial_auto_frames.py "
            f"--subject {r['subject']} --trial {r['sequence']}"
        )


if __name__ == "__main__":
    main()
