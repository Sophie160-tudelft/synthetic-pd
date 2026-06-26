from pathlib import Path
import argparse
import pickle
import re
import shutil
import subprocess
import sys


# =============================================================================
# BATCH CONVERT UNREAL PNG RENDERS TO WHAM-READY RGB MP4 VIDEOS
# =============================================================================
# Expected input structure:
#
#   D:\Experiment_renders\SUB01\E0\*.png
#   D:\Experiment_renders\SUB01\E1\*.png
#   D:\Experiment_renders\SUB01\E2\*.png
#   ...
#
# Output structure:
#
#   C:\Users\sopha\synthetic-pd\data\videos\SUB01\E0\SUB01_E0_wham_rgb_150fps.mp4
#   C:\Users\sopha\synthetic-pd\data\videos\SUB01\E1\SUB01_E1_wham_rgb_150fps.mp4
#   C:\Users\sopha\synthetic-pd\data\videos\SUB01\E2\SUB01_E2_wham_rgb_150fps.mp4
#   ...
#
# Change SUBJECT below when you want to process another subject.
# =============================================================================


# -----------------------------------------------------------------------------
# Main settings you will usually change
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\sopha\synthetic-pd")
INPUT_ROOT = Path(r"D:\Experiment_renders")
OUTPUT_ROOT = PROJECT_ROOT / "data" / "videos"

SUBJECT = "SUB17"          # Change to "SUB02", "SUB08", "SUB17", etc.
FPS = 150

# Optional frame-count check. Leave as None if you only want to use the PNG count.
# If TRIAL is set and EXPECTED_FRAMES is None, the script can infer the expected
# frame count from BMCLab.pkl.
EXPECTED_FRAMES = None
BMCLAB_PKL = PROJECT_ROOT / "data" / "raw" / "BMCLab.pkl"
TRIAL = "SUB17_off_walk_5"  # Change with SUBJECT, or set to None to skip BMCLab inference.

# Process only experiment folders with this prefix.
EXPERIMENT_PREFIX = "E"

# If True, existing MP4s are overwritten.
OVERWRITE = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def natural_sort_key(path_or_string):
    """
    Sorts E0, E1, E2, ..., E10 correctly instead of E0, E1, E10, E2.
    """
    text = str(path_or_string)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def infer_expected_frames_from_bmclab(bmclab_pkl: Path, subject: str, trial: str) -> int:
    with bmclab_pkl.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    seq = data[subject][trial]
    pose = seq.get("pose", seq.get("poses"))
    if pose is None:
        raise KeyError(f"Could not find pose/poses for {subject} {trial}")

    return int(len(pose))


def extract_frame_number(path: Path) -> int:
    """
    Extracts frame number from common Unreal/Movie Render Queue PNG names, e.g.:

        {demo}.0000.png
        SUB01_E0.0670.png
        frame_000123.png

    The script uses the last numeric group before .png.
    """
    matches = re.findall(r"(\d+)(?=\.png$)", path.name, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\d+", path.stem)

    if not matches:
        raise ValueError(
            f"Could not extract frame number from filename: {path.name}\n"
            "Expected a filename containing a frame number, for example .0000.png."
        )

    return int(matches[-1])


def run_command(cmd):
    print("\nRunning command:")
    print(" ".join(str(c) for c in cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError("Command failed.")


def find_experiment_dirs(subject_dir: Path, experiment_prefix: str):
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject render folder does not exist: {subject_dir}")

    experiment_dirs = [
        p for p in subject_dir.iterdir()
        if p.is_dir() and p.name.startswith(experiment_prefix)
    ]

    experiment_dirs = sorted(experiment_dirs, key=lambda p: natural_sort_key(p.name))

    if not experiment_dirs:
        raise FileNotFoundError(
            f"No experiment folders starting with {experiment_prefix!r} found in:\n{subject_dir}"
        )

    return experiment_dirs


def find_png_files(experiment_dir: Path):
    """
    Prefer PNGs directly inside E0/E1/etc.
    If none are found directly, search recursively one level/deeper.
    """
    png_files = list(experiment_dir.glob("*.png"))

    if not png_files:
        png_files = list(experiment_dir.rglob("*.png"))

    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {experiment_dir}")

    return sorted(png_files, key=lambda p: (extract_frame_number(p), str(p)))


def check_frame_sequence(png_files, expected_frames):
    frame_numbers = [extract_frame_number(p) for p in png_files]
    first_number = frame_numbers[0]
    last_number = frame_numbers[-1]

    print(f"Number of PNG files: {len(png_files)}")
    print(f"First frame: {png_files[0].name}")
    print(f"Last frame:  {png_files[-1].name}")
    print(f"First frame number: {first_number}")
    print(f"Last frame number:  {last_number}")

    if expected_frames is not None and len(png_files) != expected_frames:
        print(
            f"\nWARNING: Expected {expected_frames} frames, "
            f"but found {len(png_files)} PNG files."
        )

    expected_numbers = list(range(first_number, last_number + 1))
    missing_numbers = sorted(set(expected_numbers) - set(frame_numbers))

    if missing_numbers:
        print("\nWARNING: Missing frame numbers detected.")
        print("First missing numbers:", missing_numbers[:20])
    else:
        print("No missing frame numbers detected.")


def make_video_from_pngs(
    png_files,
    output_path: Path,
    fps: int,
    overwrite: bool,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"\nSkipping existing video: {output_path}")
        return

    concat_file = output_path.parent / f"_{output_path.stem}_ffmpeg_frames.txt"

    with concat_file.open("w", encoding="utf-8") as f:
        for png in png_files:
            safe_path = str(png.resolve()).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-r",
        str(fps),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "slow",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    try:
        run_command(ffmpeg_cmd)
    finally:
        concat_file.unlink(missing_ok=True)

    print("\nSaved video:")
    print(output_path)

    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration,nb_read_frames,pix_fmt",
        "-of",
        "default=nokey=0:noprint_wrappers=1",
        str(output_path),
    ]

    print("\nOutput video check")
    print("------------------")
    run_command(ffprobe_cmd)


def check_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found. Install it with:\n"
            "winget install Gyan.FFmpeg\n"
            "Then restart VS Code or PowerShell."
        )

    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffprobe was not found. It should come with FFmpeg.\n"
            "Install FFmpeg with:\n"
            "winget install Gyan.FFmpeg\n"
            "Then restart VS Code or PowerShell."
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch convert all experiment PNG folders for one subject to "
            "WHAM-ready RGB MP4 videos."
        )
    )

    parser.add_argument("--subject", default=SUBJECT, help="Subject folder, e.g. SUB01.")
    parser.add_argument("--input_root", default=str(INPUT_ROOT), help="Root render folder.")
    parser.add_argument("--output_root", default=str(OUTPUT_ROOT), help="Root output video folder.")
    parser.add_argument("--fps", type=int, default=FPS, help="Video FPS.")
    parser.add_argument("--expected_frames", type=int, default=EXPECTED_FRAMES)
    parser.add_argument("--bmclab_pkl", default=str(BMCLAB_PKL))
    parser.add_argument("--trial", default=TRIAL, help="BMCLab trial name. Use None to skip inference.")
    parser.add_argument("--experiment_prefix", default=EXPERIMENT_PREFIX)
    parser.add_argument("--no_overwrite", action="store_true", help="Do not overwrite existing MP4 files.")

    args = parser.parse_args()

    check_ffmpeg_available()

    subject = args.subject
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    subject_dir = input_root / subject
    subject_output_dir = output_root / subject
    overwrite = not args.no_overwrite

    expected_frames = args.expected_frames

    trial = args.trial
    if isinstance(trial, str) and trial.lower() in {"none", "", "skip"}:
        trial = None

    if expected_frames is None and trial is not None:
        bmclab_pkl = Path(args.bmclab_pkl)
        if not bmclab_pkl.is_absolute():
            bmclab_pkl = PROJECT_ROOT / bmclab_pkl

        if bmclab_pkl.exists():
            expected_frames = infer_expected_frames_from_bmclab(
                bmclab_pkl=bmclab_pkl.resolve(),
                subject=subject,
                trial=trial,
            )
            print(f"Inferred expected frame count from BMCLab: {expected_frames}")
        else:
            print(f"BMCLab.pkl not found, skipping expected-frame inference: {bmclab_pkl}")

    print("\nBatch video conversion")
    print("======================")
    print(f"Subject:       {subject}")
    print(f"Input folder:  {subject_dir}")
    print(f"Output folder: {subject_output_dir}")
    print(f"FPS:           {args.fps}")
    print(f"Overwrite:     {overwrite}")

    experiment_dirs = find_experiment_dirs(subject_dir, args.experiment_prefix)

    print("\nExperiments found:")
    for exp_dir in experiment_dirs:
        print(f"  {exp_dir.name}")

    for exp_dir in experiment_dirs:
        experiment_id = exp_dir.name
        output_path = (
            subject_output_dir
            / experiment_id
            / f"{subject}_{experiment_id}_wham_rgb_{args.fps}fps.mp4"
        )

        print("\n" + "=" * 80)
        print(f"Processing {subject} / {experiment_id}")
        print("=" * 80)
        print(f"Input experiment folder:  {exp_dir}")
        print(f"Output video path:        {output_path}")

        png_files = find_png_files(exp_dir)
        check_frame_sequence(png_files, expected_frames)
        make_video_from_pngs(
            png_files=png_files,
            output_path=output_path,
            fps=args.fps,
            overwrite=overwrite,
        )

    print("\nDone. All MP4 videos are ready for WHAM.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
