from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data.export_smpl_npz import export_recording_to_npz


PKL_PATH = Path("data/raw/BMCLab.pkl")
OUTPUT_DIR = Path("data/processed/carepd_npz")

# Change this if you want another recording
RECORDING_NAME = "SUB01_off_walk_3"

# Use None for the full sequence, or e.g. 300 for a quick test
MAX_FRAMES = None


def main():
    output_path = export_recording_to_npz(
        pkl_path=PKL_PATH,
        recording_name=RECORDING_NAME,
        output_dir=OUTPUT_DIR,
        max_frames=MAX_FRAMES,
    )

    print("\nDay 4 export completed:")
    print(output_path)


if __name__ == "__main__":
    main()