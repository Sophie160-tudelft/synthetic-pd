from pathlib import Path
import pickle
import numpy as np


def load_bmclab(pkl_path: str | Path):
    pkl_path = Path(pkl_path)

    if not pkl_path.exists():
        raise FileNotFoundError(f"Could not find BMCLab file: {pkl_path}")

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def find_recording(data, recording_name: str):
    for subject_id, subject_data in data.items():
        if not isinstance(subject_data, dict):
            continue

        for key, value in subject_data.items():
            if key == recording_name:
                return subject_id, key, value

    raise ValueError(f"Recording not found: {recording_name}")


def export_recording_to_npz(
    pkl_path: str | Path,
    recording_name: str,
    output_dir: str | Path,
    max_frames: int | None = None,
):
    data = load_bmclab(pkl_path)
    subject_id, found_name, rec = find_recording(data, recording_name)

    pose = np.asarray(rec["pose"], dtype=np.float32)
    trans = np.asarray(rec["trans"], dtype=np.float32)
    beta = np.asarray(rec["beta"], dtype=np.float32)

    fps = int(rec.get("fps", 150))
    updrs_gait = rec.get("UPDRS_GAIT", -1)
    med = rec.get("med", "unknown")
    if med == "unknown":
        lower_name = found_name.lower()

        if "_off_" in lower_name:
            med = "off"
        elif "_on_" in lower_name:
            med = "on"
    other = rec.get("other", "unknown")

    if max_frames is not None:
        pose = pose[:max_frames]
        trans = trans[:max_frames]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{found_name}.npz"

    np.savez_compressed(
    output_path,

    # Original-style names
    pose=pose,
    beta=beta,

    # Standardized names for the rest of your pipeline
    poses=pose,
    betas=beta,

    trans=trans,
    fps=fps,
    subject=str(subject_id),
    recording=str(found_name),
    updrs_gait=int(updrs_gait),
    med=str(med),
    med_state=str(med),
    other=str(other),
)

    print("Saved SMPL parameter sequence:")
    print("  output:", output_path)
    print("  pose:", pose.shape)
    print("  trans:", trans.shape)
    print("  beta:", beta.shape)

    return output_path