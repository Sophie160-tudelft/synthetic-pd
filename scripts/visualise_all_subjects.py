import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import os
import pickle
from pathlib import Path

import torch
import smplx
import trimesh


PKL_PATH = Path("data/raw/BMCLab.pkl")
OUTPUT_ROOT = Path("data/processed/sequences_obj")
MODEL_PATH = "models"


def safe_name(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    model = smplx.create(
        MODEL_PATH,
        model_type="smpl",
        gender="neutral"
    )

    total_recordings = sum(len(data[subject]) for subject in data)
    current_recording = 0

    print(f"Found {len(data)} subjects")
    print(f"Found {total_recordings} recordings")

    for subject in data:
        subject_name = safe_name(subject)

        for recording in data[subject]:
            current_recording += 1
            recording_name = safe_name(recording)

            sample = data[subject][recording]

            pose = torch.tensor(sample["pose"], dtype=torch.float32)
            trans = torch.tensor(sample["trans"], dtype=torch.float32)

            if "beta" in sample:
                betas = torch.tensor(sample["beta"], dtype=torch.float32)
                if betas.ndim == 1:
                    betas = betas.unsqueeze(0)
            else:
                betas = torch.zeros((1, 10), dtype=torch.float32)

            output_dir = OUTPUT_ROOT / subject_name / recording_name
            output_dir.mkdir(parents=True, exist_ok=True)

            num_frames = pose.shape[0]

            print(
                f"\n[{current_recording}/{total_recordings}] "
                f"Exporting {subject_name}/{recording_name} "
                f"({num_frames} frames)"
            )

            metadata_path = output_dir / "metadata.txt"
            with open(metadata_path, "w", encoding="utf-8") as meta:
                meta.write(f"subject={subject}\n")
                meta.write(f"recording={recording}\n")
                meta.write(f"num_frames={num_frames}\n")
                meta.write(f"fps={sample.get('fps', 'unknown')}\n")
                meta.write(f"UPDRS_GAIT={sample.get('UPDRS_GAIT', 'unknown')}\n")
                meta.write(f"medication={sample.get('medication', 'unknown')}\n")
                meta.write(f"other={sample.get('other', 'unknown')}\n")

            with torch.no_grad():
                for i in range(num_frames):
                    output = model(
                        global_orient=pose[i, :3].unsqueeze(0),
                        body_pose=pose[i, 3:].unsqueeze(0),
                        transl=trans[i].unsqueeze(0),
                        betas=betas
                    )

                    vertices = output.vertices.detach().cpu().numpy()[0]
                    mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces)

                    filename = output_dir / f"frame_{i:04d}.obj"
                    mesh.export(filename)

                    if i % 100 == 0:
                        print(f"  exported frame {i}/{num_frames}")

    print("\nDone exporting all sequences.")


if __name__ == "__main__":
    main()