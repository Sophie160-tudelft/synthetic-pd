from pathlib import Path
import numpy as np


NPZ_PATH = Path("data/processed/bedlam_motion_npz/SUB01_off_walk_3_smplx_like.npz")


def main():
    data = np.load(NPZ_PATH, allow_pickle=True)

    poses = data["poses"]
    trans = data["trans"]
    betas = data["betas"]
    fps = float(data["fps"])

    print(f"Checking: {NPZ_PATH}")
    print(f"poses: {poses.shape}")
    print(f"trans: {trans.shape}")
    print(f"betas: {betas.shape}")
    print(f"fps: {fps}")
    print(f"duration: {poses.shape[0] / fps:.2f} sec")

    print("\nSanity checks:")
    print("poses shape OK:", poses.ndim == 2 and poses.shape[1] == 165)
    print("trans shape OK:", trans.ndim == 2 and trans.shape[1] == 3)
    print("same frame count:", poses.shape[0] == trans.shape[0])
    print("betas shape OK:", betas.shape in [(10,), (1, 10)])
    print("finite poses:", np.isfinite(poses).all())
    print("finite trans:", np.isfinite(trans).all())
    print("finite betas:", np.isfinite(betas).all())

    print("\nMetadata:")
    for key in ["subject", "recording", "updrs_gait", "med_state", "source_format", "target_format"]:
        if key in data.files:
            value = data[key]
            print(f"{key}: {value.item() if value.shape == () else value}")


if __name__ == "__main__":
    main()