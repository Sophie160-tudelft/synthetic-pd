from pathlib import Path
import numpy as np


NPZ_PATH = Path("data/processed/carepd_preprocessed/SUB01_off_walk_3_30fps_centered.npz")


def main():
    data = np.load(NPZ_PATH, allow_pickle=True)

    poses = data["poses"]
    trans = data["trans"]
    fps = float(data["fps"])

    print(f"Checking: {NPZ_PATH}")
    print(f"poses: {poses.shape}")
    print(f"trans: {trans.shape}")
    print(f"fps: {fps}")
    print(f"duration: {poses.shape[0] / fps:.2f} sec")

    print("\nSanity checks:")
    print("poses shape OK:", poses.ndim == 2 and poses.shape[1] == 72)
    print("trans shape OK:", trans.ndim == 2 and trans.shape[1] == 3)
    print("same frame count:", poses.shape[0] == trans.shape[0])
    print("finite poses:", np.isfinite(poses).all())
    print("finite trans:", np.isfinite(trans).all())
    print("starts near xy origin:", np.allclose(trans[0, :2], 0.0, atol=1e-5))

    print("\nTranslation:")
    print("first:", trans[0])
    print("min:", trans.min(axis=0))
    print("max:", trans.max(axis=0))


if __name__ == "__main__":
    main()