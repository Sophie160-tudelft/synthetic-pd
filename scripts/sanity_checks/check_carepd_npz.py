from pathlib import Path
import numpy as np


NPZ_PATH = Path("data/processed/carepd_npz/SUB01_off_walk_3.npz")


def scalar_to_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Could not find: {NPZ_PATH}")

    data = np.load(NPZ_PATH, allow_pickle=True)

    print(f"Checking: {NPZ_PATH}\n")

    print("Available keys:")
    for key in data.files:
        value = data[key]
        if value.shape == ():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, value={value.item()}")
        else:
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    poses = data["poses"] if "poses" in data.files else data["pose"]
    trans = data["trans"]
    betas = data["betas"] if "betas" in data.files else data["beta"]
    fps = float(data["fps"])

    print("\nMetadata:")
    for key in ["subject", "recording", "updrs_gait", "med", "med_state", "other"]:
        if key in data.files:
            print(f"  {key}: {scalar_to_str(data[key])}")

    print("\nSanity checks:")
    print("  poses shape OK:", poses.ndim == 2 and poses.shape[1] == 72)
    print("  trans shape OK:", trans.ndim == 2 and trans.shape[1] == 3)
    print("  same frame count:", poses.shape[0] == trans.shape[0])
    print("  betas shape OK:", betas.shape in [(10,), (1, 10)])
    print("  finite poses:", np.isfinite(poses).all())
    print("  finite trans:", np.isfinite(trans).all())
    print("  finite betas:", np.isfinite(betas).all())

    print("\nMotion summary:")
    print(f"  frames: {poses.shape[0]}")
    print(f"  fps: {fps}")
    print(f"  duration: {poses.shape[0] / fps:.2f} sec")
    print(f"  trans first: {trans[0]}")
    print(f"  trans min: {trans.min(axis=0)}")
    print(f"  trans max: {trans.max(axis=0)}")


if __name__ == "__main__":
    main()