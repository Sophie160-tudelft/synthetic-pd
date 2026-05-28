from pathlib import Path
import numpy as np


INPUT_NPZ = Path("data/processed/carepd_npz/SUB01_off_walk_3.npz")
OUTPUT_DIR = Path("data/processed/carepd_preprocessed")

TARGET_FPS = 30.0


def downsample_indices(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if target_fps >= source_fps:
        return np.arange(num_frames)

    duration = num_frames / source_fps
    target_num_frames = int(round(duration * target_fps))

    times = np.arange(target_num_frames) / target_fps
    indices = np.round(times * source_fps).astype(int)
    indices = np.clip(indices, 0, num_frames - 1)

    return indices


def center_translation_xy(trans: np.ndarray) -> np.ndarray:
    """
    Center the sequence so the first frame starts at x=0, y=0.
    Keep the vertical axis unchanged for now.
    """
    trans = trans.copy().astype(np.float32)
    trans[:, 0] -= trans[0, 0]
    trans[:, 1] -= trans[0, 1]
    return trans


def scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def main():
    if not INPUT_NPZ.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_NPZ}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(INPUT_NPZ, allow_pickle=True)

    poses = data["poses"].astype(np.float32)
    trans = data["trans"].astype(np.float32)
    betas = data["betas"].astype(np.float32)

    source_fps = float(data["fps"])
    recording = scalar_str(data["recording"])
    subject = scalar_str(data["subject"])

    indices = downsample_indices(
        num_frames=poses.shape[0],
        source_fps=source_fps,
        target_fps=TARGET_FPS,
    )

    poses_ds = poses[indices]
    trans_ds = trans[indices]
    trans_centered = center_translation_xy(trans_ds)

    output_path = OUTPUT_DIR / f"{recording}_30fps_centered.npz"

    np.savez_compressed(
        output_path,
        pose=poses_ds,
        poses=poses_ds,
        trans=trans_centered,
        beta=betas,
        betas=betas,
        fps=np.array(TARGET_FPS, dtype=np.float32),
        source_fps=np.array(source_fps, dtype=np.float32),
        source_indices=indices,
        subject=subject,
        recording=recording,
        updrs_gait=data["updrs_gait"],
        med=data["med"],
        med_state=data["med_state"],
        other=data["other"],
        preprocessing="downsampled_to_30fps_centered_xy",
    )

    print("Saved preprocessed CARE-PD motion:")
    print(f"  output:       {output_path}")
    print(f"  subject:      {subject}")
    print(f"  recording:    {recording}")
    print(f"  source fps:   {source_fps}")
    print(f"  target fps:   {TARGET_FPS}")
    print(f"  source frames:{poses.shape[0]}")
    print(f"  output frames:{poses_ds.shape[0]}")
    print(f"  duration:     {poses_ds.shape[0] / TARGET_FPS:.2f} sec")
    print(f"  poses:        {poses_ds.shape}")
    print(f"  trans:        {trans_centered.shape}")
    print(f"  trans min:    {trans_centered.min(axis=0)}")
    print(f"  trans max:    {trans_centered.max(axis=0)}")


if __name__ == "__main__":
    main()