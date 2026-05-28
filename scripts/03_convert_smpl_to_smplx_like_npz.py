from pathlib import Path
import numpy as np


INPUT_NPZ = Path("data/processed/carepd_preprocessed/SUB01_off_walk_3_30fps_centered.npz")
OUTPUT_DIR = Path("data/processed/bedlam_motion_npz")


def smpl72_to_smplx165(smpl_poses: np.ndarray) -> np.ndarray:
    """
    Approximate conversion from SMPL 72D pose to SMPL-X-style 165D pose.

    SMPL:
      24 joints × 3 axis-angle = 72

    Common SMPL-X 165D layout:
      global_orient: 3
      body_pose:     21 joints × 3 = 63
      jaw_pose:      3
      leye_pose:     3
      reye_pose:     3
      left_hand:     45
      right_hand:    45
      total:         165

    This first bridge keeps the main body motion and sets
    hands/face/eyes to neutral.
    """
    if smpl_poses.ndim != 2 or smpl_poses.shape[1] != 72:
        raise ValueError(f"Expected SMPL poses with shape (T, 72), got {smpl_poses.shape}")

    T = smpl_poses.shape[0]
    smplx_poses = np.zeros((T, 165), dtype=np.float32)

    # Global/root orientation
    smplx_poses[:, 0:3] = smpl_poses[:, 0:3]

    # Use first 21 SMPL body joints after root.
    # SMPL has 23 body joints after root = 69 dims.
    # SMPL-X body pose commonly uses 21 joints = 63 dims.
    smplx_poses[:, 3:66] = smpl_poses[:, 3:66]

    return smplx_poses


def scalar_to_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def main():
    if not INPUT_NPZ.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_NPZ}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(INPUT_NPZ, allow_pickle=True)

    smpl_poses = data["poses"].astype(np.float32)
    trans = data["trans"].astype(np.float32)
    betas = data["betas"].astype(np.float32)
    fps = float(data["fps"])

    recording = scalar_to_str(data["recording"])
    subject = scalar_to_str(data["subject"])

    smplx_poses = smpl72_to_smplx165(smpl_poses)

    output_path = OUTPUT_DIR / f"{recording}_smplx_like.npz"

    np.savez_compressed(
        output_path,
        poses=smplx_poses,
        trans=trans,
        betas=betas,
        gender="neutral",
        mocap_frame_rate=np.array(fps, dtype=np.float32),
        fps=np.array(fps, dtype=np.float32),
        subject=subject,
        recording=recording,
        updrs_gait=data["updrs_gait"],
        med=data["med"],
        med_state=data["med_state"],
        other=data["other"],
        source_format="CARE-PD SMPL 72D",
        target_format="SMPL-X-like 165D",
        conversion_note="SMPL body motion mapped approximately to SMPL-X body pose; face, eyes, and hands set to zero.",
    )

    print("Saved SMPL-X-like motion:")
    print(f"  output:    {output_path}")
    print(f"  subject:   {subject}")
    print(f"  recording: {recording}")
    print(f"  poses:     {smplx_poses.shape}")
    print(f"  trans:     {trans.shape}")
    print(f"  betas:     {betas.shape}")
    print(f"  fps:       {fps}")


if __name__ == "__main__":
    main()