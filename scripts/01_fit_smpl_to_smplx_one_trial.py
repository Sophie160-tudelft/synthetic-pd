"""
Fit one BMCLab/CARE-PD SMPL walking sequence to SMPL-X while preserving
the original frame rate and original frame count by default.

This corrected version does NOT downsample to 30 fps unless you explicitly
ask it to using --target-fps.

Default behavior:
    BMCLab 150 fps / 671 frames -> SMPL-X 150 fps / 671 frames

Run from project root:
    python scripts\00_fit_smpl_to_smplx_one_trial_auto_frames_corrected.py

Examples:
    python scripts\00_fit_smpl_to_smplx_one_trial_auto_frames.py --subject SUB01 --trial SUB01_off_walk_1
    python scripts\00_fit_smpl_to_smplx_one_trial_auto_frames.py --subject SUB02 --trial SUB02_off_walk_2

Optional debug/downsample examples:
    python scripts\00_fit_smpl_to_smplx_one_trial_auto_frames.py --target-fps 30
    python scripts\00_fit_smpl_to_smplx_one_trial_auto_frames.py --max-frames 40
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import smplx


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\sopha\synthetic-pd")

DEFAULT_BMCLAB_PKL = PROJECT_ROOT / "data" / "raw" / "BMCLab.pkl"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "smplx_fitted_npz"

DEFAULT_SUBJECT = "SUB01"
DEFAULT_TRIAL = "SUB01_off_walk_1"

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit one BMCLab SMPL walking sequence to SMPL-X. "
            "By default, preserves the original BMCLab FPS and frame count."
        )
    )

    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT,
        help="BMCLab subject ID, e.g. SUB01",
    )

    parser.add_argument(
        "--trial",
        default=DEFAULT_TRIAL,
        help="BMCLab trial/sequence name, e.g. SUB01_off_walk_1",
    )

    parser.add_argument(
        "--bmclab-pkl",
        default=str(DEFAULT_BMCLAB_PKL),
        help="Path to BMCLab.pkl",
    )

    parser.add_argument(
        "--model-root",
        default=str(DEFAULT_MODEL_ROOT),
        help="Path to folder containing SMPL and SMPL-X model folders/files",
    )

    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for fitted SMPL-X NPZ files",
    )

    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help=(
            "Optional target FPS. Default None preserves the original BMCLab FPS. "
            "Use 30 only for quick/debug downsampled fitting."
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit. Keep None for the full original sequence.",
    )

    parser.add_argument(
        "--num-iters",
        type=int,
        default=400,
        help="Number of optimization iterations.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        help="Adam learning rate.",
    )

    return parser.parse_args()


def load_bmclab_sequence(bmclab_pkl, subject, trial):
    with open(bmclab_pkl, "rb") as f:
        carepd = pickle.load(f, encoding="latin1")

    if subject not in carepd:
        raise KeyError(f"Subject not found: {subject}. Available subjects: {list(carepd.keys())[:20]}")

    if trial not in carepd[subject]:
        available = list(carepd[subject].keys())[:20]
        raise KeyError(
            f"Trial not found for {subject}: {trial}\n"
            f"First available trials for this subject: {available}"
        )

    trial_data = carepd[subject][trial]

    pose = trial_data["pose"].astype(np.float32)      # (T, 72)
    trans = trial_data["trans"].astype(np.float32)    # (T, 3)
    betas_smpl = trial_data["beta"].reshape(-1).astype(np.float32)  # (10,)
    fps = int(np.asarray(trial_data["fps"]).item())

    return pose, trans, betas_smpl, fps, trial_data


def maybe_resample(pose, trans, source_fps, target_fps, max_frames):
    """
    Preserves the original sequence if target_fps is None or equal to source_fps.
    Otherwise, keeps every Nth frame using a rounded source_fps / target_fps step.
    """
    if target_fps is None:
        target_fps = source_fps

    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")

    if target_fps > source_fps:
        raise ValueError(
            f"target_fps={target_fps} is higher than source_fps={source_fps}. "
            "This script can preserve or downsample, but it does not upsample."
        )

    if source_fps % target_fps != 0:
        print(
            f"WARNING: source fps {source_fps} is not an integer multiple of target fps {target_fps}. "
            "Using rounded frame step."
        )

    step = max(1, round(source_fps / target_fps))

    pose_fit = pose[::step].copy()
    trans_fit = trans[::step].copy()

    if max_frames is not None:
        pose_fit = pose_fit[:max_frames]
        trans_fit = trans_fit[:max_frames]

    return pose_fit, trans_fit, target_fps, step


def main():
    args = parse_args()

    bmclab_pkl = Path(args.bmclab_pkl)
    model_root = Path(args.model_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject = args.subject
    trial = args.trial

    if not bmclab_pkl.exists():
        raise FileNotFoundError(f"BMCLab.pkl not found:\n{bmclab_pkl}")

    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found:\n{model_root}")

    # ---------------------------------------------------------------------
    # Load CARE-PD/BMCLab SMPL parameters
    # ---------------------------------------------------------------------
    pose, trans, betas_smpl, source_fps, trial_data = load_bmclab_sequence(
        bmclab_pkl=bmclab_pkl,
        subject=subject,
        trial=trial,
    )

    pose_fit, trans_fit, target_fps, step = maybe_resample(
        pose=pose,
        trans=trans,
        source_fps=source_fps,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
    )

    T = pose_fit.shape[0]

    print("Loaded CARE-PD/BMCLab trial")
    print("---------------------------")
    print("subject:", subject)
    print("trial:", trial)
    print("original pose:", pose.shape)
    print("original trans:", trans.shape)
    print("fit pose:", pose_fit.shape)
    print("fit trans:", trans_fit.shape)
    print("betas:", betas_smpl.shape)
    print("fps:", source_fps, "->", target_fps)
    print("frame step:", step)
    print("frames fitted:", T)

    if "UPDRS_GAIT" in trial_data:
        print("UPDRS_GAIT:", trial_data["UPDRS_GAIT"])
    if "medication" in trial_data:
        print("medication:", trial_data["medication"])
    if "other" in trial_data:
        print("freezer group:", trial_data["other"])

    # ---------------------------------------------------------------------
    # Load SMPL and SMPL-X models
    # ---------------------------------------------------------------------
    print("\nLoading SMPL model...")
    smpl_model = smplx.create(
        model_path=str(model_root),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=T,
        num_betas=10,
    ).to(DEVICE)

    print("Loading SMPL-X model...")
    smplx_model = smplx.create(
        model_path=str(model_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        batch_size=T,
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
    ).to(DEVICE)

    print("Models loaded.")

    # ---------------------------------------------------------------------
    # Prepare source SMPL joints
    # ---------------------------------------------------------------------
    pose_t = torch.tensor(pose_fit, dtype=DTYPE, device=DEVICE)
    trans_t = torch.tensor(trans_fit, dtype=DTYPE, device=DEVICE)
    betas_t = torch.tensor(betas_smpl, dtype=DTYPE, device=DEVICE).view(1, 10).repeat(T, 1)

    smpl_global_orient = pose_t[:, 0:3]
    smpl_body_pose = pose_t[:, 3:72]

    with torch.no_grad():
        smpl_out = smpl_model(
            betas=betas_t,
            global_orient=smpl_global_orient,
            body_pose=smpl_body_pose,
            transl=trans_t,
            return_verts=False,
        )
        smpl_joints = smpl_out.joints[:, :22, :].detach()

    print("SMPL source joints:", smpl_joints.shape)

    # ---------------------------------------------------------------------
    # Initialize SMPL-X parameters from CARE-PD SMPL
    # ---------------------------------------------------------------------
    # SMPL-X body_pose has 21 joints = 63 values.
    # Use SMPL joints 1..21 as initialization.
    init_global_orient = pose_fit[:, 0:3].copy()
    init_body_pose = pose_fit[:, 3:66].copy()
    init_transl = trans_fit.copy()
    init_betas = betas_smpl.copy()

    global_orient = torch.nn.Parameter(torch.tensor(init_global_orient, dtype=DTYPE, device=DEVICE))
    body_pose = torch.nn.Parameter(torch.tensor(init_body_pose, dtype=DTYPE, device=DEVICE))
    transl = torch.nn.Parameter(torch.tensor(init_transl, dtype=DTYPE, device=DEVICE))

    # Keep betas fixed. This avoids shape drifting and keeps optimisation focused on pose.
    betas_x = torch.tensor(init_betas, dtype=DTYPE, device=DEVICE).view(1, 10).repeat(T, 1)

    # Neutral face/hands
    jaw_pose = torch.zeros((T, 3), dtype=DTYPE, device=DEVICE)
    leye_pose = torch.zeros((T, 3), dtype=DTYPE, device=DEVICE)
    reye_pose = torch.zeros((T, 3), dtype=DTYPE, device=DEVICE)
    left_hand_pose = torch.zeros((T, 45), dtype=DTYPE, device=DEVICE)
    right_hand_pose = torch.zeros((T, 45), dtype=DTYPE, device=DEVICE)
    expression = torch.zeros((T, 10), dtype=DTYPE, device=DEVICE)

    # ---------------------------------------------------------------------
    # Optimise SMPL-X to match SMPL joints
    # ---------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        [global_orient, body_pose, transl],
        lr=args.lr,
    )

    init_body_pose_t = torch.tensor(init_body_pose, dtype=DTYPE, device=DEVICE)
    init_global_t = torch.tensor(init_global_orient, dtype=DTYPE, device=DEVICE)

    print("\nOptimising SMPL-X pose...")
    for it in range(args.num_iters):
        optimizer.zero_grad()

        smplx_out = smplx_model(
            betas=betas_x,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=expression,
            return_verts=False,
        )

        smplx_joints = smplx_out.joints[:, :22, :]

        joint_loss = torch.mean((smplx_joints - smpl_joints) ** 2)

        # Keep solution close to CARE-PD pose initialization
        pose_reg = torch.mean((body_pose - init_body_pose_t) ** 2)
        root_reg = torch.mean((global_orient - init_global_t) ** 2)

        # Small temporal smoothness to avoid jitter
        if T > 1:
            smooth_loss = torch.mean((body_pose[1:] - body_pose[:-1]) ** 2)
        else:
            smooth_loss = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

        loss = joint_loss + 0.001 * pose_reg + 0.001 * root_reg + 0.0001 * smooth_loss

        loss.backward()
        optimizer.step()

        if it % 50 == 0 or it == args.num_iters - 1:
            print(
                f"iter {it:04d} | "
                f"loss={loss.item():.8f} | "
                f"joint={joint_loss.item():.8f} | "
                f"pose_reg={pose_reg.item():.8f}"
            )

    # ---------------------------------------------------------------------
    # Save fitted SMPL-X animation
    # ---------------------------------------------------------------------
    global_np = global_orient.detach().cpu().numpy().astype(np.float32)
    body_np = body_pose.detach().cpu().numpy().astype(np.float32)
    trans_np = transl.detach().cpu().numpy().astype(np.float32)

    jaw_np = np.zeros((T, 3), dtype=np.float32)
    leye_np = np.zeros((T, 3), dtype=np.float32)
    reye_np = np.zeros((T, 3), dtype=np.float32)
    left_hand_np = np.zeros((T, 45), dtype=np.float32)
    right_hand_np = np.zeros((T, 45), dtype=np.float32)
    expression_np = np.zeros((T, 10), dtype=np.float32)

    poses_smplx = np.concatenate(
        [
            global_np,
            body_np,
            jaw_np,
            leye_np,
            reye_np,
            left_hand_np,
            right_hand_np,
        ],
        axis=1,
    ).astype(np.float32)

    assert poses_smplx.shape == (T, 165), poses_smplx.shape

    out_path = out_dir / f"{trial}_smplx_fitted_{target_fps}fps_{T}frames.npz"

    np.savez(
        out_path,
        poses=poses_smplx,
        trans=trans_np,
        transl=trans_np,
        betas=init_betas.astype(np.float32),
        gender=np.array("neutral"),
        mocap_framerate=np.array(target_fps),
        fps=np.array(target_fps),

        global_orient=global_np,
        body_pose=body_np,
        jaw_pose=jaw_np,
        leye_pose=leye_np,
        reye_pose=reye_np,
        left_hand_pose=left_hand_np,
        right_hand_pose=right_hand_np,
        expression=expression_np,

        source_subject=np.array(subject),
        source_trial=np.array(trial),
        source_fps=np.array(source_fps),
        source_num_frames=np.array(pose.shape[0]),
        frame_step=np.array(step),
        target_fps=np.array(target_fps),
        max_frames=np.array(args.max_frames if args.max_frames is not None else -1),
    )

    print("\nSaved fitted SMPL-X file:")
    print(out_path)
    print("poses:", poses_smplx.shape)
    print("trans:", trans_np.shape)
    print("betas:", init_betas.shape)


if __name__ == "__main__":
    main()
