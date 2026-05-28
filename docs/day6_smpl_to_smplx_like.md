# Day 6 — SMPL to SMPL-X-like Motion Bridge

## Goal
Convert the preprocessed CARE-PD SMPL motion representation into a first SMPL-X-like motion file for later BEDLAM/Blender/Unreal testing.

## Input
`data/processed/carepd_preprocessed/SUB01_off_walk_3_30fps_centered.npz`

## Output
`data/processed/bedlam_motion_npz/SUB01_off_walk_3_smplx_like.npz`

## Processing
- Input pose representation: SMPL `(T, 72)`.
- Output pose representation: SMPL-X-like `(T, 165)`.
- Global orientation copied from SMPL.
- Main body pose copied approximately into the SMPL-X body pose section.
- Face, eyes, and hands set to neutral zeros.
- Translation, betas, fps, subject, recording, UPDRS gait score, medication state, and notes preserved.

## Validation
- [ ] Output pose shape is `(T, 165)`.
- [ ] Translation shape is `(T, 3)`.
- [ ] Pose and translation have same frame count.
- [ ] Values are finite.
- [ ] Metadata is preserved.

## Limitation
This is a pragmatic first bridge, not a full anatomical SMPL-to-SMPL-X retargeting. It is sufficient for testing file flow and Blender/Unreal compatibility, but later work may need a more precise SMPL-to-SMPL-X conversion.