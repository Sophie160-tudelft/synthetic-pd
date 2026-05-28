# Day 4 — CARE-PD Recording Export

## Goal
Export one clean CARE-PD/BMCLab recording from `BMCLab.pkl` into a reusable `.npz` format.

## Input
- Source file: `data/raw/BMCLab.pkl`
- Selected recording: `SUB01_off_walk_3`
- Subject: `SUB01`
- Medication state: `off`
- UPDRS gait score: `2`
- Notes: `freezers`

## Output
- Exported file: `data/processed/carepd_npz/SUB01_off_walk_3.npz`

## Exported fields
- `pose` / `poses`: SMPL pose parameters, shape `(605, 72)`
- `trans`: root translation, shape `(605, 3)`
- `beta` / `betas`: SMPL body shape coefficients, shape `(1, 10)`
- `fps`: original recording frame rate, `150`
- `subject`: subject identifier, `SUB01`
- `recording`: recording identifier, `SUB01_off_walk_3`
- `updrs_gait`: clinical gait score, `2`
- `med` / `med_state`: medication state, `off`
- `other`: additional note, `freezers`

## Validation
- [x] Pose shape is `(T, 72)`.
- [x] Translation shape is `(T, 3)`.
- [x] Pose and translation have the same number of frames.
- [x] FPS is stored.
- [x] Subject, recording, medication state, UPDRS gait score, and notes are stored.

## Notes
This `.npz` file is the clean intermediate representation used before converting the CARE-PD SMPL motion to a BEDLAM/SMPL-X-compatible motion format. It is still an SMPL file, not SMPL-X.