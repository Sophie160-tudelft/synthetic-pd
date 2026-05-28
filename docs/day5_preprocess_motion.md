# Day 5 — CARE-PD Motion Preprocessing

## Goal
Prepare the exported CARE-PD SMPL recording for later BEDLAM/SMPL-X conversion.

## Input
`data/processed/carepd_npz/SUB01_off_walk_3.npz`

## Output
`data/processed/carepd_preprocessed/SUB01_off_walk_3_30fps_centered.npz`

## Processing steps
- Downsampled the original CARE-PD sequence from 150 fps to 30 fps.
- Preserved the original SMPL pose representation `(T, 72)`.
- Centered the root translation so the first frame starts at approximately `(x=0, y=0)`.
- Stored the source frame indices for traceability.
- Preserved subject, recording, UPDRS gait score, medication state, and notes.

## Validation result
- `poses`: `(121, 72)`
- `trans`: `(121, 3)`
- `fps`: `30.0`
- duration: `4.03 sec`
- pose shape valid: yes
- translation shape valid: yes
- same frame count: yes
- finite values: yes
- starts near horizontal origin: yes

## Translation summary
- first translation: `[0.0, 0.0, -0.33003345]`
- min translation: `[0.0, -0.02567518, -0.39274138]`
- max translation: `[3.0400107, 0.00490987, -0.32994124]`

## Notes
The preprocessed file is still SMPL, not SMPL-X. The next step is to convert the 72D SMPL pose representation into a BEDLAM/SMPL-X-like 165D representation.