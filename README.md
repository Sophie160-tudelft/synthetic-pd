# Synthetic PD Gait Pipeline README

This README explains the order in which the repository scripts are used, what each script does, how to run it, and where outputs are saved. The workflow processes BMCLab/CARE-PD motion data, prepares it for Blender and Unreal Engine rendering, converts rendered frames to WHAM-ready videos, and computes the evaluation metrics used in the thesis.

Run all commands from the repository root:

```powershell
cd C:\Users\sopha\synthetic-pd
```

## Expected repository structure

```text
synthetic-pd/
├── data/
│   ├── raw/BMCLab.pkl
│   ├── processed/smplx_fitted_npz/
│   ├── videos/
│   └── wham_outputs/
├── models/
│   ├── smpl/
│   └── smplx/
├── results/
├── scripts/
└── WHAM.ipynb
```
The data, models and results folders can be downloaded from this URL: https://zenodo.org/records/20923762?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImNkOGY0YjM2LTJkYjMtNGZhNy04YWUxLTFlNzZlNTM3ZGIyMCIsImRhdGEiOnt9LCJyYW5kb20iOiJmZjNhMmZmMjk0MmE3MjM5MWM0NWZmNzhhZTY5NGIwNSJ9.Wcw4-XDXQ3-DR5XLNVIcGRJzNpowMAEDicKuWbn5IkFLrWKWAe-INwnzt9rRzoRX-NLCOVm23jUeoD7EzSgNHg

## Python environment and dependencies

The local repository scripts can be run from one shared Python environment. In earlier development, separate environments were sometimes used for SMPL-X fitting, video conversion, and metric computation, but for reproducibility it is clearer to use one combined environment for scripts `00` through `10`.

The recommended local environment is called:

```text
synthetic-pd
```

There are two supported ways to create the environment: with Conda or with the standard Python `venv` module.

### Option 1: create the environment with Conda

If Conda or Anaconda is installed, the environment can be created from the supplied Conda environment file:

```powershell
conda env create -f environment_synthetic_pd.yml
conda activate synthetic-pd
```

### Option 2: create the environment without Conda

If Conda is not installed, the environment can be created with the standard Python `venv` module. This creates a local virtual environment inside the repository folder. The environment folder can be named `synthetic-pd` so that it matches the recommended environment name.

From the repository root, run:

```powershell
python -m venv synthetic-pd
.\synthetic-pd\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_synthetic_pd.txt
```

After activation, the terminal should show the active environment at the start of the command line, for example:

```text
(synthetic-pd) PS C:\Users\sopha\synthetic-pd>
```

The environment can be tested with:

```powershell
python -c "import numpy, scipy, pandas, matplotlib, torch, smplx, joblib; print('Python environment works')"
```

If the command prints `Python environment works`, the local Python environment is ready to run the repository scripts.

This environment contains the Python packages needed for the local pipeline, including `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, `smplx`, and `joblib`. These packages are needed because the pipeline includes SMPL-to-SMPL-X fitting, coordinate corrections, validation of body-model outputs, WHAM-vs-BMCLab metric computation, and result plotting.

The only additional system dependency is FFmpeg, which is required by `04_make_wham_rgb_video.py` to convert Unreal Engine PNG frame sequences into WHAM-ready MP4 videos. On Windows, FFmpeg can be installed with:

```powershell
winget install Gyan.FFmpeg
```

After installing FFmpeg, restart PowerShell or VS Code so that the `ffmpeg` and `ffprobe` commands are available from the terminal.

The WHAM notebook is not run inside this local environment. `WHAM.ipynb` is executed separately on Kaggle, because WHAM has its own dependencies, model files, and runtime requirements. The local environment is only used before WHAM to prepare the videos and after WHAM to analyze the downloaded `.pkl` outputs.



## Workflow order

1. `00_select_bmclab_sequences.py` — select and document the BMCLab walking sequences.
2. `01_fit_smpl_to_smplx_one_trial.py` — fit one BMCLab SMPL sequence to SMPL-X.
3. `02_final_smplx_auto_frames.py` — apply the final render-coordinate correction.
4. `03_make_blender_fullframe_import.py` — create the special Blender full-frame import copy.
5. Manual Blender step — import the `.npz` file and export an FBX animation.
6. Manual Unreal step — import the FBX file, set up the scene, and render PNG frames.
7. `04_make_wham_rgb_video.py` — convert PNG frames to WHAM-ready MP4 videos.
8. `WHAM.ipynb` — run WHAM externally, for example in a Kaggle notebook, on the generated MP4 videos.
9. `05_validate_render_input_motion.py` — validate the SMPL-X render-input motion against BMCLab.
10. `06_validate_root_rotation_effect.py` — validate the render-coordinate correction.
11. `07_compare_bmclab_wham_all_experiments.py` — compute WHAM-vs-BMCLab metrics.
12. `08_analyze_experiment_degradation_relative_to_baselines.py` — analyze full-matrix degradation relative to E0 and E1.
13. `09_analyze_reduced_matrix_replication_comparisons.py` — analyze the reduced matrix for additional sequences.
14. `10_make_results_tables_figures.py` — generate thesis-ready result tables and figures.

## 0. Select BMCLab walking sequences

**Script:** `00_select_bmclab_sequences.py`

**Purpose:** Selects the BMCLab/CARE-PD walking sequences used in the thesis and writes a metadata table. The selected sequences are:

```text
SUB01 / SUB01_off_walk_1
SUB02 / SUB02_off_walk_2
SUB08 / SUB08_off_walk_4
SUB17 / SUB17_off_walk_5
```

**Run:**

```powershell
python scripts\00_select_bmclab_sequences.py
```

Optional explicit input:

```powershell
python scripts\00_select_bmclab_sequences.py --bmclab-pkl data\raw\BMCLab.pkl
```

**Output:**

```text
results\selected_first_four_bmclab_sequences.csv
```

## 1. Fit BMCLab SMPL motion to SMPL-X

**Script:** `01_fit_smpl_to_smplx_one_trial.py`

**Purpose:** Converts one original BMCLab SMPL walking sequence into SMPL-X format. By default, the script preserves the original BMCLab frame rate and frame count. Do not use `--target-fps 30` for the final thesis pipeline; that option is only for debugging or quick tests.

**Run examples:**

```powershell
python scripts\01_fit_smpl_to_smplx_one_trial.py --subject SUB01 --trial SUB01_off_walk_1
python scripts\01_fit_smpl_to_smplx_one_trial.py --subject SUB02 --trial SUB02_off_walk_2
python scripts\01_fit_smpl_to_smplx_one_trial.py --subject SUB08 --trial SUB08_off_walk_4
python scripts\01_fit_smpl_to_smplx_one_trial.py --subject SUB17 --trial SUB17_off_walk_5
```

**Output:**

```text
data\processed\smplx_fitted_npz\<TRIAL>_smplx_fitted_<FPS>fps_<N>frames.npz
```

Example:

```text
data\processed\smplx_fitted_npz\SUB01_off_walk_1_smplx_fitted_150fps_671frames.npz
```

## 2. Create the final render-ready SMPL-X file

**Script:** `02_final_smplx_auto_frames.py`

**Purpose:** Automatically finds the fitted SMPL-X file and applies the render-coordinate correction. The correction applies `rootRot_X90` and `neckHeadHalf`. The script infers FPS and frame count from the input file.

**Run examples:**

```powershell
python scripts\02_final_smplx_auto_frames.py --trial SUB01_off_walk_1
python scripts\02_final_smplx_auto_frames.py --trial SUB02_off_walk_2
python scripts\02_final_smplx_auto_frames.py --trial SUB08_off_walk_4
python scripts\02_final_smplx_auto_frames.py --trial SUB17_off_walk_5
```

**Output:**

```text
data\processed\smplx_fitted_npz\smpl_to_smplx\<TRIAL>_smplx_fitted_<FPS>fps_<N>frames_rootRot_X90_neckHeadHalf.npz
```

Example:

```text
data\processed\smplx_fitted_npz\smpl_to_smplx\SUB01_off_walk_1_smplx_fitted_150fps_671frames_rootRot_X90_neckHeadHalf.npz
```

## 3. Create a Blender full-frame import copy

**Script:** `03_make_blender_fullframe_import.py`

**Purpose:** Creates a Blender-import version of the corrected SMPL-X file. This is needed because the SMPL-X Blender add-on may resample high-frame-rate motion to 30 fps. The script keeps all poses but stores the metadata as 30 fps, so the add-on imports every pose as one Blender frame.

This file is only used for Blender import and FBX export. It should not be used as the evaluation reference.

**Run examples:**

```powershell
python scripts\03_make_blender_fullframe_import.py --trial SUB01_off_walk_1
python scripts\03_make_blender_fullframe_import.py --trial SUB02_off_walk_2
python scripts\03_make_blender_fullframe_import.py --trial SUB08_off_walk_4
python scripts\03_make_blender_fullframe_import.py --trial SUB17_off_walk_5
```

**Output:**

```text
data\processed\smplx_fitted_npz\smpl_to_smplx\<TRIAL>_smplx_fitted_<FPS>fps_<N>frames_rootRot_X90_neckHeadHalf_blenderFullFrames_importAs30fps.npz
```

**Manual next step:** Import this `.npz` file into Blender using the SMPL-X add-on, then export the animation as an FBX file. The FBX file is then imported into Unreal Engine for rendering.

## 4. Convert Unreal PNG renders to WHAM-ready RGB videos

**Script:** `04_make_wham_rgb_video.py`

**Purpose:** Converts Unreal Engine PNG render folders into MP4 videos for WHAM. The script sorts PNG files by frame number, checks for missing frames, and writes a 150 fps RGB MP4 video.

Expected input structure:

```text
D:\Experiment_renders\<SUBJECT>\E0\*.png
D:\Experiment_renders\<SUBJECT>\E1\*.png
D:\Experiment_renders\<SUBJECT>\E2\*.png
...
```

**Run examples:**

```powershell
python scripts\04_make_wham_rgb_video.py --subject SUB01 --trial SUB01_off_walk_1
python scripts\04_make_wham_rgb_video.py --subject SUB02 --trial SUB02_off_walk_2
python scripts\04_make_wham_rgb_video.py --subject SUB08 --trial SUB08_off_walk_4
python scripts\04_make_wham_rgb_video.py --subject SUB17 --trial SUB17_off_walk_5
```

Optional explicit paths:

```powershell
python scripts\04_make_wham_rgb_video.py ^
    --subject SUB01 ^
    --trial SUB01_off_walk_1 ^
    --input_root D:\Experiment_renders ^
    --output_root data\videos ^
    --fps 150
```

**Output:**

```text
data\videos\<SUBJECT>\<EXPERIMENT>\<SUBJECT>_<EXPERIMENT>_wham_rgb_150fps.mp4
```

Example:

```text
data\videos\SUB01\E0\SUB01_E0_wham_rgb_150fps.mp4
```

## 5. Run WHAM pose recovery in the Kaggle notebook

**Notebook:** `WHAM.ipynb`

**Purpose:** Runs WHAM on the subject-specific MP4 experiment videos created by script 04 and saves one WHAM `.pkl` output per experiment. This step is run externally, for example on Kaggle, because WHAM requires its own environment, dependencies, and model files.

### 5.1 Upload the experiment videos as Kaggle datasets

Upload the MP4 videos as independent Kaggle datasets, one dataset per subject. For the thesis subjects, use subject-specific dataset names:

```text
SUB01
SUB02
SUB08
SUB17
```

If additional subjects or experiments are processed later, create a new independent Kaggle dataset for each new subject using the same naming logic, for example `SUB03`, `SUB12`, or `SUB20`. Do not combine multiple subjects in one Kaggle dataset for the WHAM run, because the notebook infers the subject ID from the uploaded dataset contents and expects one subject per run.

The uploaded dataset should contain the experiment videos in the same folder structure as the subject video output. In Kaggle, the dataset browser may show a nested folder with the same uploaded folder name, as in the example below:

```text
DATASETS/
└── SUB02_Experiments/
    └── SUB02_Experiments/
        ├── E0/
        │   └── SUB02_E0_wham_rgb_150fps.mp4
        ├── E1/
        │   └── SUB02_E1_wham_rgb_150fps.mp4
        ├── E2/
        │   └── SUB02_E2_wham_rgb_150fps.mp4
        ├── E3/
        │   └── SUB02_E3_wham_rgb_150fps.mp4
        ├── E4/
        │   └── SUB02_E4_wham_rgb_150fps.mp4
        └── E5/
            └── SUB02_E5_wham_rgb_150fps.mp4
```

For the full SUB01 matrix, the same structure is used but with all available experiment folders, for example `E0` through `E13`. For the reduced replication subjects, the structure normally contains `E0` through `E5`.

A zipped upload is also accepted by the notebook, as long as the zip contains the subject folder and experiment subfolders:

```text
SUB02.zip
└── SUB02/
    ├── E0/SUB02_E0_wham_rgb_150fps.mp4
    ├── E1/SUB02_E1_wham_rgb_150fps.mp4
    └── ...
```

### 5.2 Configure and run `WHAM.ipynb`

In Kaggle, add the relevant subject dataset to the notebook, then update the input path in the first cell if needed:

```python
KAGGLE_INPUT_ROOT = Path("/kaggle/input/<your-subject-dataset>")
```

The notebook automatically searches below `KAGGLE_INPUT_ROOT`, infers the subject ID from names such as `SUB01`, `SUB02`, `SUB08`, or `SUB17`, finds all `.mp4` or `.mov` videos, sorts them by experiment number, runs WHAM for each experiment, and copies each `wham_output.pkl` to a stable subject/experiment output folder.

Expected Kaggle output:

```text
/kaggle/working/wham_pkls/<SUBJECT>/<EXPERIMENT>/<SUBJECT>_<EXPERIMENT>_wham_output.pkl
/kaggle/working/wham_pkls/<SUBJECT>/<SUBJECT>_wham_pkl_summary.csv
/kaggle/working/<SUBJECT>_wham_pkls.zip
```

Example:

```text
/kaggle/working/wham_pkls/SUB02/E0/SUB02_E0_wham_output.pkl
/kaggle/working/wham_pkls/SUB02/E1/SUB02_E1_wham_output.pkl
/kaggle/working/SUB02_wham_pkls.zip
```

### 5.3 Copy the WHAM outputs back into the repository

Download the `<SUBJECT>_wham_pkls.zip` file from Kaggle and extract it into the local repository under:

```text
data\wham_outputs\<SUBJECT>\
```

After extraction, the expected local structure for the metric scripts is:

```text
data\wham_outputs\<SUBJECT>\E0\<SUBJECT>_E0_wham_output.pkl
data\wham_outputs\<SUBJECT>\E1\<SUBJECT>_E1_wham_output.pkl
data\wham_outputs\<SUBJECT>\E2\<SUBJECT>_E2_wham_output.pkl
...
```

An equivalent output structure can also be used, as long as script 07 receives the correct `--wham-pkl-root` path and can find the experiment-specific WHAM `.pkl` files.

## 6. Validate the render-input motion

**Script:** `05_validate_render_input_motion.py`

**Purpose:** Checks whether the final SMPL-X file used for rendering still preserves the original BMCLab SMPL motion before WHAM is involved. It compares the original BMCLab SMPL motion with the final SMPL-X render-input motion.

**Run examples:**

```powershell
python scripts\05_validate_render_input_motion.py --subject SUB01 --trial SUB01_off_walk_1
python scripts\05_validate_render_input_motion.py --subject SUB02 --trial SUB02_off_walk_2
python scripts\05_validate_render_input_motion.py --subject SUB08 --trial SUB08_off_walk_4
python scripts\05_validate_render_input_motion.py --subject SUB17 --trial SUB17_off_walk_5
```

**Output:**

```text
results\render_input_validation\<SUBJECT>_<TRIAL>\
```

## 7. Validate the root-rotation correction

**Script:** `06_validate_root_rotation_effect.py`

**Purpose:** Checks whether the render-coordinate correction changes only the coordinate frame and head posture, or whether it distorts clinically relevant gait motion. It compares the raw fitted SMPL-X file with the corrected render-ready SMPL-X file.

**Run examples:**

```powershell
python scripts\06_validate_root_rotation_effect.py --trial SUB01_off_walk_1
python scripts\06_validate_root_rotation_effect.py --trial SUB02_off_walk_2
python scripts\06_validate_root_rotation_effect.py --trial SUB08_off_walk_4
python scripts\06_validate_root_rotation_effect.py --trial SUB17_off_walk_5
```

**Output:**

```text
results\render_input_validation\root_rotation_effect_<TRIAL>\
```

## 8. Compare BMCLab reference motion with WHAM outputs

**Script:** `07_compare_bmclab_wham_all_experiments.py`

**Purpose:** Main batch metric script. It compares the original BMCLab SMPL motion with WHAM output for all rendered experiments of one subject. It writes both summary metrics and per-frame metrics.

Main metrics include PA-MPJPE, root-aligned MPJPE, knee ROM error, contact timing error, cadence error, foot-clearance error, step-length error, stride-length error, walking-speed errors, and pelvis-trajectory error.

**Run examples:**

```powershell
python scripts\07_compare_bmclab_wham_all_experiments.py ^
    --subject SUB01 ^
    --trial SUB01_off_walk_1 ^
    --wham-pkl-root data\wham_outputs ^
    --fps-override 150

python scripts\07_compare_bmclab_wham_all_experiments.py ^
    --subject SUB02 ^
    --trial SUB02_off_walk_2 ^
    --wham-pkl-root data\wham_outputs ^
    --fps-override 150
```

Repeat for `SUB08 / SUB08_off_walk_4` and `SUB17 / SUB17_off_walk_5`.

**Output:**

```text
results\clinically_relevant_metrics_v1\batch_wham_experiments\<SUBJECT>\<TRIAL>\
```

Important output files:

```text
<SUBJECT>_<TRIAL>_all_experiments_metric_summary.csv
<SUBJECT>_<TRIAL>_all_experiments_metric_summary.json
```

Each experiment also receives an individual folder containing per-experiment summary files and per-frame metric CSV files.

## 9. Analyze degradation relative to baselines

**Script:** `08_analyze_experiment_degradation_relative_to_baselines.py`

**Purpose:** Reads the outputs from script 07 and compares experiment conditions relative to E0 and E1. It does not rerun SMPL, WHAM, or metric computation.

```text
E0 = neutral technical baseline
E1 = living-room/home baseline
```

**Run:**

```powershell
python scripts\08_analyze_experiment_degradation_relative_to_baselines.py ^
    --subject SUB01 ^
    --sequence SUB01_off_walk_1
```

Optional condition map:

```powershell
python scripts\08_analyze_experiment_degradation_relative_to_baselines.py ^
    --subject SUB01 ^
    --sequence SUB01_off_walk_1 ^
    --condition-map-csv data\experiment_condition_map.csv
```

**Output:**

```text
results\clinically_relevant_metrics_v1\batch_wham_experiments\<SUBJECT>\<TRIAL>\degradation_analysis\
```

Important files:

```text
degradation_vs_E0_summary.csv
degradation_vs_E1_summary.csv
experiment_ranking_by_clinical_degradation_vs_E1.csv
frame_level_reliability_summary_vs_E1.csv
frame_level_delta_vs_E1_selected.csv
thesis_key_results_table.csv
condition_map_template.csv
```

## 10. Analyze the reduced matrix for additional sequences

**Script:** `09_analyze_reduced_matrix_replication_comparisons.py`

**Purpose:** Analyzes the reduced experiment matrix for the additional walking sequences. It reads the metric outputs from script 07 and performs pairwise comparisons between reduced-matrix conditions.

The reduced matrix is interpreted as:

```text
R0 = neutral technical baseline
R1 = home/living-room baseline
R2 = stable frontal setup
R3 = upper-corner viewpoint
R4 = strong lower-body occlusion
R5 = partial frontal one-leg occlusion
```

**Run examples:**

```powershell
python scripts\09_analyze_reduced_matrix_replication_comparisons.py --subject SUB02 --trial SUB02_off_walk_2
python scripts\09_analyze_reduced_matrix_replication_comparisons.py --subject SUB08 --trial SUB08_off_walk_4
python scripts\09_analyze_reduced_matrix_replication_comparisons.py --subject SUB17 --trial SUB17_off_walk_5
```

**Output:**

```text
results\clinically_relevant_metrics_v1\batch_wham_experiments\<SUBJECT>\<TRIAL>\reduced_matrix_analysis\
```

Important files:

```text
reduced_matrix_key_results_table.csv
reduced_matrix_pairwise_comparisons.csv
reduced_matrix_reliability_classification.csv
reduced_matrix_condition_map_template.csv
reduced_matrix_frame_level_pairwise_summary.csv
reduced_matrix_frame_level_pairwise_deltas_selected.csv
```

Important thesis note: in the final results interpretation, R5 corresponds to the frontal/table partial one-leg occlusion condition. For the primary full matrix this is E11.

## 11. Generate final result tables and figures

**Script:** `10_make_results_tables_figures.py`

**Purpose:** Collects metric outputs and generates thesis-ready result tables and figures. It combines selected-sequence information, render-input validation results, full-matrix metrics, reduced-matrix metrics, pairwise comparisons, and reliability summaries.

**Run:**

```powershell
python scripts\10_make_results_tables_figures.py
```

Optional explicit inputs:

```powershell
python scripts\10_make_results_tables_figures.py ^
    --metrics-zip clinically_relevant_metrics_v1.zip ^
    --render-zip render_input_validation.zip ^
    --sequences-csv results\selected_first_four_bmclab_sequences.csv ^
    --out-dir results\updated_results_pack
```

**Output:**

```text
results\updated_results_pack\
```

Important subfolders:

```text
results\updated_results_pack\tables\
results\updated_results_pack\figures\
```

Example table outputs:

```text
table_1_sequences.tex
table_2_validation.tex
table_3_full_matrix.tex
table_4_sub01_full_metrics.tex
table_5_reduced_matrix.tex
table_6_reduced_metrics_all_sequences.tex
table_7_pairwise_comparisons.tex
table_8_reliability_summary.tex
```

Example figure outputs:

```text
fig_1_pipeline_validation_pa_mpjpe.png
fig_2_sub01_pa_mpjpe_by_experiment.png
fig_3_sub01_spatial_gait_errors.png
fig_4_sub01_pa_vs_stride_scatter.png
fig_5_sub01_frame_level_pa_mpjpe_selected.png
fig_6_aligned_pa_mpjpe_by_condition_subject.png
fig_7_aligned_foot_clearance_by_condition_subject.png
fig_8_aligned_stride_length_by_condition_subject.png
```

## Final notes

- Scripts 01--03 prepare the motion for Blender and Unreal Engine.
- Blender FBX export, Unreal Engine import, scene setup, and rendering are manual steps.
- Script 04 converts Unreal PNG frame folders into MP4 videos for WHAM.
- `WHAM.ipynb` is run externally, for example on Kaggle, and must be completed before script 07 can compute metrics. The Kaggle outputs should be copied back to `data\wham_outputs\<SUBJECT>\`.
- Scripts 05 and 06 validate that preprocessing did not materially distort the motion before WHAM.
- Script 07 is the main WHAM-versus-BMCLab metric script.
- Scripts 08 and 09 compare already-computed metric outputs.
- Script 10 creates thesis-ready result tables and figures.
- Final interpretation should use both technical pose metrics and clinically relevant gait metrics. PA-MPJPE alone is not sufficient for judging recording reliability.
