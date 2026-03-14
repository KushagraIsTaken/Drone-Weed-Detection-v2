# UAV Simulation Setup and Execution Guide

This project contains a hybrid Python and MATLAB simulation pipeline for weed detection using YOLOv11 models on simulated drone imagery. 

## Prerequisites
- **Python 3.8+** with the following packages:
  - `onnxruntime`
  - `numpy`
  - `opencv-python` (`cv2`)
  - `Pillow` (`PIL`)
  - `tqdm`
- **MATLAB** with the following toolboxes:
  - UAV Toolbox (for `uavScenario` and `show3D`)
  - Image Processing Toolbox (optional, but recommended for some plot types)

## Usage Instructions

### Step 1: Populate Data
1. Place your `.jpg` test images inside `matlab_sim/data/test_images/`.
2. Place your corresponding YOLO `.txt` labels inside `matlab_sim/data/test_labels/`.
3. Place your 6 trained `.onnx` models inside `matlab_sim/models/`:
   - `teacher.onnx`
   - `small_baseline.onnx`
   - `nano_baseline.onnx`
   - `nano_feature_kd.onnx`
   - `small_feature_kd.onnx`
   - `nano_pseudo_kd.onnx`
   - `small_pseudo_kd.onnx`

### Step 2: Verify Setup
Run the setup verification script from the root of `matlab_sim/`:
```bash
python python/verify_setup.py
```
This script will check if all dependencies are installed, and if your data and model files are correctly placed. Fix any errors before proceeding.

### Step 3: Run Inference
Execute the main inference script. This will simulate the drone at various altitudes (rescaling the images), run the ONNX models, compute mAP@50, map detections to field coordinates, and save the results to `python/results/results.json`.
```bash
python python/run_inference.py
```
*Note: Depending on the number of images and CPU speed, this may take a few minutes.*

### Step 4: Run MATLAB Simulation
1. Open MATLAB and navigate to the `matlab_sim/matlab/` folder.
2. Run the master script:
```matlab
main_simulation
```
3. This script will read `results.json`, build coverage maps, simulate battery usage, compute field coverage, and generate all necessary plots.

## Expected Outputs
The MATLAB script will generate 7 figures and save them as both `.fig` and `.png` inside `matlab_sim/matlab/figures/`:
- **Figure 1:** Altitude vs mAP50 curve
- **Figure 2:** Coverage heatmap for the best KD model at 15m
- **Figure 3:** Coverage heatmap comparison grid across models
- **Figure 4:** Missed patches per hectare bar chart
- **Figure 5:** Field coverage percentage vs altitude
- **Figure 6:** uavScenario 3D visualization (Lawnmower path)
- **Figure 7:** Battery cycles vs altitude bar chart

It will also generate a `metrics_summary.csv` file with all the computed metrics.
