# BubbleID Project

This project aims to analyze and characterize bubble behavior from videos and images, using automatic detection, tracking, and statistical analysis tools. It relies on the Detectron2 library for detection and provides many scripts for analysis, visualization, and automation of processing.

This repository is dedicated to the **post-processing** pipeline. It allows students and researchers to use a pre-trained model to extract and analyze bubble properties.

This project is an improvement of the original BUBBLEID code developed by Dunlap et al. ([see](https://github.com/cldunlap73/BubbleID)).
It was first enhanced by Gio B. to detect bubble rising velocities and implement diameter extraction methods.
The current goal is to add new features for bubble detection, especially the tracking of coalescence history.

## Project Structure

- `Customizable/`: Main scripts for bubble analysis, detection, tracking, and visualization.
- `Inputs/`: Input data (CSV, JSON, detection results, etc.).
- `MODELS/`: Directory where you must place the downloaded trained detection models.
- `My_output/`, `results/`: Analysis results and graphical outputs.
- `docs/`: files for the docs generation
- `site/`: Documentation generated. Open file `index.html` to see the doc


## Main Scripts' Description


| File/Script                                    | Quick Description |
|------------------------------------------------|------------------|
| **BubbleID_My.py**                             | Main script for bubble identification from images/videos, using Detectron2 for detection and tracking. Code of original BubbleID modified by Gio|
| **BubbleIDTutorial.ipynb**                     | Jupyter notebook tutorial for using the BubbleID pipeline. Code of the original BubbleID|
| **bubbleProperties.py**                        | Computes and saves main bubble properties (departure diameter, growth time, frequencies, velocities) from processed CSV files. |
| **comparaisonCollapse.py**                     | Compares different bubble fusion scenarios to validate fusion detection algorithms. |
| **computedwell.py**                            | Analyzes bubble dwell times and detachment events from tracking files, generates evolution chains, and saves results. |
| **correctionLabel.py**                         | Corrects labels of attached bubbles based on their vertical position in the image. |
| **csteDef.py**                                 | Defines constants used throughout the scripts (ATTACHED, DETACHED, UNKNOWN). |
| **departureDiameter.py**                       | Computes the departure diameter of bubbles at detachment from tracking data. |
| **evolution_tid.py**                           | Analyzes the temporal evolution of bubbles (tracking), statistics on lifetime and detection. |
| **frequency.py**                               | Counts the number of “attached” → “detached” transitions to estimate bubble detachment frequency. |
| **gui.py**                                     | Graphical interface (Tkinter) to launch BubbleID analyses interactively. |
| **multipleAnalysis.py**                        | Allows automatic analysis of multiple experiments (chips and voltages) in a single run. |
| **parentBubble.py**                            | Manages detection and analysis of bubble fusions, mask construction, and parent management. |
| **plot2.py**                                   | Generates global result plots (e.g., frequency vs departure diameter, frequency vs voltage). |
| **plotBubble.py**                              | Visualizes individual bubble properties and compares different datasets. |
| **plotFvsQ.py**                                | Plots bubble frequency as a function of voltage, for different chips. |
| **plotVvsD.py**                                | Compares bubble rising velocity as a function of diameter. |
| **position.py**                                | Extracts bubble position (top, bottom, centroid) from contours and enriched data. |
| **velocities.py**                              | Computes bubble velocities and provides statistics on their movement. |
| **affichage/afficher_frames_autour.py**        | Displays a set of frames around a given frame in a video for visual inspection. |
| **affichage/side_by_side.py**                  | Creates a side-by-side (2x2 or 1x2) video montage from 2 to 4 videos, with labels. |
| **functions/richFileFunctions.py**             | Utility functions for reading and processing rich CSV files with bubble data. |
| **functions/rmmissing.py**                     | Removes NaN or None values from arrays or lists. Matlab like function|
| **functions/rmoutliers.py**                    | Removes outliers from numerical arrays or lists using various methods. Matlab like function|


## Downloading the Model

Before running the analysis, you need to download the pre-trained model weights.
1. Download the trained model 
2. Place the downloaded `.pth` file inside the `MODELS/` directory.


## Installation
(La procedure d'installation est reprise de  https://github.com/cldunlap73/BubbleID)

First download and install the latest [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)


Install [Python](https://www.python.org/downloads/)

Create a new environment with python 3.10, we used [anaconda](https://www.anaconda.com/download)

```bash
conda create -n bubbleid python==3.10
conda activate bubbleid
```

Install [git](https://git-scm.com/install/)

Update dependencies:
```bash
pip install --upgrade pip setuptools wheel
```
Install [pytorch](https://pytorch.org/get-started/locally/) (Make sure cuda version matches if using gpu):
```bash
# CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# CUDA 13.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Install [detectron2](https://github.com/facebookresearch/detectron2):
```bash
pip install git+https://github.com/facebookresearch/detectron2
```
Install Additional Dependencies:
```bash
pip install numpy==1.23 opencv-python filterpy super-gradients
```
Git clone the repository
```bash
git clone https://github.com/YOUR_GITHUB_NAME/BubbleID_training.git
cd BubbleID_training
```

All the code was run on Windows 11. A warning like this one will appear but doesn't impact the performance of the code
```bash
[2025-11-23 10:49:13] INFO - crash_tips_setup.py - Crash tips is enabled. You can set your environment variable to CRASH_HANDLER=FALSE to disable it
W1123 10:49:14.237000 2428 site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
[2025-11-23 10:49:26] WARNING - env_sanity_check.py - Failed to verify operating system: Deci officially supports only Linux kernels. Some features may not work as expected.
```

Si VSC est utilisé il faut mettre le Python interpreter sur bubbleid:
Ctrl+Shift+P
Python:Select interpreter
