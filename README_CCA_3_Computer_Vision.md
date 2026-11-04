# CCA 3 -- Computer Vision Practical Assignment

A practical computer vision project using Python, OpenCV, NumPy,
Matplotlib, Pandas, scikit-image, and Ultralytics YOLO. The project is
organised into four sections in one Jupyter Notebook.

## Table of Contents

-   [1. Project Overview](#1-project-overview)
-   [2. Objectives](#2-objectives)
-   [3. Experiments](#3-experiments)
-   [4. Repository Structure](#4-repository-structure)
-   [5. Requirements and Installation](#5-requirements-and-installation)
-   [6. Prepare Input Data](#6-prepare-input-data)
-   [7. Run in Jupyter Notebook](#7-run-in-jupyter-notebook)
-   [8. Run in Google Colab](#8-run-in-google-colab)
-   [9. Step-by-Step Experiment Guide](#9-step-by-step-experiment-guide)
-   [10. Outputs and Evaluation](#10-outputs-and-evaluation)
-   [11. Troubleshooting](#11-troubleshooting)
-   [12. Submission Checklist](#12-submission-checklist)

------------------------------------------------------------------------

## 1. Project Overview

This assignment explores image restoration, image segmentation, visual
feature extraction and matching, object detection, and motion
estimation. All experiments are presented in a single notebook
containing Python code, Markdown explanations, visual results, tables,
and observations.

## 2. Objectives

-   Add noise to images and restore them using spatial filters.
-   Evaluate image quality using PSNR and SSIM.
-   Segment images using thresholding, morphology, Canny edges, and
    Watershed.
-   Extract HOG and SIFT features and match features between images.
-   Examine the effect of rotation and resizing on SIFT matching.
-   Detect objects using Haar Cascade and pretrained YOLO.
-   Detect motion using frame differencing and GMM background
    subtraction.
-   Estimate feature-point movement using Lucas--Kanade optical flow.
-   Compare the outputs and limitations of the techniques.

## 3. Experiments

### Task 1 -- Image Restoration and Quality Assessment

  -----------------------------------------------------------------------
  Experiment                          Description
  ----------------------------------- -----------------------------------
  Gaussian noise                      Adds random intensity variations
                                      sampled from a Gaussian
                                      distribution.

  Salt-and-Pepper noise               Randomly changes selected pixels to
                                      black or white.

  Gaussian filter                     Smooths an image and can reduce
                                      Gaussian noise.

  Median filter                       Uses neighbourhood median values;
                                      useful for impulse noise.

  PSNR                                Measures pixel-level reconstruction
                                      error relative to the original.

  SSIM                                Measures structural similarity
                                      relative to the original.
  -----------------------------------------------------------------------

Expected results: original/noisy/filtered image comparisons, a PSNR and
SSIM table, and metric graphs.

### Task 2 -- Image Segmentation

  -----------------------------------------------------------------------
  Experiment                          Description
  ----------------------------------- -----------------------------------
  Binary thresholding                 Separates pixels according to an
                                      intensity threshold.

  Erosion                             Shrinks foreground regions.

  Dilation                            Expands foreground regions.

  Opening                             Erosion followed by dilation; can
                                      remove small foreground noise.

  Closing                             Dilation followed by erosion; can
                                      fill small gaps.

  Canny                               Detects edges using image gradients
                                      and thresholding.

  Watershed                           Uses markers to identify region
                                      boundaries.
  -----------------------------------------------------------------------

Expected results: binary masks, morphology outputs, edge maps, and
Watershed boundaries.

### Task 3 -- Feature Extraction and Matching

  -----------------------------------------------------------------------
  Experiment                          Description
  ----------------------------------- -----------------------------------
  HOG                                 Extracts Histogram of Oriented
                                      Gradients features and displays a
                                      visualisation.

  SIFT                                Detects keypoints and computes
                                      local descriptors.

  SIFT matching                       Matches descriptors between the two
                                      photographs.

  Rotation test                       Rotates one image and repeats
                                      feature matching.

  Resizing test                       Resizes one image and repeats
                                      feature matching.
  -----------------------------------------------------------------------

Expected results: HOG visualisations, SIFT keypoint images, match
visualisations, and a comparison of good-match counts. Meaningful
matching is most likely when the two photographs show the same object or
overlapping scene.

### Task 4 -- Object Detection and Motion Estimation

  -----------------------------------------------------------------------
  Experiment                          Description
  ----------------------------------- -----------------------------------
  Haar Cascade                        Uses a pretrained cascade, such as
                                      the frontal-face cascade, to detect
                                      the category it was trained for.

  YOLO                                Detects supported object classes
                                      and returns bounding boxes and
                                      confidence scores.

  Frame differencing                  Finds changes between consecutive
                                      video frames.

  GMM background subtraction          Models the background over time and
                                      identifies likely foreground
                                      motion.

  Lucas--Kanade optical flow          Tracks selected feature points
                                      between frames and visualises
                                      motion vectors.
  -----------------------------------------------------------------------

Expected results: annotated detection images, foreground masks, bounding
boxes, and optical-flow vectors.

**Limitations:** A frontal-face Haar cascade is not a general object
detector. YOLO detects only classes supported by its selected pretrained
weights. Motion results depend on camera movement, lighting, scene
content, and video quality.

## 4. Repository Structure

Recommended project layout:

``` text
CCA-3-Computer-Vision/
│
├── CCA_3_Computer_Vision.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── photo1.jpeg
│   ├── photo2.jpeg
│   └── video.mp4
└── outputs/
    ├── Task_1/
    ├── Task_2/
    ├── Task_3/
    └── Task_4/
```

The output subfolders are optional. If your notebook saves all files
directly in `outputs/`, that is fine. The expected input paths in the
notebook are `data/photo1.jpeg`, `data/photo2.jpeg`, and
`data/video.mp4`. Change the paths if your filenames differ.

## 5. Requirements and Installation

### Software and libraries

-   Python 3.9 or later (use a version supported by the installed
    package releases).
-   Jupyter Notebook, JupyterLab, VS Code with the Jupyter extension, or
    Google Colab.
-   `opencv-contrib-python`
-   `numpy`
-   `matplotlib`
-   `pandas`
-   `scikit-image`
-   `ultralytics`

Internet access may be required the first time YOLO model weights are
downloaded.

### Anaconda / Jupyter setup

Open Anaconda Prompt and run:

``` bash
conda create -n cca3-cv python=3.11 -y
conda activate cca3-cv
python -m pip install --upgrade pip
python -m pip install opencv-contrib-python numpy matplotlib pandas scikit-image ultralytics jupyter ipykernel
python -m ipykernel install --user --name cca3-cv --display-name "Python (cca3-cv)"
jupyter notebook
```

Open the project notebook and select the `Python (cca3-cv)` kernel.

### VS Code setup

1.  Install Python and the Python/Jupyter extensions in VS Code.
2.  Open the project folder.
3.  Open a terminal and run the environment and installation commands
    above.
4.  Open the `.ipynb` file.
5.  Select the `cca3-cv` kernel.
6.  Run the cells from top to bottom.

### Verify packages

Run this notebook cell:

``` python
import cv2
import numpy as np
import matplotlib
import pandas as pd
import skimage
import ultralytics

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pandas:", pd.__version__)
print("scikit-image:", skimage.__version__)
print("Ultralytics:", ultralytics.__version__)
print("SIFT available:", hasattr(cv2, "SIFT_create"))
print("CascadeClassifier available:", hasattr(cv2, "CascadeClassifier"))
```

Use one OpenCV package variant in the environment. Do not install
`opencv-python`, `opencv-contrib-python`, and headless variants together
because they provide the same `cv2` module. Restart the notebook kernel
after changing OpenCV packages.

## 6. Prepare Input Data

1.  Create a folder named `data` next to the notebook.
2.  Put your own original photographs and self-recorded video in that
    folder.
3.  Use these filenames, or update the notebook paths:
    -   `photo1.jpeg`
    -   `photo2.jpeg`
    -   `video.mp4`
4.  Keep the original files unchanged so experiments can be repeated.

Check the working directory and input files:

``` python
import os

print("Working directory:", os.getcwd())
for path in ["data/photo1.jpeg", "data/photo2.jpeg", "data/video.mp4"]:
    print(path, "exists:", os.path.exists(path))
```

If a path returns `False`, correct the filename, folder, or relative
path before proceeding.

## 7. Run in Jupyter Notebook

1.  Start Jupyter from the project directory.
2.  Open `CCA_3_Computer_Vision.ipynb`.
3.  Select the correct Python kernel.
4.  Confirm the input files exist in `data/`.
5.  Choose **Kernel → Restart Kernel and Run All** (menu wording may
    vary), or run each cell in order using **Shift + Enter**.
6.  Wait for each cell to finish. YOLO may take longer on its first run
    while model weights are downloaded.
7.  Inspect images, masks, tables, graphs, and motion visualisations.
8.  Save the notebook after execution so the outputs are stored in the
    `.ipynb` file.

Always test the notebook from a fresh kernel. Do not rely on variables
left over from an earlier interactive session.

## 8. Run in Google Colab

### Open the notebook

1.  Go to Google Colab.
2.  Choose **File → Upload notebook** and select the `.ipynb` file, or
    open it from GitHub.

### Upload the input files

Run the following cell and select the three input files when prompted:

``` python
from google.colab import files
import os

os.makedirs("data", exist_ok=True)
uploaded = files.upload()

for filename in uploaded:
    os.replace(filename, os.path.join("data", filename))

print("Files in data:", os.listdir("data"))
```

Upload `photo1.jpeg`, `photo2.jpeg`, and `video.mp4`. If uploading in
multiple batches, run the cell again for each batch.

### Install packages

``` python
!pip install -q opencv-contrib-python numpy matplotlib pandas scikit-image ultralytics
```

If OpenCV packages have been changed, restart the runtime before
importing them.

### Execute and save

1.  Select **Runtime → Run all**.
2.  Confirm all cells complete and outputs display.
3.  Save a copy to Drive or use **File → Save a copy in GitHub**.
4.  Keep backups of inputs and outputs. Colab's local runtime storage is
    temporary and may be cleared when the session ends.

## 9. Step-by-Step Experiment Guide

### Task 1 -- Image Restoration

1.  **Import libraries:** Import OpenCV, NumPy, Matplotlib, Pandas, and
    PSNR/SSIM functions.
2.  **Create output directory:** Create `outputs/` before saving files.
3.  **Load original image:** Read `data/photo1.jpeg`; convert OpenCV BGR
    to RGB for Matplotlib display.
4.  **Add Gaussian noise:** Generate random Gaussian values, add them to
    the original image, clip values to 0--255, and convert to `uint8`.
5.  **Add Salt-and-Pepper noise:** Randomly select pixel positions and
    set them to white or black.
6.  **Apply Gaussian filter:** Use `cv2.GaussianBlur()` on the
    Gaussian-noisy image. Compare noise reduction with detail smoothing.
7.  **Apply Median filter:** Use `cv2.medianBlur()` on the
    Salt-and-Pepper image. Inspect removal of isolated black/white
    pixels.
8.  **Compare images:** Display original, noisy, and restored images
    side by side.
9.  **Calculate PSNR and SSIM:** Compare noisy and filtered images
    against the original. Higher values generally indicate greater
    similarity.
10. **Present results:** Display a metric table and graphs, and save the
    actual results.

### Task 2 -- Image Segmentation

1.  Load the selected image and convert it to grayscale when needed.
2.  Apply binary thresholding and compare the masks.
3.  Create a structuring element (kernel).
4.  Apply erosion, dilation, opening, and closing; display each result.
5.  Apply Canny edge detection using the lower/upper thresholds in the
    notebook.
6.  Prepare foreground/background markers for Watershed; use
    thresholding, morphology, distance transform, and connected
    components as implemented in the notebook.
7.  Run Watershed and display the resulting regions/boundaries.
8.  Save the masks and visualisations if save commands are present.

Watershed depends on marker quality and image content. It may not
clearly separate objects that do not touch or that have weak boundaries.
Describe the actual result observed.

### Task 3 -- Feature Extraction and Matching

1.  Load `data/photo1.jpeg` and `data/photo2.jpeg`.
2.  Convert images to grayscale where required.
3.  Resize to a common size if the notebook does so for HOG.
4.  Extract HOG features and display the HOG visualisation.
5.  Create SIFT using `cv2.SIFT_create()` and detect
    keypoints/descriptors with `detectAndCompute()`.
6.  Match descriptors using the notebook's Brute-Force matcher and,
    where used, the Lowe ratio test.
7.  Display the SIFT matches.
8.  Rotate the second image by the angle specified in the notebook and
    repeat feature extraction/matching.
9.  Resize the second image by the scale factor specified in the
    notebook and repeat matching.
10. Compare the actual good-match counts and visualisations. Save the
    results if save commands are present.

### Task 4 -- Object Detection and Motion Estimation

1.  Load the image selected for object detection.
2.  Load the Haar cascade XML file, convert the image to grayscale, call
    `detectMultiScale()`, and draw detection boxes.
3.  Load the pretrained YOLO model specified in the notebook (for
    example, `yolov8n.pt`), run inference, and display annotated
    detections, labels, and confidence scores where available.
4.  Compare the detectors while noting that Haar and YOLO may detect
    different categories.
5.  Open `data/video.mp4` using `cv2.VideoCapture()` and check that it
    opens.
6.  **Frame differencing:** Read consecutive frames, convert to
    grayscale, blur if required, calculate absolute difference,
    threshold the result, and display the motion mask/boxes.
7.  **GMM:** Create a MOG2 subtractor using
    `cv2.createBackgroundSubtractorMOG2()`, process video frames
    sequentially, and display foreground masks.
8.  **Lucas--Kanade:** Detect feature points with
    `cv2.goodFeaturesToTrack()` and track them with
    `cv2.calcOpticalFlowPyrLK()`. Draw vectors for successfully tracked
    points.
9.  Compare the outputs: frame differencing highlights changes between
    frames; GMM estimates foreground against a learned background;
    Lucas--Kanade tracks selected feature-point movement.
10. Save the actual annotated images, masks, and visualisations if save
    commands are present.

## 10. Outputs and Evaluation

The notebook should display results directly below the corresponding
code cells. Save useful outputs in `outputs/`.

Suggested filenames (adapt to the actual notebook save commands):

``` text
outputs/
├── Task_1/
│   ├── original_image.png
│   ├── gaussian_noise.png
│   ├── salt_pepper_noise.png
│   ├── gaussian_filtered.png
│   ├── median_filtered.png
│   ├── image_quality_results.csv
│   ├── psnr_comparison.png
│   └── ssim_comparison.png
├── Task_2/
│   ├── threshold_result.png
│   ├── erosion.png
│   ├── dilation.png
│   ├── opening.png
│   ├── closing.png
│   ├── canny_edges.png
│   └── watershed_result.png
├── Task_3/
│   ├── hog_visualisation.png
│   ├── sift_keypoints.png
│   ├── sift_matches.png
│   └── transformed_image_matches.png
└── Task_4/
    ├── haar_detection.png
    ├── yolo_detection.png
    ├── frame_difference.png
    ├── gmm_foreground_mask.png
    └── lucas_kanade_optical_flow.png
```

Save Matplotlib figures before showing them:

``` python
plt.savefig("outputs/example_graph.png", bbox_inches="tight", dpi=150)
plt.show()
```

Save an RGB image using OpenCV after converting it to BGR:

``` python
cv2.imwrite(
    "outputs/example_image.png",
    cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
)
```

Save a grayscale image or mask directly:

``` python
cv2.imwrite("outputs/example_mask.png", mask)
```

**Interpretation:** Higher PSNR generally means lower pixel-level error;
SSIM closer to 1 generally means greater structural similarity. Do not
assume a filter always gives the best score. The actual image, noise
amount, and filter parameters affect the values. Report the measurements
produced by your notebook rather than inventing results.

## 11. Troubleshooting

### Image not found / image is `None`

-   Confirm the file is inside `data/`.
-   Check spelling and file extension.
-   Check the current working directory.
-   Update the relative path if your folder layout differs.

### Video does not open

-   Confirm `data/video.mp4` exists and plays outside Python.
-   Check `cap.isOpened()` after creating `cv2.VideoCapture()`.
-   The codec may not be supported by the installed OpenCV build.

### `ModuleNotFoundError`

Install the missing package in the active notebook environment, for
example:

``` python
%pip install scikit-image ultralytics
```

Restart the kernel if necessary.

### OpenCV functions are missing

-   Confirm the notebook is using the intended Python environment.
-   Avoid installing multiple OpenCV variants together.
-   Install `opencv-contrib-python` if required by the notebook.
-   Restart the kernel and verify `cv2.__version__`.

### YOLO download/inference fails

-   Check internet access on the first run.
-   Verify the model filename/path.
-   Confirm `ultralytics` imports successfully.
-   Ensure local weights exist if using a local model file.

### Haar Cascade detects nothing

-   A frontal-face cascade detects frontal faces, not arbitrary objects.
-   Use a clear, sufficiently large, front-facing face.
-   Check that the cascade XML file loaded correctly.

### SIFT produces few matches

-   Use photographs of the same object or overlapping scene.
-   Prefer sharp, textured images.
-   Large viewpoint, lighting, and blur differences can reduce matches.

### Watershed result is poor

-   Check foreground/background polarity.
-   Inspect sure-foreground and sure-background markers.
-   Try a suitable image with distinct or touching objects.
-   Adjust threshold and morphology parameters carefully.

### Outputs folder is empty

-   Run the save cells.
-   Confirm the output path is relative to the current working
    directory.
-   Create the folder before saving:

``` python
import os
os.makedirs("outputs", exist_ok=True)
```

### Notebook outputs are missing on GitHub

GitHub does not execute notebooks. Run the notebook, wait for the output
cells to render, save the notebook, and upload/commit the saved `.ipynb`
again.

## 12. Submission Checklist

-   [ ] Notebook is named clearly, e.g. `CCA_3_Computer_Vision.ipynb`.
-   [ ] All four task sections are included.
-   [ ] All cells execute from a fresh kernel, in order.
-   [ ] Markdown explanations accompany the experiments.
-   [ ] Original input photographs and self-recorded video are available
    in `data/` or an instructor-approved location.
-   [ ] Actual outputs, visualisations, metric tables, and graphs are
    visible in the notebook.
-   [ ] Generated output files are saved in `outputs/`, where
    applicable.
-   [ ] `requirements.txt` is included and reflects the packages used.
-   [ ] This `README.md` is included in the repository.
-   [ ] No passwords, API keys, private information, or unrelated
    personal files are committed.
-   [ ] Technical report and explanation video are prepared if required
    by the assignment.
-   [ ] GitHub repository and any required Drive access are shared with
    the instructor.

**Large files:** GitHub blocks regular Git files larger than 100 MB. If
the raw video exceeds this limit, use an instructor-approved Drive link
and explain in this README how the input can be accessed. Do not
silently omit required input data.

------------------------------------------------------------------------

This README describes the workflow. The code, parameter values,
screenshots, metric values, and observations submitted should be the
ones actually produced by your own notebook and input files.
