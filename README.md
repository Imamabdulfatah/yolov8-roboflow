# YOLOv8 Roboflow Integration

## Project Overview
This repository provides a **YOLOv8** model pipeline that is tightly integrated with **Roboflow** for data management, labeling, and training. The goal is to enable rapid object detection model development using Roboflow's powerful annotation tools and dataset versioning.

---

## Table of Contents
- [Setup](#setup)
- [Roboflow Labeling Guide](#roboflow-labeling-guide)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Inference](#inference)
- [Running the Project](#running-the-project)
- [References](#references)

---

## Setup
1. **Clone the repository** (already done).
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Sign up for a free **Roboflow** account at https://roboflow.com and create a new project.
4. Install the Roboflow Python SDK (if not already in `requirements.txt`):
   ```bash
   pip install roboflow
   ```

---

## Roboflow Labeling Guide
### 1. Create a New Project
- Log in to Roboflow and click **"Create New Project"**.
- Choose a descriptive name (e.g., `yolov8-objects`).
- Select the **"Object Detection"** task.
- Pick the appropriate **image format** (JPEG/PNG) and click **"Create"**.

### 2. Upload Images
- In the project dashboard, click **"Upload Images"**.
- Drag‑and‑drop your raw images (or use the **"Bulk Upload"** button).
- After uploading, each image will appear in the **"Dataset"** view.

### 3. Label Images
- Click an image to open the **annotation canvas**.
- Use the toolbar to draw bounding boxes around each object.
- Assign a **class label** (e.g., `person`, `car`, `bicycle`).
- Press **"Save"** to commit the annotation.
- Repeat for all images.

### 4. Review & Export
- Use the **"Dataset"** tab to review all annotations. You can delete or edit mislabeled boxes.
- Once satisfied, click **"Export"** → **"YOLOv8"** (or generic **YOLO** if YOLOv8 format is unavailable). Choose **"v8"** for the version.
- Click **"Download"**. This provides a ZIP containing:
  - `train/` and `valid/` image folders.
  - Corresponding `.txt` label files in YOLOv8 format.
  - A `data.yaml` configuration file.

### 5. Integrate with the Repository
- Extract the ZIP into the project root (replace the existing `train/` and `valid/` directories if they exist).
- Ensure `data.yaml` points to the correct paths:
  ```yaml
  train: ./train/images
  val: ./valid/images
  nc: <number_of_classes>
  names: ["class1", "class2", ...]
  ```
- Commit the updated dataset files.

---

## Dataset Preparation
The repository ships with a sample `train.yaml` that can be renamed to `data.yaml`. Adjust the `nc` (number of classes) and `names` list to match your Roboflow labels.

```bash
mv train.yaml data.yaml  # optional rename
```

---

## Training the Model
Run the training script provided (`main.py`) with the YOLOv8 CLI:
```bash
python main.py --data data.yaml --weights yolov8n.pt --epochs 50
```
- `--weights` can be any pretrained YOLOv8 weight (`n`, `s`, `m`, `l`, `x`).
- Adjust `--epochs` based on dataset size.

Training logs and best model checkpoints are saved under `runs/`.

---

## Inference
After training, perform inference on new images:
```bash
python main.py --source path/to/image_or_folder --weights runs/detect/train/weights/best.pt
```
The script will output detection images in `runs/detect/exp/`.

---

## Running the Project
1. **Prepare the environment** (as described in Setup).
2. **Label data** using the Roboflow guide above.
3. **Train** the model with `python main.py`.
4. **Run inference** on test images.

---

## References
- YOLOv8 Documentation: https://docs.ultralytics.com/
- Roboflow Documentation: https://docs.roboflow.com/

---

*This README was auto‑generated to provide a clear workflow for labeling with Roboflow and training a YOLOv8 model.*
