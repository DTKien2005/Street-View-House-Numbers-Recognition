# 🏠 Street View House Numbers Recognition

> **USTH — Computer Vision Final Project | Group 57**

A complete house number detection and recognition system using the **SVHN** (Street View House Numbers) dataset. The project implements two approaches:

- 🚀 **YOLO26-based detector** — a deep learning single-stage object detection pipeline
- 📊 **HOG + SVM baseline** — a traditional machine learning pipeline for comparison

---

## 👥 Team Members

| Name | Student ID |
|---|---|
| Đỗ Trung Kiên | 23BI14238 |
| Ngô Hoàng Khánh Duy | 23BI14131 |
| Phùng Đỗ Việt Dũng | 23BI14112 |
| Trần Hữu Duy | 22BA13103 |

---

## 📌 Overview

Automatic house number detection is an important computer vision task with applications in navigation systems, postal automation, autonomous driving, and smart city infrastructure. This project addresses the challenges of detecting house numbers in natural scenes, including variations in illumination, viewing angles, background clutter, and overlapping digits.

### Key Features

- **SVHN → YOLO format conversion** tool for dataset preprocessing
- **YOLO26 training pipeline** with configurable hyperparameters
- **HOG + SVM digit classifier** as a traditional baseline
- **Post-processing digit grouping** algorithm to reconstruct multi-digit house numbers
- **Parameter sensitivity analysis** (confidence threshold, IoU, image resolution)

---

## 🏗️ Project Structure

```
.
├── README.md
├── HOG_SVM.py                      # Train HOG+SVM digit classifier
├── hog_svm_model.pkl               # Trained HOG+SVM model (large file)
├── hog_svm_scaler.pkl              # Feature scaler for HOG+SVM
├── yolo26s.pt                      # Pre-trained YOLO26s weights
├── checkgpu.py                     # GPU availability check
│
├── Dataset/
│   ├── svhn.yaml                   # YOLO dataset config (10 digit classes)
│   ├── convert_svhn_to_yolo.py     # Convert SVHN .mat → YOLO format
│   ├── train_svhn_yolo26.py        # Train YOLO26 on SVHN
│   ├── detect_house_number_yolo.py # YOLO inference + digit grouping
│   ├── train_32x32.mat             # SVHN training data
│   ├── extra_32x32.mat             # SVHN extra data
│   ├── test_32x32.mat              # SVHN test data
│   ├── train/ extra/ test/         # Raw SVHN images
│   ├── dataset/                    # Converted YOLO-format dataset
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── HOG_SVM/
│       ├── predict_house_number.py  # HOG+SVM inference pipeline
│       ├── hog_svm_model.pkl
│       ├── hog_svm_scaler.pkl
│       └── addresses-amber*.jpg     # Test images & results
│
├── YOLO26_img320/
│   ├── detect_house_number_yolo.py  # Multi-resolution inference test
│   └── runs/                        # Detection results
│
└── runs/                            # Training runs & results
```

---

## 🔧 Requirements

### Hardware (used in experiments)

- Intel i5-14600K CPU
- NVIDIA RTX 3060 (12 GB VRAM)
- 32 GB RAM

### Software

```
Python >= 3.8
PyTorch (with CUDA support)
ultralytics          # YOLO framework
scikit-learn         # SVM, StandardScaler
scikit-image         # HOG feature extraction
scipy                # .mat file loading
opencv-python        # Image processing
numpy
h5py                 # SVHN annotation parsing
joblib               # Model serialization
```

Install dependencies:

```bash
pip install torch torchvision ultralytics scikit-learn scikit-image scipy opencv-python numpy h5py joblib
```

---

## 📦 Dataset

The project uses the [**SVHN (Street View House Numbers)**](http://ufldl.stanford.edu/housenumbers/) dataset, which contains over **600,000 labeled digit instances** from Google Street View images.

### Download

Download the following files and place them in the `Dataset/` folder:

| File | Description |
|---|---|
| `train_32x32.mat` | Cropped training images (73,257 images) |
| `extra_32x32.mat` | Extra training images (531,131 images) |
| `test_32x32.mat` | Test images (26,032 images) |
| `train.tar.gz` | Full training images with bounding boxes |
| `extra.tar.gz` | Full extra images with bounding boxes |
| `test.tar.gz` | Full test images with bounding boxes |

### Convert to YOLO Format

```bash
cd Dataset
python convert_svhn_to_yolo.py
```

This script:
1. Reads `digitStruct.mat` from train/extra/test splits
2. Extracts bounding box coordinates
3. Normalizes to YOLO format `(class x_center y_center width height)`
4. Maps label `10` → `0` (digit zero)
5. Outputs to `dataset/images/` and `dataset/labels/`

---

## 🚀 Usage

### 1. YOLO26 — Training

```bash
cd Dataset
python train_svhn_yolo26.py
```

**Training configuration (optimized):**

| Parameter | Value |
|---|---|
| Model | YOLO26s |
| Image size | 320 × 320 |
| Epochs | 30 |
| Batch size | 128 |
| Optimizer | Default (SGD) |
| AMP | Enabled |
| Mosaic | 1.0 (disabled last 10 epochs) |
| Flip LR/UD | Disabled |

### 2. YOLO26 — Inference

```bash
cd Dataset
python detect_house_number_yolo.py
```

The inference script:
1. Loads the best trained weights
2. Runs YOLO prediction on the input image
3. Groups detected digits into house numbers based on:
   - Horizontal proximity (gap < 1.5× average digit width)
   - Vertical alignment (vertical difference < max digit height)
4. Outputs detected house numbers with confidence scores

**Inference parameters:**

| Parameter | Selected Value |
|---|---|
| Confidence threshold | 0.3 |
| IoU threshold (NMS) | 0.2 |
| Image size | 960–1280 |

### 3. HOG + SVM — Training

```bash
python HOG_SVM.py
```

This script:
1. Loads SVHN `.mat` files (train + extra + test)
2. Converts to grayscale
3. Extracts HOG features (9 orientations, 8×8 pixels/cell, 2×2 cells/block)
4. Scales features with `StandardScaler`
5. Trains `LinearSVC` with Nyström RBF kernel approximation (3000 components)
6. Evaluates on test set and saves model

### 4. HOG + SVM — Inference

```bash
cd Dataset/HOG_SVM
python predict_house_number.py <image_path>
```

Pipeline:
1. Detect digit candidates using MSER + adaptive thresholding
2. Filter by aspect ratio and size constraints
3. Classify each candidate with HOG + SVM
4. Apply Non-Maximum Suppression (NMS)
5. Sort left-to-right and assemble house number

---

## 📈 Results

### YOLO26 Performance (Best Configuration: 320×30 epochs)

| Metric | Value |
|---|---|
| **Precision** | **0.907** |
| **Recall** | **0.845** |
| **mAP@0.5** | **0.904** |
| **mAP@0.5:0.95** | **0.451** |

### Training Configuration Comparison

| Setting | mAP@0.5 | mAP@0.5:0.95 | Training Time |
|---|---|---|---|
| 640 × 10 epochs | 0.890 | 0.425 | ~10 h |
| **320 × 30 epochs** | **0.904** | **0.451** | **~8 h** |

### Per-Class Detection Performance

| Digit | Instances | Precision | Recall | mAP@0.5 |
|---|---|---|---|---|
| 0 | 1,744 | 0.899 | 0.872 | 0.914 |
| 1 | 5,099 | 0.891 | 0.762 | 0.853 |
| 2 | 4,148 | 0.918 | 0.864 | 0.922 |
| 3 | 2,882 | 0.910 | 0.840 | 0.900 |
| 4 | 2,523 | 0.912 | 0.868 | 0.908 |
| 5 | 2,383 | 0.902 | 0.861 | 0.913 |
| 6 | 1,977 | 0.899 | 0.849 | 0.914 |
| 7 | 2,019 | 0.916 | 0.842 | 0.908 |
| 8 | 1,660 | 0.910 | 0.846 | 0.903 |
| 9 | 1,595 | 0.912 | 0.847 | 0.905 |

### HOG + SVM Baseline

| Metric | Value |
|---|---|
| Accuracy | 0.86 |
| Macro Precision | 0.84 |
| Macro Recall | 0.84 |
| Macro F1-Score | 0.84 |

### Inference Speed (YOLO, RTX 3060)

| Stage | Time |
|---|---|
| Preprocessing | 4.2 ms |
| Inference | 47.7 ms |
| Postprocessing | 1.5 ms |

---

## 🔍 Key Findings

1. **More epochs > higher resolution**: Training for 30 epochs at 320×320 outperformed 10 epochs at 640×640, both in accuracy and training time.
2. **YOLO26 >> HOG+SVM**: The YOLO-based detector significantly outperforms the traditional HOG+SVM baseline in both detection accuracy and robustness to complex backgrounds.
3. **Digit '1' is hardest**: Due to its narrow shape and similarity to background edges/poles, digit '1' has the lowest recall (0.762).
4. **IoU threshold has minimal impact**: Since digits rarely overlap in SVHN, NMS behavior is consistent across IoU thresholds 0.2–0.6.
5. **Higher inference resolution improves detection**: Resolutions of 960–1280 pixels produce more complete detection results for real-world images.

---

## 📄 Reports & Slides

- **Report**: `Group57-Street-View-House-Numbers-Recognition-Report.pdf`
- **Slides**: `Group57-Street-View-House-Numbers-Recognition-Slides.pdf`

---

## 📚 References

1. Redmon, J. et al., "You Only Look Once: Unified, Real-Time Object Detection", CVPR 2016.
2. Netzer et al., "Reading Digits in Natural Images with Unsupervised Feature Learning", NIPS 2011.
3. Dalal and Triggs, "Histograms of Oriented Gradients for Human Detection", CVPR 2005.
4. Cortes and Vapnik, "Support Vector Networks", Machine Learning, 1995.
5. Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", NIPS 2015.
6. Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016.
7. Bochkovskiy, A. et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection", 2020.

---

## 📜 License

This project was developed as part of the Computer Vision course at the **University of Science and Technology of Hanoi (USTH)**, March 2026.
