# 📁 git_push_pr — Hướng Dẫn Push Lên Git

## Cấu trúc thư mục

```
git_push_pr/
├── README.md           ← File này
├── .gitignore          ← Loại bỏ dataset, model weights, runs
├── push_to_git.bat     ← Script cho Windows
└── push_to_git.sh      ← Script cho Linux/macOS
```

## 🚀 Hướng dẫn sử dụng

### Bước 1: Tạo repository trên GitHub

1. Vào [github.com/new](https://github.com/new)
2. Đặt tên repo: `Street-View-House-Numbers-Recognition`
3. **KHÔNG** tick "Initialize with README" (vì đã có sẵn)
4. Click **Create repository**
5. Copy URL repo (ví dụ: `https://github.com/YOUR_USERNAME/Street-View-House-Numbers-Recognition.git`)

### Bước 2: Sửa URL trong script

Mở file `push_to_git.bat` (Windows) hoặc `push_to_git.sh` (Linux/macOS), tìm dòng:

```
set REMOTE_URL=https://github.com/YOUR_USERNAME/Street-View-House-Numbers-Recognition.git
```

Thay `YOUR_USERNAME` bằng username GitHub của bạn.

### Bước 3: Chạy script

**Windows:**
```cmd
cd git_push_pr
push_to_git.bat
```

**Linux/macOS:**
```bash
cd git_push_pr
chmod +x push_to_git.sh
./push_to_git.sh
```

## 📝 .gitignore sẽ loại bỏ

Script sẽ **KHÔNG** push lên Git những file sau (quá lớn):

| Loại file | Kích thước | Lý do |
|---|---|---|
| `*.mat` (dataset) | ~1.5 GB | SVHN dataset files |
| `*.tar.gz` (archives) | ~2.6 GB | Raw dataset archives |
| `*.pkl` (model weights) | ~80 MB | Trained model files |
| `*.pt` (YOLO weights) | ~20 MB | Pre-trained YOLO weights |
| `runs/` | varies | Training artifacts |
| `Dataset/train/`, `extra/`, `test/` | ~GB | Raw images |
| `Dataset/dataset/` | ~GB | Converted YOLO images |
| `*.docx`, `*.pptx` | varies | Source documents |

## ✅ Những file SẼ được push

- `README.md` — Mô tả project
- `HOG_SVM.py` — Train HOG+SVM
- `checkgpu.py` — Check GPU
- `Dataset/svhn.yaml` — YOLO dataset config
- `Dataset/convert_svhn_to_yolo.py` — Conversion script
- `Dataset/train_svhn_yolo26.py` — YOLO training script
- `Dataset/detect_house_number_yolo.py` — YOLO inference
- `Dataset/HOG_SVM/predict_house_number.py` — HOG+SVM inference
- `YOLO26_img320/detect_house_number_yolo.py` — Multi-resolution test
- `Group57-Street-View-House-Numbers-Recognition-Report.pdf` — Report
- `Group57-Street-View-House-Numbers-Recognition-Slides.pdf` — Slides
