"""
House Number Detection & Recognition
=====================================
Uses the trained HOG + SVM model to detect and read multi-digit
house numbers from an input image.

Pipeline:
  1. Load trained model & scaler
  2. Preprocess image (gray, blur, edge/threshold)
  3. Find digit candidates via MSER + contour detection
  4. Classify each candidate with HOG + SVM
  5. Sort left-to-right → assemble the house number

Usage:
  python predict_house_number.py <image_path>

Example:
  python predict_house_number.py test_house.jpg
"""

import sys
import os
import numpy as np
import cv2
import joblib
from skimage.feature import hog

# ==============================
# Config
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hog_svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "hog_svm_scaler.pkl")

# Digit candidate size constraints (relative to image height)
MIN_DIGIT_H_RATIO = 0.05   # minimum digit height / image height
MAX_DIGIT_H_RATIO = 0.70   # maximum digit height / image height
MIN_ASPECT_RATIO  = 0.15   # width / height
MAX_ASPECT_RATIO  = 1.2   # width / height

# ==============================
# Load model
# ==============================

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run HOG_SVM.py first to train and save the model.")
        sys.exit(1)

    svm = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model loaded successfully.")
    return svm, scaler

# ==============================
# HOG feature extraction (same params as training)
# ==============================

def extract_hog_single(img_gray_32):
    """Extract HOG from a 32x32 grayscale image."""
    return hog(img_gray_32,
               orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               block_norm='L2-Hys')

# ==============================
# Digit candidate detection
# ==============================

def find_digit_candidates(image):
    """
    Find rectangular regions that likely contain single digits.
    Uses MSER + adaptive thresholding contours, then filters by
    aspect ratio and size.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    min_h = int(h * MIN_DIGIT_H_RATIO)
    max_h = int(h * MAX_DIGIT_H_RATIO)

    candidates = []

    # --- Method 1: MSER regions ---
    mser = cv2.MSER_create()
    mser.setMinArea(150)
    mser.setMaxArea(int(h * w * 0.3))
    regions, _ = mser.detectRegions(gray)
    for region in regions:
        x, y, rw, rh = cv2.boundingRect(region)
        candidates.append((x, y, rw, rh))

    # --- Method 2: Adaptive threshold + contours ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for block_size in [15, 25, 35]:
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, 5)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            candidates.append((x, y, rw, rh))

    # --- Filter candidates ---
    filtered = []
    for (x, y, rw, rh) in candidates:
        if rh < min_h or rh > max_h:
            continue
        aspect = rw / rh
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            continue
        # Add some padding
        pad = int(rh * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + rw + pad)
        y2 = min(h, y + rh + pad)
        filtered.append((x1, y1, x2, y2))

    return filtered

# ==============================
# Non-Maximum Suppression
# ==============================

def nms(boxes, scores, iou_threshold=0.3):
    """Standard NMS on (x1, y1, x2, y2) boxes."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return [int(k) for k in keep]

# ==============================
# Classify digit crops
# ==============================

def classify_candidates(image, boxes, svm, scaler, confidence_threshold=0.45):
    """
    Crop each box from the image, resize to 32x32, extract HOG,
    classify with the SVM, and return (box, digit, confidence).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = []

    for (x1, y1, x2, y2) in boxes:
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Resize to 32x32 (same as training images)
        crop_resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        crop_float = crop_resized.astype(np.float64)

        # Extract HOG + scale
        feat = extract_hog_single(crop_float).reshape(1, -1)
        feat = scaler.transform(feat)

        # Predict — convert decision_function to 0-1 probabilities via softmax
        decision = svm.decision_function(feat)[0]  # shape (10,) for 10 classes
        exp_d = np.exp(decision - np.max(decision))  # numerically stable softmax
        probs = exp_d / exp_d.sum()
        digit = np.argmax(probs)
        confidence = probs[digit]  # probability of top class (0.0 – 1.0)

        if confidence >= confidence_threshold:
            results.append(((x1, y1, x2, y2), int(digit), float(confidence)))

    return results

# ==============================
# Main pipeline
# ==============================

def detect_house_number(image_path):
    """
    Full pipeline: detect digits in the image, classify them,
    sort left-to-right, and return the house number string.
    """
    # Load
    svm, scaler = load_model()

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot read image '{image_path}'")
        sys.exit(1)

    print(f"Image: {image_path}  ({image.shape[1]}x{image.shape[0]})")

    # Detect candidates
    candidates = find_digit_candidates(image)
    print(f"Digit candidates found: {len(candidates)}")

    if len(candidates) == 0:
        print("No digit candidates detected.")
        return ""

    # Classify all candidates
    results = classify_candidates(image, candidates, svm, scaler)
    print(f"Digits classified (above threshold): {len(results)}")

    if len(results) == 0:
        print("No confident digit detections.")
        output = image.copy()

        cv2.namedWindow("House Number Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("House Number Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return ""

    # NMS to remove duplicates
    boxes = [r[0] for r in results]
    scores = [r[2] for r in results]
    keep_idx = nms(boxes, scores, iou_threshold=0.3)

    kept = [results[i] for i in keep_idx]

    # Sort left-to-right by x-center
    kept.sort(key=lambda r: (r[0][0] + r[0][2]) / 2)

    # Assemble house number
    house_number = "".join(str(r[1]) for r in kept)

    # Draw results on image
    output = image.copy()
    for (x1, y1, x2, y2), digit, conf in kept:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{digit} ({conf:.2f})"
        cv2.putText(output, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Print results
    print(f"\n{'='*40}")
    print(f"  Detected House Number: {house_number}")
    print(f"{'='*40}")
    print(f"\nIndividual digits:")
    for i, ((x1, y1, x2, y2), digit, conf) in enumerate(kept):
        print(f"  Digit {i+1}: {digit}  (confidence: {conf:.2f}, box: [{x1},{y1},{x2},{y2}])")

    # Save annotated image (upscale if too small)
    min_display_width = 800
    scale = 1
    if output.shape[1] < min_display_width:
        scale = min_display_width / output.shape[1]
    output_large = cv2.resize(output, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

    out_path = os.path.splitext(image_path)[0] + "_result.jpg"
    cv2.imwrite(out_path, output_large)
    print(f"\nAnnotated image saved to: {out_path}  ({output_large.shape[1]}x{output_large.shape[0]})")

    # Show image (resizable window, press any key to close)
    cv2.namedWindow("House Number Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("House Number Detection", output_large.shape[1], output_large.shape[0])
    cv2.imshow("House Number Detection", output_large)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return house_number


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_house_number.py <image_path>")
        print("Example: python predict_house_number.py test_house.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_house_number(image_path)
