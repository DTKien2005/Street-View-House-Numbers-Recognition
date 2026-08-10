import numpy as np
from scipy.io import loadmat
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import multiprocessing
import time
import gc
import os
import joblib

# Use all CPU cores
N_JOBS = multiprocessing.cpu_count()
print(f"Using {N_JOBS} CPU cores")

# ==============================
# 1. Load SVHN .mat files
# ==============================

def load_svhn(path):
    data = loadmat(path)
    X = data['X']                # (32,32,3,N)
    y = data['y'].flatten()
    y[y == 10] = 0               # Replace label 10 with 0
    X = np.transpose(X, (3, 0, 1, 2))  # (N,32,32,3)
    return X, y

print("Loading data...")
t0 = time.time()

base = os.path.dirname(os.path.abspath(__file__))
X_train, y_train = load_svhn(os.path.join(base, "Dataset", "train_32x32.mat"))
X_extra, y_extra = load_svhn(os.path.join(base, "Dataset", "extra_32x32.mat"))
X_test, y_test   = load_svhn(os.path.join(base, "Dataset", "test_32x32.mat"))

# Combine train + extra
X_train = np.concatenate([X_train, X_extra], axis=0)
y_train = np.concatenate([y_train, y_extra], axis=0)
del X_extra, y_extra
gc.collect()

print(f"Loaded in {time.time()-t0:.1f}s  |  Train+Extra: {len(y_train)}, Test: {len(y_test)}")

# ==============================
# 2. Convert to Grayscale (vectorized)
# ==============================

def rgb_to_gray(images):
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])

X_train_gray = rgb_to_gray(X_train)
X_test_gray  = rgb_to_gray(X_test)

# Free colour arrays (~600 MB)
del X_train, X_test
gc.collect()

# ==============================
# 3. Extract HOG Features (parallel)
# ==============================

def _hog_single(img):
    return hog(img,
               orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               block_norm='L2-Hys')

def extract_hog_parallel(images, n_jobs=N_JOBS):
    features = Parallel(n_jobs=n_jobs, backend='threading', verbose=1)(
        delayed(_hog_single)(img) for img in images
    )
    return np.array(features)

print("Extracting HOG features (parallel)...")
t0 = time.time()

X_train_hog = extract_hog_parallel(X_train_gray)
X_test_hog  = extract_hog_parallel(X_test_gray)

del X_train_gray, X_test_gray
gc.collect()

print(f"HOG done in {time.time()-t0:.1f}s  |  Feature dim: {X_train_hog.shape[1]}")

# ==============================
# 4. Feature Scaling
# ==============================

scaler = StandardScaler()
X_train_hog = scaler.fit_transform(X_train_hog)
X_test_hog  = scaler.transform(X_test_hog)

# ==============================
# 5. Train SVM (Nystroem RBF approx + LinearSVC)
# ==============================
# Approximates RBF kernel in O(n*k) instead of O(n^2-n^3).
# ~50-100x faster than SVC(kernel='rbf') with similar accuracy.

print("Training SVM...")
t0 = time.time()

# Compute gamma='scale' manually: 1 / (n_features * X.var())
gamma_scale = 1.0 / (X_train_hog.shape[1] * X_train_hog.var())

svm = Pipeline([
    ('rbf_approx', Nystroem(kernel='rbf', gamma=gamma_scale,
                            n_components=3000, random_state=42)),
    ('svm', LinearSVC(C=5, max_iter=10000, dual=False)),
])
svm.fit(X_train_hog, y_train)

print(f"SVM training done in {time.time()-t0:.1f}s")

# ==============================
# 6. Evaluation
# ==============================

y_pred = svm.predict(X_test_hog)

acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 7. Save Model
# ==============================

model_path = os.path.join(base, "hog_svm_model.pkl")
scaler_path = os.path.join(base, "hog_svm_scaler.pkl")

joblib.dump(svm, model_path)
joblib.dump(scaler, scaler_path)

print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")

from sklearn.metrics import classification_report, accuracy_score

# Predict on test set
y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))