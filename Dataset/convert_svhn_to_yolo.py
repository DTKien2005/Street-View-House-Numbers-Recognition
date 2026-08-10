import os
import cv2
import h5py
import numpy as np


def get_value(f, obj):
    """Handles both direct values and references."""
    if isinstance(obj, h5py.Reference):
        return f[obj][()]
    else:
        return obj


def extract_bbox(mat_file, img_dir, out_img_dir, out_label_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    f = h5py.File(mat_file, 'r')
    digitStruct = f['digitStruct']

    n_images = len(digitStruct['name'])
    print(f"Processing {n_images} images from {img_dir}")

    for i in range(n_images):
        # Get image name
        name_ref = digitStruct['name'][i][0]
        name = ''.join(chr(c[0]) for c in f[name_ref][()])
        img_path = os.path.join(img_dir, name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Get bbox structure
        bbox_ref = digitStruct['bbox'][i][0]
        bbox = f[bbox_ref]

        labels = bbox['label']
        lefts = bbox['left']
        tops = bbox['top']
        widths = bbox['width']
        heights = bbox['height']

        label_lines = []

        num_digits = len(labels)
        for j in range(num_digits):

            label = get_value(f, labels[j][0])
            left = get_value(f, lefts[j][0])
            top = get_value(f, tops[j][0])
            width = get_value(f, widths[j][0])
            height = get_value(f, heights[j][0])

            label = int(np.array(label).squeeze())
            if label == 10:
                label = 0

            left = float(np.array(left).squeeze())
            top = float(np.array(top).squeeze())
            width = float(np.array(width).squeeze())
            height = float(np.array(height).squeeze())

            # Convert to YOLO format
            xc = (left + width / 2) / w
            yc = (top + height / 2) / h
            nw = width / w
            nh = height / h

            label_lines.append(f"{label} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        # Save image
        cv2.imwrite(os.path.join(out_img_dir, name), img)

        # Save label
        txt_name = name.replace('.png', '.txt')
        with open(os.path.join(out_label_dir, txt_name), "w") as f_out:
            f_out.write("\n".join(label_lines))

    f.close()
    print(f"Finished {img_dir}")


# =============================
# RUN CONVERSION
# =============================

extract_bbox(
    "train/digitStruct.mat",
    "train",
    "dataset/images/train",
    "dataset/labels/train"
)

extract_bbox(
    "extra/digitStruct.mat",
    "extra",
    "dataset/images/train",
    "dataset/labels/train"
)

extract_bbox(
    "test/digitStruct.mat",
    "test",
    "dataset/images/val",
    "dataset/labels/val"
)