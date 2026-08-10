from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\Kien\Desktop\Computer_Vision\Final\YOLO26_img320\runs\detect\runs_svhn\yolo26_full\weights\best.pt")

image_path = r"C:\Users\Kien\Desktop\Computer_Vision\Final\Dataset\HOG_SVM\addresses-amber.jpg"

sizes = [320, 640, 960, 1280]

for size in sizes:
    print(f"\nTesting with image size = {size}")

    results = model.predict(
        image_path,
        conf=0.3,
        iou=0.2,
        imgsz=size,
        save=True,
        project="runs/imgsz_test",
        name=f"imgsz_{size}"
    )

for r in results:
    boxes = r.boxes
    digits = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        width = x2 - x1

        digits.append((x1, x2, y1, y2, cls, conf, width))

    # Sort by x1 (left edge)
    digits.sort(key=lambda x: x[0])

    # Group nearby digits into house numbers
    # Two digits belong to the same number if the gap between them
    # is less than 1.5x the average width of the two digits
    groups = []
    if digits:
        current_group = [digits[0]]
        for i in range(1, len(digits)):
            prev = current_group[-1]
            curr = digits[i]
            gap = curr[0] - prev[1]  # gap = current x1 - previous x2
            avg_width = (prev[6] + curr[6]) / 2
            # Also check vertical overlap: digits in the same number
            # should be at roughly the same height
            prev_cy = (prev[2] + prev[3]) / 2
            curr_cy = (curr[2] + curr[3]) / 2
            vertical_diff = abs(prev_cy - curr_cy)
            max_height = max(prev[3] - prev[2], curr[3] - curr[2])

            if gap < avg_width * 1.5 and vertical_diff < max_height * 1.0:
                current_group.append(curr)
            else:
                groups.append(current_group)
                current_group = [curr]
        groups.append(current_group)

    print(f"Found {len(groups)} house number(s):\n")
    for idx, group in enumerate(groups):
        number = "".join(str(d[4]) for d in group)
        avg_conf = sum(d[5] for d in group) / len(group)
        print(f"House Number {idx + 1}: {number}  (avg confidence: {avg_conf:.2f})")
        for d in group:
            print(f"  Digit: {d[4]}  Confidence: {d[5]:.2f}")