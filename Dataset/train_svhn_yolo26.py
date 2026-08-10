from ultralytics import YOLO
import torch.multiprocessing as mp

def main():
    model = YOLO("yolo26s.pt")

    model.train(
        data="svhn.yaml",
        epochs=30,
        imgsz=320,
        batch=128,
        device=0,
        workers=4,
        amp=True,
        cache="disk",
        mosaic=1.0,
        close_mosaic=10,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        box=7.5,
        cls=0.5,
        project="runs_svhn",
        name="yolo26_full"
    )

if __name__ == "__main__":
    mp.freeze_support()   # 🔥 important on Windows
    main()