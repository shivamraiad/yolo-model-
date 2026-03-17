import os
from pathlib import Path
from ultralytics import YOLO

TRAIN_CONFIG = dict(
    data     = "data.yaml",
    epochs   = 300,
    imgsz    = 1280,
    batch    = 16,     
    device   = 0,
    workers  = 8,
    project  = "runs/segment",
    name     = "yolo26x-seg",
    exist_ok = True,
)

def main():
    print("\n" + "="*80)
    print("YOLO SEGMENTATION MODEL TRAINING")
    print("="*80 + "\n")
    
    if not Path(TRAIN_CONFIG['data']).exists():
        print(f"Error: {TRAIN_CONFIG['data']} not found!")
        return False
    
    print("Training Configuration:")
    for key, value in TRAIN_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nLoading YOLOv8x-seg model...")
    model = YOLO("yolov8x-seg.pt")
    
    print("Starting training...\n")
    
    results = model.train(**TRAIN_CONFIG)
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Results saved in: {TRAIN_CONFIG['project']}/{TRAIN_CONFIG['name']}")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
