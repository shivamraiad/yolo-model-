import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class YOLOValidator:
    def __init__(self, model_path, val_images_dir):
        self.model_path = model_path
        self.val_images_dir = Path(val_images_dir)
        self.output_dir = Path("validation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"Classes: {self.class_names}\n")
    
    def get_color_by_class(self, class_id):
        """Generate consistent color for each class"""
        colors = [
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 0),      # Dark Blue
        ]
        return colors[class_id % len(colors)]
    
    def draw_results(self, image, results):
        """Draw bounding boxes, masks and labels on image"""
        img = image.copy()
        img_h, img_w = img.shape[:2]
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Draw segmentation masks first
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                
                for idx, mask in enumerate(masks):
                    if idx < len(class_ids):
                        class_id = class_ids[idx]
                        color = self.get_color_by_class(class_id)
                        
                        mask_resized = cv2.resize(
                            mask.astype(np.uint8),
                            (img_w, img_h),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        mask_indices = mask_resized > 0.5

                        overlay = img.copy()
                        overlay[mask_indices] = color

                        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                        
                        contours, _ = cv2.findContours(
                            (mask_resized * 255).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(img, contours, -1, color, 2)
            
            # Draw bounding boxes on top
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                class_name = self.class_names.get(class_id, f"Class {class_id}")
                confidence = conf * 100
                
                color = self.get_color_by_class(class_id)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"{class_name} {confidence:.1f}%"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, thickness
                )
                
                cv2.rectangle(
                    img,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width + 5, y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    img,
                    label_text,
                    (x1 + 2, y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
        
        return img
    
    def validate(self):
        """Run validation on all images"""
        print("="*80)
        print("YOLO SEGMENTATION VALIDATION")
        print("="*80 + "\n")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in self.val_images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No images found in {self.val_images_dir}")
            return False
        
        print(f"Found {len(image_files)} images\n")
        
        total_detections = 0
        processed_count = 0
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] {image_path.name}...", end=" ")
            
            image = cv2.imread(str(image_path))
            if image is None:
                print("Error")
                continue
            
            results = self.model.predict(
                source=image_path,
                conf=0.25,
                iou=0.45,
                device=0,
                verbose=False
            )
            
            annotated_image = self.draw_results(image, results)
            
            num_detections = len(results[0].boxes) if results[0].boxes else 0
            total_detections += num_detections
            
            output_path = self.output_dir / f"pred_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_image)
            
            print(f"Detections: {num_detections}")
            processed_count += 1
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Images processed: {processed_count}")
        print(f"Total detections: {total_detections}")
        if processed_count > 0:
            print(f"Average detections per image: {total_detections/processed_count:.2f}")
        print(f"Results: {self.output_dir.absolute()}")
        print("="*80 + "\n")
        
        return True

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", required=True, help="Path to images folder")
    parser.add_argument("--model", default="runs/segment/runs/segment/yolo26x-seg/weights/best.pt")

    args = parser.parse_args()

    model_path = args.model
    val_images_dir = args.image_folder

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return False

    if not Path(val_images_dir).exists():
        print(f"Error: Validation images not found at {val_images_dir}")
        return False

    validator = YOLOValidator(model_path, val_images_dir)
    success = validator.validate()

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)