import os
import shutil
import yaml
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

class YOLODatasetOrganizer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.train_path = self.base_path / "train"
        self.prepared_path = self.base_path / "prepared"
        self.classes = []
        self.selected_classes = [
            'BONDTITE_FAST - CLEAR',
            'BONDTITE_SUPER STRENGTH',
            'Fevicol_HEATX',
            'Fevicol_SH',
            'Fevikwik_203',
            'RESIBOND_GENERAL PURPOSE GP 100',
            'VETRA_LV 401'
        ]
        
    def read_yaml(self):
        """Read data.yaml and extract classes"""
        yaml_path = self.base_path / "data.yaml"
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            self.classes = data.get('names', [])
            print(f"Classes loaded: {len(self.classes)} total classes")
            return True
        except Exception as e:
            print(f"Error reading YAML: {e}")
            return False
    
    def create_prepared_structure(self):
        """Create prepared folder with train, test, val subfolders"""
        try:
            self.prepared_path.mkdir(exist_ok=True)
            
            for split in ['train', 'test', 'val']:
                split_path = self.prepared_path / split
                split_path.mkdir(exist_ok=True)
                (split_path / 'images').mkdir(exist_ok=True)
                (split_path / 'labels').mkdir(exist_ok=True)
            
            print(f"Prepared folder structure created")
            return True
        except Exception as e:
            print(f"Error creating structure: {e}")
            return False
    
    def get_image_annotation_pairs(self):
        """Get all image and annotation files for selected classes"""
        pairs = []
        
        images_path = self.train_path / 'images'
        labels_path = self.train_path / 'labels'
        
        if not images_path.exists() or not labels_path.exists():
            print("Images or labels folder not found")
            return pairs
        
        for img_file in images_path.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                txt_path = labels_path / (img_file.stem + '.txt')
                
                if txt_path.exists() and self.has_selected_classes(txt_path):
                    pairs.append({
                        'image': img_file,
                        'label': txt_path,
                        'filename': img_file.name
                    })
        
        print(f"Found {len(pairs)} image-annotation pairs with selected classes")
        return pairs
    
    def has_selected_classes(self, annotation_path):
        """Check if annotation contains selected classes"""
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id < len(self.classes):
                        if self.classes[class_id] in self.selected_classes:
                            return True
        except:
            pass
        
        return False
    
    def box_to_polygon(self, x_center, y_center, width, height):
        """Convert YOLO box format to polygon format (4 corners)"""
        # YOLO box format: x_center, y_center, width, height (normalized 0-1)
        # Convert to polygon: corners as x1,y1,x2,y1,x2,y2,x1,y2
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Return as polygon: 4 corners in order
        polygon = [
            x1, y1,  # top-left
            x2, y1,  # top-right
            x2, y2,  # bottom-right
            x1, y2   # bottom-left
        ]
        return polygon
    
    def filter_and_remap_labels(self, label_path):
        """Filter labels to selected classes, remap IDs, and convert boxes to polygons"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            cleaned_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                try:
                    class_id = int(parts[0])
                    if class_id < len(self.classes):
                        class_name = self.classes[class_id]
                        
                        if class_name in self.selected_classes:
                            new_class_id = self.selected_classes.index(class_name)
                            coords = parts[1:]
                            
                            # Validate coordinates exist
                            if coords and len(coords) >= 4:
                                float_coords = [float(c) for c in coords]
                                
                                # Check if it's a box (exactly 4 values) or polygon (more than 4)
                                if len(float_coords) == 4:
                                    # It's a box - convert to polygon
                                    x_center, y_center, width, height = float_coords
                                    polygon = self.box_to_polygon(x_center, y_center, width, height)
                                    formatted = ' '.join([f"{p:.6f}" for p in polygon])
                                else:
                                    # It's already a polygon - keep as is
                                    formatted = ' '.join([f"{c:.6f}" for c in float_coords])
                                
                                cleaned_lines.append(f"{new_class_id} {formatted}\n")
                except:
                    pass
            
            return cleaned_lines if cleaned_lines else None
        except:
            return None
    
    def split_dataset(self, pairs):
        """Split into 100% train, 30% test, 10% val"""
        shuffled = pairs.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        test_count = int(total * 0.3)
        val_count = int(total * 0.1)
        
        return pairs.copy(), shuffled[:test_count], shuffled[test_count:test_count + val_count]
    
    def copy_files_to_split(self, pairs, split_name):
        """Copy image and label files to split folder"""
        split_path = self.prepared_path / split_name
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        for pair in pairs:
            try:
                # Copy image
                shutil.copy2(pair['image'], images_path / pair['filename'])
                
                # Filter and remap label
                cleaned_lines = self.filter_and_remap_labels(pair['label'])
                if cleaned_lines:
                    label_filename = pair['filename'].replace(pair['image'].suffix, '.txt')
                    with open(labels_path / label_filename, 'w') as f:
                        f.writelines(cleaned_lines)
            except Exception as e:
                pass
    
    def create_data_yaml(self, train_count, test_count, val_count):
        """Create data.yaml in prepared folder"""
        data = {
            'path': str(self.prepared_path.absolute()),
            'train': str((self.prepared_path / 'train' / 'images').absolute()),
            'val': str((self.prepared_path / 'val' / 'images').absolute()),
            'test': str((self.prepared_path / 'test' / 'images').absolute()),
            'nc': 7,
            'names': {i: name for i, name in enumerate(self.selected_classes)}
        }
        
        yaml_path = self.prepared_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        print(f"Created data.yaml at {yaml_path}")
    
    def create_train_py(self):
        """Create train.py in prepared folder"""
        train_script = '''import os
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
    print("\\n" + "="*80)
    print("YOLO SEGMENTATION MODEL TRAINING")
    print("="*80 + "\\n")
    
    if not Path(TRAIN_CONFIG['data']).exists():
        print(f"Error: {TRAIN_CONFIG['data']} not found!")
        return False
    
    print("Training Configuration:")
    for key, value in TRAIN_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\\nLoading YOLOv8x-seg model...")
    model = YOLO("yolov8x-seg.pt")
    
    print("Starting training...\\n")
    
    results = model.train(**TRAIN_CONFIG)
    
    print("\\n" + "="*80)
    print("Training completed!")
    print(f"Results saved in: {TRAIN_CONFIG['project']}/{TRAIN_CONFIG['name']}")
    print("="*80 + "\\n")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
'''
        
        train_path = self.prepared_path / 'train.py'
        with open(train_path, 'w') as f:
            f.write(train_script)
        
        print(f"Created train.py at {train_path}")
    
    def create_val_py(self):
        """Create val.py in prepared folder"""
        val_script = '''import os
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
        print(f"Classes: {self.class_names}\\n")
    
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
                        img[mask_indices] = cv2.addWeighted(
                            img[mask_indices],
                            0.6,
                            np.array(color, dtype=np.uint8),
                            0.4,
                            0
                        )
                        
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
        print("="*80 + "\\n")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in self.val_images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No images found in {self.val_images_dir}")
            return False
        
        print(f"Found {len(image_files)} images\\n")
        
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
        
        print("\\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Images processed: {processed_count}")
        print(f"Total detections: {total_detections}")
        if processed_count > 0:
            print(f"Average detections per image: {total_detections/processed_count:.2f}")
        print(f"Results: {self.output_dir.absolute()}")
        print("="*80 + "\\n")
        
        return True

def main():
    model_path = "runs/segment/yolo26x-seg/weights/best.pt"
    val_images_dir = "./val/images"
    
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
'''
        
        val_path = self.prepared_path / 'val.py'
        with open(val_path, 'w') as f:
            f.write(val_script)
        
        print(f"Created val.py at {val_path}")
    
    def print_statistics(self, train_pairs, test_pairs, val_pairs):
        """Print dataset statistics"""
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        
        print(f"\nSPLIT DISTRIBUTION (100% train, 30% test, 10% val):")
        print(f"  Train: {len(train_pairs)} images")
        print(f"  Test:  {len(test_pairs)} images ({len(test_pairs)/len(train_pairs)*100:.1f}%)")
        print(f"  Val:   {len(val_pairs)} images ({len(val_pairs)/len(train_pairs)*100:.1f}%)")
        
        print(f"\nSelected Classes: {len(self.selected_classes)}")
        for i, cls_name in enumerate(self.selected_classes):
            print(f"  {i}: {cls_name}")
        
        print(f"\nLabel Processing:")
        print(f"  All boxes converted to polygons for segmentation")
        print(f"  Polygons with 4+ points supported")
        
        print("\n" + "="*80)
    
    def organize_dataset(self):
        """Main organization process"""
        print("\n" + "="*80)
        print("YOLO DATASET ORGANIZER")
        print("="*80 + "\n")
        
        # Step 1: Read YAML
        print("Step 1: Reading data.yaml...")
        if not self.read_yaml():
            return False
        
        # Step 2: Create structure
        print("Step 2: Creating folder structure...")
        if not self.create_prepared_structure():
            return False
        
        # Step 3: Get image pairs
        print("Step 3: Finding image-annotation pairs...")
        pairs = self.get_image_annotation_pairs()
        if not pairs:
            print("No image-annotation pairs found!")
            return False
        
        # Step 4: Split dataset
        print("Step 4: Splitting dataset...")
        train_pairs, test_pairs, val_pairs = self.split_dataset(pairs)
        print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}, Val: {len(val_pairs)}")
        
        # Step 5: Copy files
        print("\nStep 5: Copying and processing files...")
        print("  Train (converting boxes to polygons)...", end=" ")
        self.copy_files_to_split(train_pairs, 'train')
        print("Done")
        
        print("  Test (converting boxes to polygons)...", end=" ")
        self.copy_files_to_split(test_pairs, 'test')
        print("Done")
        
        print("  Val (converting boxes to polygons)...", end=" ")
        self.copy_files_to_split(val_pairs, 'val')
        print("Done")
        
        # Step 6: Create data.yaml
        print("\nStep 6: Creating configuration files...")
        self.create_data_yaml(len(train_pairs), len(test_pairs), len(val_pairs))
        self.create_train_py()
        self.create_val_py()
        
        # Step 7: Print statistics
        print("\nStep 7: Dataset statistics...")
        self.print_statistics(train_pairs, test_pairs, val_pairs)
        
        print("Dataset organization completed!")
        print(f"Output: {self.prepared_path}")
        print("\nWhat was done:")
        print("  - Filtered to 7 selected classes")
        print("  - Remapped class IDs to 0-6")
        print("  - Converted ALL boxes to polygon format (4 corners)")
        print("  - Created 100% train, 30% test, 10% val split")
        print("\nNext steps:")
        print("  1. cd prepared/")
        print("  2. python train.py")
        print("  3. python val.py (after training)\n")
        
        return True

def main():
    base_path = os.getcwd()
    print(f"Working directory: {base_path}\n")
    
    organizer = YOLODatasetOrganizer(base_path)
    success = organizer.organize_dataset()
    
    if not success:
        print("Dataset organization failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())