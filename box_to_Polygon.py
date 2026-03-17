import os
from pathlib import Path

class BoxToPolygonConverter:
    def __init__(self):
        self.converted_count = 0
        self.already_polygon_count = 0
        self.error_count = 0
    
    def is_polygon(self, coords):
        """Check if coordinates are polygon (more than 4 points) or box (exactly 4 coords)"""
        try:
            float_coords = [float(c) for c in coords]
            # Box has 4 values: x1, y1, x2, y2
            # Polygon has 6+ values (3+ points with x,y each)
            return len(float_coords) > 4
        except:
            return False
    
    def box_to_polygon(self, x1, y1, x2, y2):
        """Convert bounding box to polygon (4 corners)"""
        # Box corners as normalized polygon: top-left, top-right, bottom-right, bottom-left
        polygon = [
            x1, y1,      # top-left
            x2, y1,      # top-right
            x2, y2,      # bottom-right
            x1, y2       # bottom-left
        ]
        return polygon
    
    def convert_label_file(self, label_path):
        """Convert all boxes in a label file to polygons"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            converted_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                try:
                    class_id = int(parts[0])
                    coords = parts[1:]
                    
                    if not coords:
                        continue
                    
                    # Check if it's already a polygon
                    if self.is_polygon(coords):
                        # Already polygon, keep as is
                        converted_lines.append(line)
                        self.already_polygon_count += 1
                    else:
                        # It's a box (4 values), convert to polygon
                        try:
                            float_coords = [float(c) for c in coords]
                            
                            if len(float_coords) == 4:
                                x1, y1, x2, y2 = float_coords
                                
                                # Normalize if needed (boxes are often in center format)
                                # YOLO box format: x_center, y_center, width, height (normalized)
                                # Need to convert to corner format
                                x_center, y_center, width, height = x1, y1, x2, y2
                                
                                # Convert from center format to corner format
                                x1_corner = x_center - width / 2
                                y1_corner = y_center - height / 2
                                x2_corner = x_center + width / 2
                                y2_corner = y_center + height / 2
                                
                                # Create polygon from box corners
                                polygon = self.box_to_polygon(x1_corner, y1_corner, x2_corner, y2_corner)
                                
                                # Format as polygon line
                                polygon_str = ' '.join([f"{p:.6f}" for p in polygon])
                                converted_lines.append(f"{class_id} {polygon_str}\n")
                                self.converted_count += 1
                            else:
                                # Keep as is if format is unknown
                                converted_lines.append(line)
                        except:
                            converted_lines.append(line)
                except:
                    continue
            
            # Write back
            if converted_lines:
                with open(label_path, 'w') as f:
                    f.writelines(converted_lines)
                return True
            else:
                return False
        except Exception as e:
            self.error_count += 1
            return False
    
    def convert_split(self, split_name):
        """Convert all labels in a split"""
        labels_path = Path(f"./{split_name}/labels")
        
        if not labels_path.exists():
            print(f"Labels folder not found: {split_name}")
            return 0
        
        label_files = list(labels_path.glob('*.txt'))
        print(f"\nConverting {split_name} split ({len(label_files)} files)...")
        
        success_count = 0
        for label_file in label_files:
            if self.convert_label_file(label_file):
                success_count += 1
        
        return success_count
    
    def print_summary(self):
        """Print conversion summary"""
        print("\n" + "="*80)
        print("BOX TO POLYGON CONVERSION SUMMARY")
        print("="*80)
        print(f"Boxes converted to polygons: {self.converted_count}")
        print(f"Already in polygon format: {self.already_polygon_count}")
        print(f"Errors: {self.error_count}")
        print("="*80 + "\n")

def main():
    print("\n" + "="*80)
    print("CONVERT ALL BOUNDING BOXES TO POLYGONS FOR SEGMENTATION")
    print("="*80)
    
    converter = BoxToPolygonConverter()
    
    total_files = 0
    for split in ['train', 'test', 'val']:
        count = converter.convert_split(split)
        total_files += count
    
    converter.print_summary()
    
    if converter.converted_count > 0 or converter.already_polygon_count > 0:
        print("Conversion complete! All labels are now in polygon format.")
        print("\nNext steps:")
        print("1. Remove YOLO cache files (optional):")
        print("   rm -f train/labels.cache test/labels.cache val/labels.cache")
        print("2. Run training:")
        print("   python train.py")
        return True
    else:
        print("No labels were converted.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
