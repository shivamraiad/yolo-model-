import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse


class YOLOValidator:

    def __init__(self, model_path, image_dir):
        self.model = YOLO(model_path)
        self.image_dir = Path(image_dir)

        self.output_dir = Path("validation_results")
        self.output_dir.mkdir(exist_ok=True)

        self.class_names = self.model.names

        print(f"Loaded model: {model_path}")
        print(f"Classes: {self.class_names}\n")

    def get_color(self, cid):
        colors = [
            (255,0,0),(0,255,0),(0,0,255),
            (255,255,0),(255,0,255),(0,255,255),
            (128,0,0)
        ]
        return colors[cid % len(colors)]

    # -----------------------------------------------------
    # IMAGE TILING
    # -----------------------------------------------------

    def split_tiles(self, image):

        h, w = image.shape[:2]

        rows = 2
        cols = 5

        tile_h = h // rows
        tile_w = w // cols

        tiles = []

        for r in range(rows):

            for c in range(cols):

                y1 = r * tile_h
                x1 = c * tile_w

                y2 = (r+1)*tile_h if r < rows-1 else h
                x2 = (c+1)*tile_w if c < cols-1 else w

                tile = image[y1:y2, x1:x2]

                tiles.append((tile, x1, y1))

        return tiles

    # -----------------------------------------------------
    # IOU FOR MASKS
    # -----------------------------------------------------

    def mask_iou(self, m1, m2):

        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()

        if union == 0:
            return 0

        return inter / union

    # -----------------------------------------------------
    # TILE INFERENCE
    # -----------------------------------------------------

    def tiled_inference(self, image):

        img_h, img_w = image.shape[:2]

        tiles = self.split_tiles(image)

        all_masks = []
        all_boxes = []
        all_classes = []
        all_conf = []

        for tile, off_x, off_y in tiles:

            results = self.model.predict(
                source=tile,
                conf=0.25,
                iou=0.45,
                device=0,
                verbose=False
            )

            r = results[0]

            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
            else:
                masks = []

            for i, box in enumerate(boxes):

                x1,y1,x2,y2 = box

                x1 += off_x
                x2 += off_x
                y1 += off_y
                y2 += off_y

                all_boxes.append([x1,y1,x2,y2])
                all_classes.append(classes[i])
                all_conf.append(confs[i])

                if len(masks):

                    m = masks[i]

                    m = cv2.resize(
                        m,
                        (tile.shape[1], tile.shape[0])
                    )

                    full = np.zeros((img_h,img_w), dtype=np.uint8)

                    full[
                        off_y:off_y+tile.shape[0],
                        off_x:off_x+tile.shape[1]
                    ] = m > 0.5

                    all_masks.append(full)

        return all_boxes, all_masks, all_classes, all_conf

    # -----------------------------------------------------
    # REMOVE OVERLAPPING MASKS
    # -----------------------------------------------------

    def remove_overlap(self, masks, classes, confs):

        keep = []

        for i in range(len(masks)):

            discard = False

            for j in keep:

                iou = self.mask_iou(masks[i], masks[j])

                if iou > 0.5:

                    if confs[i] < confs[j]:
                        discard = True
                        break
                    else:
                        keep.remove(j)

            if not discard:
                keep.append(i)

        masks = [masks[i] for i in keep]
        classes = [classes[i] for i in keep]
        confs = [confs[i] for i in keep]

        return masks, classes, confs

    # -----------------------------------------------------
    # DRAW RESULT
    # -----------------------------------------------------

    def draw(self, image, masks, classes, confs):

        img = image.copy()

        for mask, cid, conf in zip(masks, classes, confs):

            color = self.get_color(cid)

            overlay = img.copy()
            overlay[mask.astype(bool)] = color

            img = cv2.addWeighted(img,0.7,overlay,0.3,0)

            contours,_ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(img, contours, -1, color, 2)

            x,y,w,h = cv2.boundingRect(mask.astype(np.uint8))

            label = f"{self.class_names[cid]} {conf*100:.1f}%"

            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

            cv2.putText(
                img,
                label,
                (x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

        return img

    # -----------------------------------------------------
    # VALIDATE
    # -----------------------------------------------------

    def validate(self):

        exts = [".jpg",".jpeg",".png",".bmp"]

        images = [
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in exts
        ]

        print(f"Found {len(images)} images\n")

        for idx,img_path in enumerate(images,1):

            print(f"[{idx}/{len(images)}] {img_path.name}")

            image = cv2.imread(str(img_path))

            boxes,masks,classes,confs = self.tiled_inference(image)

            masks,classes,confs = self.remove_overlap(
                masks,classes,confs
            )

            out = self.draw(image,masks,classes,confs)

            save = self.output_dir / f"pred_{img_path.name}"

            cv2.imwrite(str(save), out)

        print("\nDone")
        print(f"Results saved to: {self.output_dir}")


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-folder",
        required=True,
        help="folder with images"
    )

    parser.add_argument(
        "--model",
        default="runs/segment/runs/segment/yolo26x-seg/weights/best.pt"
    )

    args = parser.parse_args()

    validator = YOLOValidator(
        args.model,
        args.image_folder
    )

    validator.validate()


if __name__ == "__main__":
    main()