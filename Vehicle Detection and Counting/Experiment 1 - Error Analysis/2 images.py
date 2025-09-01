from ultralytics import YOLO
import cv2
import os
import numpy as np


model = YOLO("yolo11x.pt")

target_classes = {"car", "bus", "truck"}

image_path = "./Cars Detection/train/images/33f44ac135f3aef7_jpg.rf.ce8b19116fee64ae79c2544100d59777.jpg"
label_path = "./Cars Detection/train/labels/33f44ac135f3aef7_jpg.rf.ce8b19116fee64ae79c2544100d59777.txt"

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

image_pred = image.copy()
image_gt = image.copy()

results = model.predict(image_pred)

car_detections = []
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        conf = float(box.conf)

        if class_name in target_classes:
            car_detections.append((class_name, conf))

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image_pred, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_pred, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if os.path.exists(label_path):
    print(f"Label file found: {label_path}")
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print("Skipping invalid line:", line)
                continue

            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)

            # Convert to pixel coords
            img_h, img_w = image.shape[:2]
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            print(f"Drawing GT box: ({x1}, {y1}) to ({x2}, {y2})")

            cv2.rectangle(image_gt, (x1, y1), (x2, y2), (255, 0, 0), 2)

else:
    print(f"No label file found at {label_path}")

# plot
gap = 20
title_height = 40
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
title_color = (0, 0, 0)
bg_color = (255, 255, 255)


def add_title_bar(image, title_text):
    img_h, img_w = image.shape[:2]
    title_bar = np.full((title_height, img_w, 3), bg_color, dtype=np.uint8)
    text_size = cv2.getTextSize(title_text, font, font_scale, font_thickness)[0]
    text_x = (img_w - text_size[0]) // 2
    text_y = (title_height + text_size[1]) // 2
    cv2.putText(title_bar, title_text, (text_x, text_y), font, font_scale, title_color, font_thickness)
    return np.vstack((title_bar, image))


image_gt_labeled = add_title_bar(image_gt, "Ground Truth")
image_pred_labeled = add_title_bar(image_pred, "YOLO Prediction")

gap_array = np.full((image_gt_labeled.shape[0], gap, 3), 255, dtype=np.uint8)  # white vertical strip

combined_image = np.hstack((image_gt_labeled, gap_array, image_pred_labeled))

output_path = "image 6.png"
cv2.imwrite(output_path, combined_image)
print(f"Comparison image saved to: {output_path}")

cv2.imshow("Comparison", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Print summary ---
if len(car_detections) == 1:
    print(f"{car_detections[0][0].title()} detected in the image! Confidence: {car_detections[0][1]:.2f}")
else:
    print("No valid vehicle detected or multiple vehicles found.")

print("Detected objects:", [cls for cls, _ in car_detections])
