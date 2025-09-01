from ultralytics import YOLO
import os
import cv2
import csv

# Load YOLO model
model = YOLO("yolo11x.pt")

# Paths
image_dir = "./car_images/images"
label_dir = "./car_images/labels"
output_csv = "./evaluation/car_detection_yolo11x.csv"

# Output rows
csv_rows = [("image_name", "GT_car_count", "Predicted_car_count")]

# things to calculate
correct_matches = 0
total_images_with_cars = 0
absolute_errors = []
squared_errors = []

# Loop through images
for image_name in os.listdir(image_dir):
    if not image_name.lower().endswith(('.jpg', '.png')):
        continue

    name_no_ext = os.path.splitext(image_name)[0]
    label_path = os.path.join(label_dir, name_no_ext + ".txt")

    # Check for ground truth car presence
    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()
        gt_car_lines = [line for line in lines if line.strip().startswith(("2", "0", "1", "4"))]

    if len(gt_car_lines) == 0:
        gt_car_count = 0
    else:
        gt_car_count = len(gt_car_lines)

    total_images_with_cars += 1

    # Predict with YOLO
    image_path = os.path.join(image_dir, image_name)
    img = cv2.imread(image_path)
    results = model.predict(img)

    predicted_car_count = sum(
        1 for box in results[0].boxes if model.names[int(box.cls)] in {"car", "bus", "truck"}
    )

    # Compare
    if predicted_car_count == gt_car_count:
        correct_matches += 1

    abs_error = abs(predicted_car_count - gt_car_count)
    sq_error = (predicted_car_count - gt_car_count) ** 2
    absolute_errors.append(abs_error)
    squared_errors.append(sq_error)

    # Store result
    csv_rows.append((image_name, gt_car_count, predicted_car_count))



# Accuracy (exact match)
accuracy = correct_matches / total_images_with_cars if total_images_with_cars > 0 else 0
mae = sum(absolute_errors) / total_images_with_cars if total_images_with_cars else 0
mse = sum(squared_errors) / total_images_with_cars if total_images_with_cars else 0

# Add summary rows
csv_rows.append(("", "", ""))
csv_rows.append(("", "", ""))
csv_rows.append(("", "", ""))
csv_rows.append(("", "", ""))
csv_rows.append(("Metric", "Value", ""))
csv_rows.append(("Exact Match Accuracy", f"{accuracy:.3%}", ""))
csv_rows.append(("Mean Absolute Error", f"{mae:.3f}", ""))
csv_rows.append(("Mean Squared Error", f"{mse:.3f}", ""))

# Write CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

# Print results
print(f"\nEvaluated {total_images_with_cars} images with car annotations.")
print(f"Exact Match Accuracy: {accuracy:.3%} ({correct_matches}/{total_images_with_cars})")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"Mean Squared Error: {mse:.3f}")
print(f"CSV saved to: {output_csv}")
