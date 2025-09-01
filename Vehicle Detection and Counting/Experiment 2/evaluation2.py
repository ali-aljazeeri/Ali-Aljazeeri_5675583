from ultralytics import YOLO
import os
import cv2
import json
import csv

# YOLO model
model = YOLO("yolo11x.pt")

# paths
image_dir = "./combined_car_images/images"
annotation_file = "./combined_car_images/labels/annotations.json"
output_csv = "./final evaluation/combined_car_images_yolo11x.csv"

# load annotations
with open(annotation_file, "r") as f:
    annotations = json.load(f)

# output col
csv_rows = [("image_name", "GT_car_count", "Predicted_car_count")]

# things to calculate
correct_matches = 0
total_images_with_cars = 0
absolute_errors = []
squared_errors = []


for image_name, dots in annotations.items():
    gt_vehicle_count = len(dots)

    if gt_vehicle_count == 0:
        continue

    total_images_with_cars += 1
    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Skipping.")
        continue

    img = cv2.imread(image_path)
    results = model.predict(img, verbose=False)

    predicted_vehicle_count = sum(
        1 for box in results[0].boxes if model.names[int(box.cls)] in {"car", "bus", "truck"}
    )

    if predicted_vehicle_count == gt_vehicle_count:
        correct_matches += 1

    abs_error = abs(predicted_vehicle_count - gt_vehicle_count)
    sq_error = (predicted_vehicle_count - gt_vehicle_count) ** 2
    absolute_errors.append(abs_error)
    squared_errors.append(sq_error)

    csv_rows.append((image_name, gt_vehicle_count, predicted_vehicle_count))

# Compute metrics
exact_match_accuracy = correct_matches / total_images_with_cars if total_images_with_cars else 0
mae = sum(absolute_errors) / total_images_with_cars if total_images_with_cars else 0
mse = sum(squared_errors) / total_images_with_cars if total_images_with_cars else 0

# Add summary rows
csv_rows.append(("", "", ""))
csv_rows.append(("", "", ""))
csv_rows.append(("", "", ""))
csv_rows.append(("", "", ""))
csv_rows.append(("Metric", "Value", ""))
csv_rows.append(("Exact Match Accuracy", f"{exact_match_accuracy:.3%}", ""))
csv_rows.append(("Mean Absolute Error", f"{mae:.3f}", ""))
csv_rows.append(("Mean Squared Error", f"{mse:.3f}", ""))

# Save to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

# Print results
print(f"\nEvaluated {total_images_with_cars} images with car annotations.")
print(f"Exact Match Accuracy: {exact_match_accuracy:.3%} ({correct_matches}/{total_images_with_cars})")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"Mean Squared Error: {mse:.3f}")
print(f"CSV saved to: {output_csv}")

