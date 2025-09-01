# Vehicle Detection and Counting

This project contains experiments, evaluations, and error analyses for vehicle detection and counting chapter.


### **Experiment 1**
- **`car_detection_yolo11x.csv`** – YOLOv11x detection results for Experiment 1.  
- **`add_images.py`** – Combines train test vval image datasets into one dataset and remove duplication.
- **`evaluation1.py`** – Script to evaluate detection and counting performance.  

---

### **Experiment 1 – Error Analysis**
- **`2 images.py`** – Script to visualize images.  
- **`car_detection_yolo11x.xlsx`**  results from Experiment 1 with manual analysis .  
- **`image 3.png`**, **`image 4.png`**, **`image 5.png`**, **`image 6.png`** – Example images illustrating detection errors.

---

### **Experiment 2**
- **`combined_car_images/`** – all car images used in this experiment as we combined datasets.  

#### **Final Evaluation**
- **`combined_car_images_yolo11l.csv`** - YOLOv11l results.  
- **`combined_car_images_yolo11m.csv`** – YOLOv11m results.  
- **`combined_car_images_yolo11n.csv`** – YOLOv11n results.  
- **`combined_car_images_yolo11s.csv`** – YOLOv11s results.  
- **`combined_car_images_yolo11x.csv`** – YOLOv11x results.  

#### **Scripts**
- **`add_all_car_images.py`** – Combines multiple car image datasets into one dataset.  
- **`annotation.py`** – Handles annotation processing for evaluation.  
- **`evaluation2.py`** – Evaluates detection performance for Experiment 2.

---

## **Notes**
- All CSV files contain:  
  - **Image name**  
  - **Ground truth**
  - **Predicted count** 
  - **Evaluation metrics** at the end of the file

---
