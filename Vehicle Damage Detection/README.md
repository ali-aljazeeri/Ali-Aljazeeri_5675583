# Vehicle Damage Detection  


## Folder / File Structure  

### 1. `sample_images_of_cardd.ipynb`  
- Provides a preview of the CarDD dataset.   

---

### 2. `damage_detection/`  
  - `V1_damage_detection_DCN+.ipynb` - **Experiment 1**  
  - `damage_detection_DCN+.ipynb` - **Experiment 2**  
  - `damage_detection_cascade_mask_rcnn_x101.ipynb` - **Experiment 3**  
  - `damage_detection_v2_albumentations_cascade_mask_rcnn_x101.ipynb` - **Experiment 4**  
  - `damage_detection_albumentations_cascade_mask_rcnn_convnext_s.ipynb` - **Experiment 5**  
- Includes **configuration files** used to run the experiments:  
  - `albumentations-cascade-mask-rcnn_convnext-s_cfg.py`  
  - `cascade-mask-rcnn_x101_cfg.py`  
  - `dcn_plus_cfg.py`  
  - `v1_dcn_plus_cfg.py`  
  - `v2_albumentations_cascade-mask-rcnn_x101_cfg.py`  

- Subfolders such as `*_output` having the logs and results. 

---

### 3. `all_models_images/`  
 
- `Predict.ipynb`: Runs inference on given input images and visualizes results.  
- `damage/`: images used in test.  
- Helps in visualizes the performance of all models on the same input image(s).  

---
### Notes  
1. **MMDetection toolbox** is used for the detection models. When running experiments, there will be a comand line to clon it. 
2. **MMDetection toolbox** generates weights for each epoch; however, only the weights from the last epoch are kept in the files, as they are the ones used. 
3. **CarDD (Car Damage Dataset)** is a public dataset and can be downloaded from {https://cardd-ustc.github.io/}.   

