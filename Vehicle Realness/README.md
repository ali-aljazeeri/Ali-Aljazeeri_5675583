# Vehicle Realness and Spoof Detection  

This contains code, created datasets, and trained models for Vehicle Realness check - Spoof Detection 


###  Training  
main directory contains **model training experiments** and **final model weights**:  

- **Final trained weights (`.pth`)**  
  - `best_car_real_model_.mobilenet_v2.pth`  
  - `best_car_real_model_mobilenet_v3_large.pth`  
  - `best_car_real_model_resnet18.pth`  
  - `best_car_real_model_resnet50.pth`  
  - `best_car_real_model_resnet101.pth`  
  - `best_car_real_model_resnet152.pth`  
  - `best_car_real_model_vgg16.pth`  
  - `best_car_real_model_vgg19.pth`  
   These are the **final saved weights** for each architecture trained on the **created dataset**.  

- **Training & evaluation notebooks (`*.ipynb`)**  
  - Each notebook corresponds to a backbone experiment  
  - Includes training and evaluation
---
### Dataset spoof dataset  
this is the created dataset
  - `vehicle spoof dataset/`  
    - `car images/`  
      - `final dataset/` - cleaned dataset of real & spoofed cars  
      - `not real original/` - original spoof images before retaking them as spoofed  
    - `data random/split/`  - the final dataset after augemntion
      - `train/`, `val/`, `test/` - splits for training and evaluation  

  - `aug.ipynb` - image augmentation and split the images  

---

### Cross domain 
  - `COCO spoof created dataset/final real not real random/` - created dataset for cross domain
    - `real/` - real 
    - `spoof/` - spoofed created images
  - **Final trained weights (`.pth`)**  - same set of CNN weights as in `training`
  - **`isCarRealModel.py`** - Python file will all diffrent models classes
  - **`cross_domain_COCO_final.ipynb`** - evaluation on created COCO spoof dataset  
  - **`cross_domain_face.ipynb`** - evaluation CelebA-Spoof test set

---

### Grad-CAM 
- `grad_cam_test_images/` - images that are used for Grad-CAM  
- **`grad_cam.ipynb`**  - useing Grad-CAM with resnet-152, the most stable model.

---
### Notes 
1. **CelebA-Spoof** is a public dataset and can be downloaded from {https://github.com/ZhangYuanhan-AI/CelebA-Spoof}.   

