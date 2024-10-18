# ProtoViT

This code package implements the adaptive Protoype based Vision Transformer (ProtoViT) 
from the paper "Interpretable Image Classification with Adaptive Prototype-based Vision Transformers"

## Prerequisites
PyTorch, NumPy, cv2, Augmentor (https://github.com/mdbloice/Augmentor), Timm

Recommended hardware: 1 NVIDIA Quadro RTX 6000 (24 GB), 1 NVIDIA Ge Force RTX 4090 (24 GB) or 1 NVIDIA RTX A6000 (48 GB).
<img src="./arch2.png" width="600px" >

## Dataset 
Instructions for preparing the data:
1. Download the dataset CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
3. Unpack CUB_200_2011.tgz
4. Crop the images using information from bounding_boxes.txt (included in the dataset)
5. Split the cropped images into training and test sets, using train_test_split.txt (included in the dataset)
6. Put the cropped training images in the directory "./datasets/cub200_cropped/train_cropped/"
7. Put the cropped test images in the directory "./datasets/cub200_cropped/test_cropped/"
8. Augment the training set using img_aug.py (included in this code package)
   -- this will create an augmented training set in the following directory:
      "./datasets/cub200_cropped/train_cropped_augmented/"

Dataset Stanford Cars can be downloaded from: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
If the link doesn't work, the dataset can also be found here: 
https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset/data 

The code is based on the other repositories: https://github.com/cfchen-duke/ProtoPNet

## Instructions for training the model:
1. In settings.py, provide the appropriate strings for data_path, train_dir, test_dir,
train_push_dir:
(1) data_path is where the dataset resides
    -- if you followed the instructions for preparing the data, data_path should be "./datasets/cub200_cropped/"
(2) train_dir is the directory containing the augmented training set
    -- if you followed the instructions for preparing the data, train_dir should be data_path + "train_cropped_augmented/"
(3) test_dir is the directory containing the test set
    -- if you followed the instructions for preparing the data, test_dir should be data_path + "test_cropped/"
(4) train_push_dir is the directory containing the original (unaugmented) training set
    -- if you followed the instructions for preparing the data, train_push_dir should be data_path + "train_cropped/"
2. Run main.py

## Instructions for finding the nearest prototypes to a test image:
1. Run local_analysis.py and supply the following arguments to analysis_settings.py:
   
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
load_model_dir is the directory containing the model you want to analyze
load_model_name is the filename of the saved model you want to analyze
save_analysis_path is the directory you want to save for local analysis result 
img_name is the directory containing the image you want to analyze
test_data is the directorty containing all the test images 
check_test_acc if you would like to check the model accuracy 
check_list the list of test image you would like to perform the local analysis on 

## Instructions for finding the nearest patches to each prototype:
1. Run global_analysis.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
