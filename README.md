# ProtoViT: Interpretable Image Classification with Adaptive Prototype-based Vision Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper ["Interpretable Image Classification with Adaptive Prototype-based Vision Transformers"](#).

<div align="center">
<img src="./arch2.png" width="600px">
</div>

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Analysis](#analysis)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

ProtoViT is a novel approach that combines Vision Transformers with prototype-based learning to create interpretable image classification models. Our implementation provides both high accuracy and explainability through learned prototypes.

## Prerequisites

### Software Requirements
- Python 3.8+
- PyTorch
- NumPy
- OpenCV (cv2)
- [Augmentor](https://github.com/mdbloice/Augmentor)
- Timm==0.4.12 (Note: Higher versions may require modifications to the ViT encoder)

### Hardware Requirements
Recommended GPU configurations:
- 1× NVIDIA Quadro RTX 6000 (24GB) or
- 1× NVIDIA GeForce RTX 4090 (24GB) or
- 1× NVIDIA RTX A6000 (48GB)

## Installation

```bash
git clone https://github.com/yourusername/ProtoViT.git
cd ProtoViT
pip install -r requirements.txt
```

## Dataset Preparation

### CUB-200-2011 Dataset

1. Download [CUB_200_2011.tgz](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
2. Extract the dataset:
   ```bash
   tar -xzf CUB_200_2011.tgz
   ```
3. Process the dataset:
   ```bash
   # Create directory structure
   mkdir -p ./datasets/cub200_cropped/{train_cropped,test_cropped}
   
   # Crop and split images using provided scripts
   python scripts/crop_images.py  # Uses bounding_boxes.txt
   python scripts/split_dataset.py  # Uses train_test_split.txt
   
   # Augment training data
   python img_aug.py
   ```

### Stanford Cars Dataset
Alternative dataset option available from:
- [Official Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Kaggle Mirror](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset/data)

## Training

1. Configure settings in `settings.py`:

```python
# Dataset paths
data_path = "./datasets/cub200_cropped/"
train_dir = data_path + "train_cropped_augmented/"
test_dir = data_path + "test_cropped/"
train_push_dir = data_path + "train_cropped/"
```

2. Start training:
```bash
python main.py
```

## Analysis

### Local Analysis
Analyze nearest prototypes for specific test images:

```bash
python local_analysis.py \
    --gpuid 0 \
    --load_model_dir path/to/model \
    --load_model_name model.pth \
    --save_analysis_path path/to/save \
    --img_name path/to/image \
    --test_data path/to/test/data \
    --check_test_acc \
    --check_list path/to/test/list
```

### Global Analysis
Find nearest patches for each prototype:

```bash
python global_analysis.py --gpuid 0
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{protovit2023,
  title={Interpretable Image Classification with Adaptive Prototype-based Vision Transformers},
  author={},
  journal={},
  year={2023}
}
```

## Acknowledgments

This implementation is based on the [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) repository. We thank the authors for their valuable work.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
