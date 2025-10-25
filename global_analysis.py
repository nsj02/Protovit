"""Global prototype analysis script.

학습이 완료된 ProtoViT/PPNet 모델을 로드해, 각 프로토타입이 가장 가깝게 매칭되는
train/test 이미지를 찾아 시각화를 저장한다. push 이후 생성된 bbox 이미지도 복사하여
`*_nearest_{split}` 디렉터리에 정리한다.
"""

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2
import matplotlib.pyplot as plt

import re

import os

from helpers import makedir
import find_nearest
import argparse
import model
from torch.serialization import add_safe_globals

from preprocess import preprocess_input_function

# Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
parser = argparse.ArgumentParser(description='Global prototype analysis: collect nearest train/test images')
parser.add_argument('-gpuid', nargs=1, type=str, default='0')  # 단일 GPU ID 지정 (예: '0')
#parser.add_argument('-modeldir', nargs=1, type=str)
#parser.add_argument('-model', nargs=1, type=str)
args = parser.parse_args()

from analysis_settings import load_model_dir, load_model_name, img_name
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]  # 분석 시 사용할 GPU 고정
load_model_dir = load_model_dir  # analysis_settings.py에서 가져온 디렉터리
load_model_name = load_model_name

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)  # 예: '30push' -> '30'
start_epoch_number = int(epoch_number_str)                       # 최근 push가 수행된 에폭 번호

# load the model
print('Load model from ' + load_model_path)
print('start_epoch_number: ', start_epoch_number)
add_safe_globals([model.PPNet])
try:
    ppnet = torch.load(load_model_path, weights_only=False)          # PyTorch >= 2.6 호환
except TypeError:
    ppnet = torch.load(load_model_path)          # 하위 버전 호환
ppnet = ppnet.cuda()                         # GPU로 이동해 push_forward/forward를 그대로 활용
#ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet.img_size  # 모델이 기대하는 입력 해상도 (예: 224)

# load the data
# must use unaugmented (original) dataset
from settings import train_push_dir, test_dir, train_push_batch_size, test_batch_size

train_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),  # push 시 입력 크기와 동일하게 맞춘다.
        transforms.ToTensor(),                         # 정규화 전 float 텐서 (0~1)로 변환
    ]))  # push용 원본 학습 이미지 (_aug 없이)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_push_batch_size,
    shuffle=True,          # 전체 데이터 순회 시 배치 순서만 섞음 (최근접 탐색에는 영향 없음)
    num_workers=2,
    pin_memory=False)


# test set: do not normalize
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))  # 테스트 셋도 동일 preprocessing (정규화 전)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=False)

root_dir_for_saving_train_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_train')
root_dir_for_saving_test_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_test')

makedir(root_dir_for_saving_train_images)  # train 최근접 결과 저장 루트 (프로토타입별 하위 폴더 생성 예정)
makedir(root_dir_for_saving_test_images)


def save_prototype(fname, img_dir):
    """프로토타입 원본 이미지를 지정 경로에 복사 저장."""
    p_img = plt.imread(img_dir)
    plt.imsave(fname, p_img)

    
prototype_img_filename_prefix = 'prototype-img'
for j in range(ppnet.num_prototypes):
    # --- 프로토타입 j의 push 시각화 자료 복사 ---
    # push.py/push_greedy.py가 저장한 bbox 이미지를 최근접 결과 디렉터리 안에도 넣어 비교가 쉽도록 한다.
    train_proto_dir = os.path.join(root_dir_for_saving_train_images, str(j))
    test_proto_dir = os.path.join(root_dir_for_saving_test_images, str(j))
    makedir(train_proto_dir)
    makedir(test_proto_dir)

    load_img_dir = os.path.join(load_model_dir, img_name)  # push 시 생성된 prototype 이미지 폴더
    bb_dir = os.path.join(load_img_dir,
                          prototype_img_filename_prefix + 'bbox-original' + str(j) + '.png') # bbox 이미지 경로

    # train/test 폴더 각각에 동일한 bbox 이미지를 복사해 기준으로 삼는다.
    saved_bb_dir_tr = os.path.join(train_proto_dir, 'prototype_in_original_bb.png')
    save_prototype(saved_bb_dir_tr, bb_dir)

    saved_bb_dir_ts = os.path.join(test_proto_dir, 'prototype_in_original_bb.png')
    save_prototype(saved_bb_dir_ts, bb_dir)

num_nearest_neighbors = 5
find_nearest.find_k_nearest_patches_to_prototypes(
    dataloader=train_loader,
    ppnet=ppnet,
    num_nearest_neighbors=num_nearest_neighbors,
    preprocess_input_function=preprocess_input_function,
    root_dir_for_saving_images=root_dir_for_saving_train_images,
    log=print)  # 훈련셋 기준 top-K 패치 저장

find_nearest.find_k_nearest_patches_to_prototypes(
    dataloader=test_loader,
    ppnet=ppnet,
    num_nearest_neighbors=num_nearest_neighbors,
    preprocess_input_function=preprocess_input_function,
    root_dir_for_saving_images=root_dir_for_saving_test_images,
    log=print)  # 테스트셋 기준 top-K 패치 저장 (train 결과와 동일한 구조)
