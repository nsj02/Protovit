"""Local prototype analysis script.

지정한 테스트 이미지를 로드해 가장 활성화된 프로토타입과 슬롯 위치를
시각화(원본 이미지, bbox, prototype 이미지) 형태로 저장한다.
"""

# MODEL AND DATA LOADING
import torch
import torch.utils.data
import os
import shutil
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
import argparse
import re
from helpers import makedir,find_high_activation_crop
import model
import train_and_test as tnt
from pathlib import Path
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from preprocess import undo_preprocess_input_function
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
import copy
import pickle
from torch.serialization import add_safe_globals

##### HELPER FUNCTIONS FOR PLOTTING
def makedir(path):
    """simple mkdir wrapper (중복 정의 주의: helpers.makedir과 동일)."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    """전처리된 텐서를 undo_preprocess하여 (H,W,C) 이미지로 저장."""
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0].detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1, 2, 0])
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

def save_prototype(fname, img_dir):
    """push 단계에서 저장한 PNG를 그대로 복사한다 (추가 크롭/가공 없음)."""
    p_img = plt.imread(img_dir)
    plt.imsave(fname, p_img)

def save_prototype_original_img_with_bbox(save_dir,
                                          img_rgb,
                                          sub_patches,
                                          bound_box_j,
                                          color=(0, 255, 255)):
    """슬롯별 bbox를 원본 이미지에 덧씌워 저장.

    Parameters
    ----------
    save_dir : str
        결과 이미지를 저장할 경로.
    img_rgb : np.ndarray
        `[H, W, 3]` float (0~1) 이미지. 분석 대상 원본 이미지.
    sub_patches : int
        슬롯 개수 (예: 4). 색상 팔레트는 슬롯 수 이상일 경우 순환한다.
    bound_box_j : np.ndarray
        shape `[5(~6), sub_patches]`. 각 슬롯 bbox의 픽셀 좌표. -1이면 비활성 슬롯.
    color : tuple, optional
        기본 BGR 색상. 팔레트는 내부에서 정의된다.
    """

    # matplotlib용 float RGB → OpenCV용 BGR 변환 후, 픽셀 범위를 [0,255]로 맞춘다.
    p_img_bgr = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)

    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (255, 255, 255)]
    for k in range(sub_patches):
        if bound_box_j[1, k] != -1:
            bbox_height_start_k = bound_box_j[1, k]
            bbox_height_end_k = bound_box_j[2, k]
            bbox_width_start_k = bound_box_j[3, k]
            bbox_width_end_k = bound_box_j[4, k]
            color = colors[k % len(colors)]
            cv2.rectangle(p_img_bgr,
                          (bbox_width_start_k, bbox_height_start_k),
                          (bbox_width_end_k - 1, bbox_height_end_k - 1),
                          color,
                          thickness=2)

    # 다시 RGB로 변환해 float 이미지로 저장
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(save_dir, p_img_rgb, vmin=0.0, vmax=1.0)


def local_analysis(imgs,
                   ppnet,
                   save_analysis_path,
                   test_image_dir,
                   start_epoch_number,
                   load_img_dir,
                   log,
                   prototype_layer_stride=1):
    """단일 이미지에 대한 로컬 분석을 수행하고 시각화/로그를 저장."""

    ppnet.eval()  # 분석 동안 dropout/BN 동결

    # 입력 이미지 경로를 클래스/파일명으로 분할해 결과 저장 위치를 구성한다.
    imgs_sep = imgs.split('/')              # 예: ['083.ClassName', 'Image_0001.jpg']
    img_file_name = imgs_sep[0]
    analysis_rt = os.path.join(save_analysis_path, imgs_sep[0], imgs_sep[1])  # 결과 저장 루트
    makedir(analysis_rt)

    # 모델이 기대하는 입력 크기(img_size)를 기준으로 리사이즈 후 Imagenet 정규화를 적용한다.
    img_size = ppnet.img_size               # 예: 224
    normalize = transforms.Normalize(mean=mean, std=std)
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 원본 크기를 모델 입력 크기로 맞춤
        transforms.ToTensor(),                    # [H,W,C] → [C,H,W] 텐서로 변환 (0~1)
        normalize                                 # Imagenet mean/std로 정규화
    ])

    # 분석 대상 이미지를 읽어 전처리한 뒤 배치 차원을 추가해 GPU 텐서로 변환한다.
    img_rt = os.path.join(test_image_dir, imgs)    # 원본 테스트 이미지 경로
    img_pil = Image.open(img_rt)                   # PIL.Image
    img_tensor = preprocess(img_pil)               # torch.FloatTensor [3,H,W]
    img_variable = Variable(img_tensor.unsqueeze(0))  # [1,3,H,W]; Variable 유지 (구 코드 호환)
    images_test = img_variable.cuda()              # GPU 텐서
    test_image_label = 0                           # 실제 라벨은 사용하지 않으므로 placeholder
    labels_test = torch.tensor([test_image_label])

    # greedy_distance에 필요한 슬롯 확률과 push index, 슬롯별 활성도를 계산한다.
    slots = torch.sigmoid(ppnet.patch_select * ppnet.temp)       # [1, num_proto, num_slots]
    factor = (slots.sum(-1)).unsqueeze(-1) + 1e-10               # 슬롯 확률 합 (0 방지 epsilon)
    logits, min_distances, values = ppnet(images_test)           # forward -> greedy_distance 호출
    proto_h = ppnet.prototype_shape[2]
    n_p = proto_h  # 슬롯 개수 (예: 4)
    _, _, indices = ppnet.push_forward(images_test)              # greedy_distance 기반 slot index (0~195)
    values_slot = values.clone() * (slots * n_p / factor)        # 슬롯 확률로 가중치 보정
    cosine_act = values_slot.sum(-1)                             # [1, num_proto] ProtoPNet-style 활성도

    _, predicted = torch.max(logits.data, 1) # 예측된 클래스
    log(f'The predicted label is {predicted}')
    print(f'The actual label is {labels_test.item()}')

    original_img = save_preprocessed_img(os.path.join(analysis_rt, 'original_img.png'),
                                         img_variable, index=0)     # 전처리된 입력 이미지 저장
    prototype_img_filename_prefix = 'prototype-img'               # push 시 생성된 prototype 이미지 파일명 접두사

    ##### PROTOTYPES FROM TOP-k predicted CLASSES
    k = 5

    log('Prototypes from top-%d classes:' % k)
    topk_logits, topk_classes = torch.topk(logits[0], k=k)  # 예측 logits 기준 top-k 클래스 shape [k], [k]
    for idx, c in enumerate(topk_classes.detach().cpu().numpy()): # top-k 클래스 순회
        topk_dir = os.path.join(analysis_rt, 'top-%d_class_prototypes_class%d' % ((idx + 1), c + 1))
        makedir(topk_dir)
        log('top %d predicted class: %d' % (idx + 1, c + 1))
        log('logit of the class: %f' % topk_logits[idx])
        # 해당 클래스에 배정된 프로토타입 index 추출
        class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0] # [num_class_prototypes]
        # 해당 프로토타입들의 활성 값 (cosine_act) 가져오기
        class_prototype_activations = cosine_act[0][class_prototype_indices] # [num_class_prototypes]
        # 작은 값→큰 값 순 정렬, 이후 reversed로 큰 값부터 순회
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations) # [num_class_prototypes]
        iterat = 0
        for s in reversed(sorted_indices_cls_act.detach().cpu().numpy()): # 해당 클래스 프로토타입 순회
            proto_bound_boxes = np.full(shape=[5, n_p], fill_value=-1) # [5, n_p] bbox 좌표 저장용
            prototype_index = class_prototype_indices[s] # 전체 프로토타입 중에서의 index
            proto_slots_j = (slots.squeeze())[prototype_index] # [num_slots] 이 프로토타입의 슬롯 확률
            log('prototype index: {0}'.format(prototype_index)) # 전체 프로토타입 index
            log('activation value (similarity score): {0}'.format(class_prototype_activations[s]))
            log('proto_slots_j: {0}'.format(proto_slots_j))
            log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))
            min_j_indice = indices[0][prototype_index].cpu().numpy()  # [num_slots]; flatten index 0~195
            min_j_indice = np.unravel_index(min_j_indice.astype(int), (14, 14))  # (rows, cols)
            grid_width = 16  # 224 / 14
            for k in range(n_p):
                if proto_slots_j[k] != 0:
                    fmap_height_start_index_k = min_j_indice[0][k] * prototype_layer_stride
                    fmap_height_end_index_k = fmap_height_start_index_k + 1
                    fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
                    fmap_width_end_index_k = fmap_width_start_index_k + 1
                    bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                                            [fmap_width_start_index_k, fmap_width_end_index_k]])
                    pix_bound_k = bound_idx_k * grid_width  # feature index → pixel 좌표 변환
                    proto_bound_boxes[0] = s  # 정렬된 index (시각화용 meta)
                    proto_bound_boxes[1, k] = pix_bound_k[0][0]
                    proto_bound_boxes[2, k] = pix_bound_k[0][1]
                    proto_bound_boxes[3, k] = pix_bound_k[1][0]
                    proto_bound_boxes[4, k] = pix_bound_k[1][1]

            rt = os.path.join(topk_dir,
                        'most_highly_activated_patch_in_original_img_by_top-%d_class.png' % (iterat + 1))
            save_prototype_original_img_with_bbox(rt, original_img,
                                                  sub_patches=n_p,
                                                  bound_box_j=proto_bound_boxes, color=(0, 255, 255))
             # save the prototype img
            bb_dir = os.path.join(load_img_dir, prototype_img_filename_prefix + 'bbox-original' + str(prototype_index) + '.png')
            saved_bb_dir = os.path.join(topk_dir, 'top-%d_activated_prototype_in_original_pimg_%d.png' % (iterat + 1, prototype_index))
            save_prototype(saved_bb_dir, bb_dir)
            iterat += 1


    ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE ############################
    most_activated_proto_dir = os.path.join(analysis_rt, 'most_activated_prototypes')
    makedir(most_activated_proto_dir)
    log('Most activated 10 prototypes of this image:')

    # 모든 프로토타입에 대한 cosine 활성도를 내림차순으로 정렬해 상위 10개를 고른다.
    sorted_act, sorted_indices_act = torch.sort(cosine_act[0])  # 오름차순 → 뒤에서부터 top 활성도
    for i in range(0, 10):
        proto_bound_boxes = np.full(shape=[5, n_p], fill_value=-1)  # 슬롯별 bbox 좌표 저장 배열
        log('top {0} activated prototype for this image:'.format(i + 1))
        log('top {0} activation for this image:'.format(sorted_act[-(i + 1)]))
        log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted[0]][sorted_indices_act[-(i + 1)].item()]))

        proto_indx = sorted_indices_act[-(i + 1)].detach().cpu().numpy()  # 전체 프로토타입 index
        slots_j = (slots.squeeze())[proto_indx]  # 슬롯 마스크 (0/1)

        # push 단계에서 저장한 bbox-overlay 이미지를 복사해 비교용으로 보관
        bb_dir = os.path.join(load_img_dir, prototype_img_filename_prefix + 'bbox-original' + str(proto_indx) + '.png')
        saved_bb_dir = os.path.join(most_activated_proto_dir,
                                    'top-%d_activated_prototype_in_original_pimg_%d.png' % (i + 1, proto_indx))
        save_prototype(saved_bb_dir, bb_dir)  # push 시 저장한 bbox 이미지를 그대로 복사

        # 슬롯이 선택한 patch index를 14x14 grid 좌표로 변환해 실제 픽셀 bbox를 계산한다.
        min_j_indice = indices[0][proto_indx].cpu().numpy()
        min_j_indice = np.unravel_index(min_j_indice.astype(int), (14, 14))  # 14x14 grid 좌표
        grid_width = 16  # grid index → 픽셀(224) 변환에 사용
        for k in range(n_p):
            if slots_j[k] != 0:
                fmap_height_start_index_k = min_j_indice[0][k] * prototype_layer_stride
                fmap_height_end_index_k = fmap_height_start_index_k + 1
                fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
                fmap_width_end_index_k = fmap_width_start_index_k + 1
                bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                                        [fmap_width_start_index_k, fmap_width_end_index_k]])
                pix_bound_k = bound_idx_k * grid_width
                proto_bound_boxes[0] = 0
                proto_bound_boxes[1, k] = pix_bound_k[0][0]
                proto_bound_boxes[2, k] = pix_bound_k[0][1]
                proto_bound_boxes[3, k] = pix_bound_k[1][0]
                proto_bound_boxes[4, k] = pix_bound_k[1][1]

        rt = os.path.join(most_activated_proto_dir,
                    'top-%d_most_highly_activated_patch_in_original_img.png' % (i + 1))
        save_prototype_original_img_with_bbox(rt, original_img,
                                              sub_patches=n_p,
                                              bound_box_j=proto_bound_boxes,
                                              color=(0, 255, 255))

    return None


def analyze(opt: Optional[List[str]]) -> None:
    """메인 엔트리: 모델 로드 -> 로컬 분석 실행.

    Parameters
    ----------
    opt : Optional[List[str]]
        외부에서 전달된 argv 리스트. None이면 실제 CLI 인자를 사용한다.

    Returns
    -------
    None. 결과 이미지는 `save_analysis_path` 하위에 저장되고 로그는 `local_analysis.log`에 기록된다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs=1, type=str, default='0')
    #parser.add_argument('--modeldir', nargs=1, type=str)
    #parser.add_argument('--model', nargs=1, type=str)
    #parser.add_argument('--save_analysis_dir',type = str, help = 'Path for saving analysis result') 
    #parser.add_argument('--test_dir',type = str)
    #parser.add_argument('--check_test_acc', type = bool, default=False)
    if opt is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]  # 분석에 사용할 GPU 선택

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')  # 실행 환경 확인용 출력
    kwargs = {}
    from analysis_settings import load_model_dir, load_model_name, save_analysis_path, img_name, test_data, check_test_acc

    model_base_architecture = 'cait'  # TODO: settings에서 자동 추론하도록 개선 여지
    experiment_run = '/'.join(load_model_dir.split('/')[3:])  # 실험 경로 요약 (로그용)
    test_image_dir = test_data
    makedir(save_analysis_path)  # 분석 산출물 저장 폴더 생성
    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

    load_model_path = os.path.join(load_model_dir, load_model_name)  # 분석에 사용할 push 완료 모델 경로
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)  # 파일명에서 push epoch 추출
    start_epoch_number = int(epoch_number_str)  # push epoch (로그용)
    log('load model from ' + load_model_path)
    log('model base architecture: ' + model_base_architecture)
    log('experiment run: ' + experiment_run)

    add_safe_globals([model.PPNet])
    try:
        ppnet = torch.load(load_model_path, weights_only=False)  # PyTorch >= 2.6 호환
    except TypeError:
        ppnet = torch.load(load_model_path)  # 하위 버전 호환
    ppnet = ppnet.cuda()
    normalize = transforms.Normalize(mean=mean, std=std)
    img_size = ppnet.img_size
    prototype_shape = ppnet.prototype_shape
    load_img_dir = os.path.join(load_model_dir, img_name)  # push 단계에서 저장된 prototype 이미지 위치
    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)     # 각 프로토타입이 가장 강하게 연결된 클래스 index
    prototype_max_connection = prototype_max_connection.cpu().numpy() # numpy array
    if np.sum(np.sort(prototype_max_connection) == prototype_max_connection) == ppnet.num_prototypes:
        log('All prototypes connect most strongly to their respective classes.')
    else:
        log('WARNING: Not all prototypes connect most strongly to their respective classes.')
    from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size, coefs
    #heck_test_acc = False
    if check_test_acc:
        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=2, pin_memory=False)
        log('test set size: {0}'.format(len(test_loader.dataset)))
        accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                            class_specific=True, log=log)
        log(f'the accuracy of the model is: {accu}')
    from analysis_settings import check_list
    for name in check_list:
        local_analysis(name, ppnet,
                        save_analysis_path, test_image_dir,
                        start_epoch_number,
                        load_img_dir, log=log, prototype_layer_stride=1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prototype local analysis entry point')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='The run evaluation training model')
    args, unknown = parser.parse_known_args()

    analyze(unknown)
