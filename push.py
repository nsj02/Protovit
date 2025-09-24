import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):
    """입력/출력과 핵심 절차

    - 입력
        * `search_batch_input`: `[batch, 3, H, W]` push 대상 이미지 배치 (정규화 전)
        * `start_index_of_search_batch`: 전체 push 데이터셋에서 현재 배치 시작 인덱스
        * `prototype_network_parallel`: CUDA 위에 올라간 `PPNet` (push_forward 사용)
        * `global_min_proto_dist`: `[num_proto]` 최소 거리 기록 (numpy, in-place 업데이트)
        * `global_min_fmap_patches`: `[num_proto, feat_dim, proto_h, proto_w]` 최소 거리 패치 저장소
        * `proto_rf_boxes` / `proto_bound_boxes`: 리셉티브필드/활성 영역 bbox 기록 배열
        * 나머지 인자: 클래스별 탐색 여부, 전처리 함수, stride, 저장 경로 등
    - 출력: 없음. 위 배열/딕셔너리를 in-place로 갱신
    - 절차 개요
        1. (옵션) 전처리 후 `push_forward`로 특징맵/거리맵을 얻는다.
        2. 각 프로토타입에 대해 batch 내 최소 거리를 탐색하고, 글로벌 최소를 갱신한다.
        3. 최소를 갱신한 경우, 해당 특징 패치와 원본 이미지의 RF/고활성 영역을 기록·저장한다.
    """

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # (예: Imagenet 정규화) push 전처리가 필요하면 함수에 위임
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # push_forward: conv 특징맵, 최소 거리, 패치 index 반환 (GPU 텐서)
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())  # [B, feat_dim, h, w]
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())      # [B, num_proto, h, w]

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # 예: search_y(tensor([3, 1, 3])) → {0: [], 1: [1], 2: [], 3: [0, 2], ...}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]  # prototype patch height (보통 1)
    proto_w = prototype_shape[3]  # prototype patch width (보통 1)
    max_dist = prototype_shape[1] * proto_h * proto_w  # L2 거리 최대값 (feature_dim × patch_area)

    for j in range(n_prototypes):
        # --- (1) 프로토타입 j에 대해 배치 내 각 샘플/위치의 거리 맵을 선택 ---
        if class_specific:
            # prototype_class_identity[j]: one-hot -> argmax gives 해당 프로토타입의 타깃 클래스 ID
            target_class = torch.argmax(prototype_network_parallel.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :, :]  # [num_img_cls, h, w]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j, :, :]  # [batch, h, w]

        # --- (2) 거리 맵에서 최소 거리와 위치 찾기 (프로토타입당 최소 패치 1개만 유지) ---
        # 현재 배치에서 프로토타입 j가 본 최소 거리 (값만 비교용)
        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            # global 최소 거리보다 작으면, 위치 정보를 포함해 전역 갱신 필요
            # flatten된 최소값 위치 → np.unravel_index로 (이미지 idx, h idx, w idx) 복원
            # 예: proto_dist_j shape [num_img_cls, h, w], np.argmin → flatten index, np.unravel_index → (img_idx, h_idx, w_idx) 리스트형식으로 좌표반환)
            batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                                             proto_dist_j.shape))
            if class_specific:
                # class_specific이면 proto_dist_j가 "해당 클래스 이미지들"만 보는 상태라
                # 이미지 index를 (전체 배치 기준)으로 다시 매핑해야 함 - class_to_img_index_dict에 전체 배치 기준 들가있음
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # --- (3) 최소 거리 위치에 해당하는 특징맵 패치 추출 ---
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            # 예) 이미지 idx=3, (h_idx, w_idx)=(5,10) → stride=1이면 feature map [5:6, 10:11] 범위 추출
            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]  # [feat_dim, proto_h, proto_w]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            # --- (4) 리셉티브필드를 원본 이미지 좌표계로 변환 ---
            protoL_rf_info = prototype_network_parallel.proto_layer_rf_info  # (h_idx,w_idx --> 원본 픽셀) 변환용 정보
            # compute_rf_prototype: feature map 좌표와 receptive field info를 이용해 (이미지 idx, y1,y2,x1,x2) 반환
            rf_prototype_j = compute_rf_prototype(search_batch.size(2),
                                                  batch_argmin_proto_dist_j,
                                                  protoL_rf_info)
            
            # push는 원본 배치를 numpy로 다루므로 torch 텐서를 (H,W,C) float 이미지로 변환
            original_img_j = search_batch_input[rf_prototype_j[0]].numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))  # [H,W,C]
            original_img_size = original_img_j.shape[0]
            
            # --- (5) 리셉티브필드 및 고활성 영역 기반 시각화 자료 생성 ---
            # rf_prototype_j: feature map stride/패치 크기를 고려한 "원본 이미지의 receptive field" 위치
            #  → proto_rf_boxes에 저장되는 bbox: 모델이 L2 거리 기준으로 참조한 원본 영역
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch  # 전체 데이터셋 기준 이미지 index 이전 단계에서 class -> batch index로 바뀌었으므로 start_index 더함 
            proto_rf_boxes[j, 1] = rf_prototype_j[1] # y1
            proto_rf_boxes[j, 2] = rf_prototype_j[2] # y2
            proto_rf_boxes[j, 3] = rf_prototype_j[3] # x1
            proto_rf_boxes[j, 4] = rf_prototype_j[4] # x2
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item() # 라벨 값 달아주기 

            # 고활성 영역(activation 맵 상위 percentile) bbox 계산 (모델이 "유사도" 기준으로 강조한 영역)
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]  # 거리 맵 [h,w]
            if prototype_network_parallel.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.epsilon))
            elif prototype_network_parallel.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j  # 거리→유사도로 변환
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)

            # upsampled_act_img_j: 원본 이미지 크기로 리사이즈된 활성 맵 → find_high_activation_crop에 입력
            upsampled_act_img_j = cv2.resize(proto_act_img_j,
                                             dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            # 상위 percentile 영역 bbox (row_start, row_end, col_start, col_end)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # proto_bound_j는 활성도가 높았던 범위 → proto_bound_boxes에 기록 (시각화용)
            # proto_img_j는 그 bbox만큼 원본에서 잘라낸 patch
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]

            # 고활성 bbox를 기록 (원본 이미지 기준)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]  # 같은 이미지 index 공유
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # self activation heatmap numpy 저장
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # --- (6) push 결과 시각화 파일 저장 ---
                    # original_img_j: [H, W, 3] float32 (0~1) 범위, 정규화 해제된 원본 프레임
                    # proto_act_img_j / upsampled_act_img_j: [H, W] float, 거리 맵을 유사도로 변환한 활성도
                    # rf_img_j: 리셉티브필드 영역을 잘라낸 [rf_h, rf_w, 3] 배열 (전체와 동일하면 == 전체 이미지)
                    # proto_img_j: high-activation bbox로 잘라낸 최종 프로토타입 패치

                    # ① 원본 이미지 저장 (push 시 참고용 스냅샷)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    # ② upsampled_act_img_j를 [0,1] 범위로 정규화한 뒤 COLORMAP_JET을 적용해 heatmap 생성
                    #    - 최소값을 0으로 이동시키고 (amin) 최대값으로 나눠 대비를 확보
                    #    - cv2는 BGR 순서를 사용하므로, 나중에 RGB로 전환해 matplotlib이 올바른 색을 표시하도록 한다.
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    # overlayed_original_img_j: 원본(0.5)과 heatmap(0.3)을 합성해 활성 영역을 강조한 RGB 이미지
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    # ③ 리셉티브필드(rf) 범위가 전체 이미지와 다르면 해당 영역만 따로 저장
                    #    - rf_img_j: 모델이 cosine 거리로 선택한 원본 영역 (proto_rf_boxes 기준)
                    #    - overlayed_rf_img_j: 위 heatmap 합성 이미지에서 동일 영역을 잘라 시각화
                    #    - rf가 전체 이미지를 덮을 수도 있다 (stride나 receptive field 계산이 전체 프레임으로 역산될 때);
                    #      이런 경우에는 원본과 동일한 이미지를 중복 저장할 필요가 없어 이 조건이 false가 된다.
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        # overlayed_original_img_j는 원본+heatmap 합성 이미지이므로,
                        # rf_prototype_j의 (y1, y2, x1, x2) 좌표를 그대로 사용해 동일 영역만 잘라낸다.
                        # -> overlayed_rf_img_j shape: [rf_h, rf_w, 3], heatmap이 얹힌 리셉티브필드 하이라이트
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)

                    # ④ proto_img_j: find_high_activation_crop이 반환한 bbox로 잘라낸 고활성 patch
                    #    - slot 선택과 무관하게, 실제로 가장 높은 유사도를 보인 부분만 강조하는 최종 시각화 결과
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)
                
    if class_specific:
        del class_to_img_index_dict

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader,
                    prototype_network_parallel,
                    class_specific=True,
                    preprocess_input_function=None,
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None,
                    epoch_number=None,
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,
                    log=print,
                    prototype_activation_function_in_numpy=None):
    """입력/출력과 절차

    - 입력
        * `dataloader`: push용 dataloader (정규화 전 이미지를 `[0,1]` 범위로 제공)
        * `prototype_network_parallel`: push 대상 모델 (CUDA)
        * `class_specific`: True면 프로토타입 클래스와 동일한 샘플만 탐색
        * 기타: 전처리 함수, stride, 저장 경로/파일 prefix 등 부가 옵션
    - 출력: 없음. 내부에서 `prototype_vectors`와 기록용 배열을 갱신하고, 필요한 파일을 저장한다.
    - 절차
        1. 글로벌 최소 거리 배열을 초기화하고 push 결과를 저장할 디렉터리를 준비한다.
        2. dataloader 배치를 순회하면서 `update_prototypes_on_batch`를 호출해 최소 거리를 갱신한다.
        3. 모든 배치 처리가 끝나면 `global_min_fmap_patches`를 토치 텐서로 변환해 `prototype_vectors`에 복사한다.
    """

    prototype_network_parallel.eval()
    log('\tpush')

    # push 단계 전체 시간 측정을 위해 시작 시각 기록
    start = time.time()
    prototype_shape = prototype_network_parallel.prototype_shape  # (num_proto, feat_dim, proto_h, proto_w)
    n_prototypes = prototype_network_parallel.num_prototypes

    # --- (1) 글로벌 최소 거리/특징 패치 버퍼 초기화 ---
    #     - batch마다 numpy 배열을 갱신하고, 마지막에 torch 텐서로 변환해 proto 벡터를 교체한다.
    #     - `global_min_proto_dist`는 프로토타입별 최소 거리 값(스칼라)을 누적, 초기값은 +∞.
    #     - `global_min_fmap_patches`는 대응하는 특징 패치(`feat_dim × proto_h × proto_w`)를 저장.
    global_min_proto_dist = np.full(n_prototypes, np.inf) # [num_proto], 초기값 +∞
    global_min_fmap_patches = np.zeros( 
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1) # [num_proto, 6], (img_idx, y1, y2, x1, x2, label)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1) # [num_proto, 6], (img_idx, y1, y2, x1, x2, label)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    # --- (2) push 결과를 저장할 epoch 디렉터리 설정 ---
    #     - `root_dir_for_saving_prototypes`가 지정되면, (옵션) epoch별 하위 폴더를 만든다.
    #     - None이면 저장을 생략하고 경로 인자를 update 함수에 전달하지 않는다.
    if root_dir_for_saving_prototypes is not None:
        if epoch_number is not None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    # dataloader는 push 전용 이미지(정규화 전)를 배치 단위로 제공한다.
    search_batch_size = dataloader.batch_size
    num_classes = prototype_network_parallel.num_classes

    # --- (3) dataloader 순회: 배치마다 최소 거리/시각화 정보 갱신 ---
    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        # start_index_of_search_batch: 전체 push 데이터셋에서 현재 배치가 차지하는 전역 인덱스 오프셋
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir is not None and proto_bound_boxes_filename_prefix is not None:
        # --- (4) push 결과로 얻은 bbox 메타데이터 저장 ---
        #     - receptive field/activation bbox 모두 numpy로 남겨 후처리나 해석 스크립트에서 재사용한다.
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    # --- (5) numpy → torch 변환 후 실제 prototype 벡터 갱신 ---
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    # --- (6) 실행 시간 로깅 ---
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))
