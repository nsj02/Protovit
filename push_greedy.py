import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
import torch.nn.functional as F

from helpers import makedir, find_high_activation_crop

def save_prototype_original_img_with_bbox(dir_for_saving_prototypes,
                                          img_dir,
                                          prototype_img_filename_prefix,
                                          j,
                                          sub_patches,
                                          indices,
                                          bound_box_j,
                                          color=(0, 255, 255)):
    """입력/출력과 역할

    - 입력
        * `img_dir`: 원본 이미지 경로 (bbox를 덧씌울 대상)
        * `indices`: 슬롯별 패치 위치 index. `np.unravel_index` 결과로 `(row_idx, col_idx)` 2-tuple이며
          `np.vstack` 되어 shape `[2, sub_patches]` (14×14 grid 좌표).
        * `bound_box_j`: 슬롯별 픽셀 좌표 bbox. shape `[5, sub_patches]` (또는 클래스 레이블 포함 시 `[6, sub_patches]`)이며
          순서는 `[dataset_img_idx, y1, y2, x1, x2, (optional) class_id]`.
        * `sub_patches`: 선택된 슬롯 개수 (보통 `prototype_shape[-1]`).
    - 출력: 없음. bbox가 표시된 원본 이미지와 마스크 이미지를 디스크에 저장한다.
    - 역할: greedy push로 선택된 다수 슬롯을 서로 다른 색상(노랑/빨강/초록/파랑)으로 시각화한다.
    """
    save_dir = os.path.join(dir_for_saving_prototypes,
                 prototype_img_filename_prefix + 'bbox-original' + str(j) +'.png')
    p_img_bgr = cv2.imread(img_dir)
    img_bbox = p_img_bgr.copy()
    # cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
    #               color, thickness=2)
    # 슬롯 index 0~3까지 시각화에 사용할 BGR 색상 팔레트 (노랑, 빨강, 초록, 파랑 순).
    # sub_patches가 4보다 크면 색상이 재사용되므로 필요 시 팔레트를 확장할 것.
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # mask_val: 14x14 grid 전체를 0.4로 초기화 → 비활성 슬롯 영역은 어둡게(0.4배) 표시.
    # 선택된 슬롯 위치는 아래 루프에서 1로 올려 원본 밝기를 유지한다.
    mask_val = np.ones((14, 14)) * 0.4
    for k in range(sub_patches):
        if bound_box_j[1, k] != -1:
            # bound_box_j가 -1이면 해당 슬롯은 비활성화된 상태 → bbox/색상 갱신 생략.
            # indices: (row_idx, col_idx) grid 좌표 → 16x16 패치 기준 위치.
            x, y = indices[0][k], indices[1][k]
            mask_val[x, y] = 1  # 선택된 슬롯 위치는 밝기 1로 유지.
            bbox_height_start_k = bound_box_j[1, k]
            bbox_height_end_k = bound_box_j[2, k]
            bbox_width_start_k = bound_box_j[3, k]
            bbox_width_end_k = bound_box_j[4, k]
            color = colors[k % len(colors)]  # 슬롯이 4개 초과 시 색상 순환.
            #cv2.rectangle(img_bbox, (bbox_width_start_k, bbox_height_start_k), (bbox_width_end_k-1, bbox_height_end_k-1),
                    #color, thickness=1)
            cv2.rectangle(p_img_bgr, (bbox_width_start_k, bbox_height_start_k), (bbox_width_end_k-1, bbox_height_end_k-1),
                    color, thickness=2)
    # OpenCV는 BGR 순서를 사용하므로 RGB로 뒤집은 뒤 float32(0~1) 범위로 변환
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(save_dir, p_img_rgb, vmin=0.0, vmax=1.0)  # bbox가 그려진 원본 이미지를 저장
    size = p_img_rgb.shape[1]  # 이미지 한 변(픽셀), 패치 크기 계산에 사용
    
    img_bbox_rgb = np.clip(img_bbox + 150, 0, 255)# increase the brightness
    img_bbox_rgb = img_bbox[...,::-1]
    img_bbox_rgb = np.float32(img_bbox_rgb) / 255
    width = size//14
    
    #bb_og = p_img_rgb.copy()
    for i in range(0, 196):
        x = i %14
        y = i//14
        img_bbox_rgb[y*width:(y+1)*width, x*width:(x+1)*width]*=mask_val[y,x]
        #bb_og[y*width:(y+1)*width, x*width:(x+1)*width]*=mask_val[y,x]

    save_dir2 = os.path.join(dir_for_saving_prototypes,
                 prototype_img_filename_prefix + '_vis_' + str(j) +'.png')
    plt.imsave(save_dir2, img_bbox_rgb,vmin=0.0,vmax=1.0)
    
    #save_dir3 = os.path.join(dir_for_saving_prototypes,
                 #prototype_img_filename_prefix + '_vis_bb_' + str(j) +'.png')
    
    #plt.imsave(save_dir3, img_bbox_rgb,vmin=0.0,vmax=1.0)
    #for k in range()
    


def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               pnet,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
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
    """입력/출력과 핵심 절차 (슬롯 기반 push)

    - 입력
        * `search_batch_input`: `[batch, 3, H, W]` push 배치 (정규화 전)
        * `pnet`: 슬랏이 있는 `PPNet` (ProtoViT) 모델
        * `global_min_proto_dist`: `[num_proto]` numpy 배열, 최소 거리 저장소
        * `global_min_fmap_patches`: `[num_proto, feat_dim, num_slots]` numpy 배열, 슬롯별 최소 특징 벡터
        * `proto_bound_boxes`: `[num_proto, 5~6, num_slots]` numpy 배열, 픽셀 bbox 기록
        * 나머지 인자: 클래스 전용 탐색, 전처리, stride, 저장 옵션 등
    - 출력: 없음 (입력 numpy 배열/딕셔너리를 in-place 갱신)
    - 절차
        1. push_forward로 (특징맵, 거리, 슬롯별 patch index) 텐서를 얻는다.
        2. 클래스별 혹은 전체 배치에서 프로토타입별 최소 거리를 찾는다.
        3. 슬랏 단위로 선택된 patch index(`proto_indices_torch`)를 이용해 `global_min_fmap_patches`와 bbox를 갱신한다.
        4. (옵션) 원본 이미지와 bbox 시각화를 파일로 저장한다.
    """
    pnet.eval()
    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    with torch.no_grad():
        search_batch = search_batch.cuda()
    # push_forward는 (1) conv 특징맵, (2) 프로토타입별 최소거리, (3) 선택된 서브패치 index를 반환한다.
    protoL_input_torch, proto_dist_torch, proto_indices_torch = pnet.push_forward(search_batch)
    slots_torch_raw = torch.sigmoid(pnet.patch_select * pnet.temp)
    # 슬롯 시그모이드값을 0.1 단위로 반올림해 미세 노이즈를 제거 (대부분 0/1 근처로 정리)
    slots_torch = torch.round(slots_torch_raw, decimals=1)
    proto_slots = np.copy(slots_torch.detach().cpu().numpy())
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())      # [B, feat_dim, h, w]
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())          # [B, num_proto]
    proto_indice_ = np.copy(proto_indices_torch.detach().cpu().numpy())    # [B, num_proto, num_slots]
    del protoL_input_torch, proto_dist_torch, proto_indices_torch, slots_torch, slots_torch_raw

    if class_specific:
        # 배치 내 클래스별 이미지 index를 모아, 각 프로토타입이 자신의 클래스 샘플만 검색하도록 한다.
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # 예: search_y = tensor([2, 0]) → {0: [1], 1: [], 2: [0], ...}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)
    prototype_shape = pnet.prototype_shape  # (num_proto, feat_dim, num_slots)
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    # 슬롯 수 (원본 코드에서 proto_h를 slot 개수로 사용) -> 여기서는 1x1 슬랏 구성 가정
    n_p = proto_h
    for j in range(n_prototypes):
        # --- 프로토타입 j에 대해 greedy 슬롯 선택 ---
        if class_specific:
            # prototype_class_identity[j]의 argmax → 이 프로토타입이 담당하는 클래스 ID
            target_class = torch.argmax(pnet.prototype_class_identity[j]).item()
            # 현재 배치에 해당 클래스 이미지가 없으면 탐색 스킵
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            # 해당 클래스에 속한 샘플의 거리 맵만 추출 `[num_img_cls, num_proto]`
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j]
        else:
            # class_specific=False → 배치 전체 `[batch, num_proto]`에서 거리 탐색
            # ⚠️ 주의: 이 모드에서는 특정 클래스의 프로토타입이라도 가장 가까운 패치가 다른 클래스 이미지일 수 있음
            #          → projection 후 해당 프로토타입이 엉뚱한 클래스 패치로 대체될 위험이 있다.
            proto_dist_j = proto_dist_[:, j] # [batch]

        # --- 프로토타입 j의 현재 배치 최소 거리 계산 ---
        batch_min_proto_dist_j = np.amin(proto_dist_j) # 배치 내 최소 거리 [1]
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            # np.unravel_index로 (이미지 index, slot index) 등 최소 거리가 발생한 위치 복원
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape)) # [1], index in the batch
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                batch_argmin_proto_dist_j, the index of closest img to p_j
                min_j_indice, the indices of the sub-part of p_j on the closet img
                '''
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]] # class -> entire search batch
            # retrieve the corresponding feature map
            batch_argmin_j_patch_indices = proto_indice_[batch_argmin_proto_dist_j, j][0]  # [n_p] slot별 선택된 patch index
            #batch_argmin_j_patch_subvalues = protot_subvalues[batch_argmin_proto_dist_j, j][0]
            proto_slots_j = (proto_slots.squeeze())[j]  # [n_p]; 0/1 슬롯 활성
            min_j_indice = np.unravel_index(batch_argmin_j_patch_indices.astype(int), (14, 14)) # [2, n_p]; slot별 (row_idx, col_idx) grid 좌표
            img_index_in_batch = batch_argmin_proto_dist_j[0] # 최소 거리 이미지의 배치 내 index
            global_min_proto_dist[j] = batch_min_proto_dist_j # update the global min distance
            # get the whole image 
            original_img_j = search_batch_input[batch_argmin_proto_dist_j[0]].numpy() # [3, H, W]
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            grid_width = 16
            # 슬롯별로 선택된 grid 위치를 특징 패치 및 bbox에 반영
            for k in range(n_p):
                if proto_slots_j[k] != 0:  # 활성화된 슬롯만 유지
                    # min_j_indice는 14x14 feature grid 좌표 → 먼저 feature map index 범위로 변환한다.
                    # prototype_layer_stride는 ViT 백본에서 1이라 결과가 그대로지만, stride가 바뀌면 여기서 반영된다.
                    # 실제 픽셀 좌표는 아래에서 grid_width(=16)를 곱해 계산하므로 여기서는 곱하지 않는다.
                    fmap_height_start_index_k = min_j_indice[0][k] * prototype_layer_stride # 14x14 grid -> feature map index
                    fmap_height_end_index_k = fmap_height_start_index_k + 1
                    fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
                    fmap_width_end_index_k = fmap_width_start_index_k + 1

                    batch_min_fmap_patch_j_k = protoL_input_[img_index_in_batch,
                                                            :,
                                                            fmap_height_start_index_k:fmap_height_end_index_k,
                                                            fmap_width_start_index_k:fmap_width_end_index_k] # [feat_dim, 1, 1]
                    global_min_fmap_patches[j, :, k] = batch_min_fmap_patch_j_k.squeeze(-1).squeeze(-1) # update the global min fmap patch

                    bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                                             [fmap_width_start_index_k, fmap_width_end_index_k]])
                    pix_bound_k = bound_idx_k * grid_width
                    proto_img_j_k = original_img_j[bound_idx_k[0][0]:bound_idx_k[0][1],
                                                   bound_idx_k[1][0]:bound_idx_k[1][1], :]
                    proto_bound_boxes[j, 0, k] = batch_argmin_proto_dist_j[0] + start_index_of_search_batch
                    proto_bound_boxes[j, 1, k] = pix_bound_k[0][0]
                    proto_bound_boxes[j, 2, k] = pix_bound_k[0][1]
                    proto_bound_boxes[j, 3, k] = pix_bound_k[1][0]
                    proto_bound_boxes[j, 4, k] = pix_bound_k[1][1]
                    if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                        proto_bound_boxes[j, 5, k] = search_y[batch_argmin_proto_dist_j[0]].item()
            # start saving images 
            if dir_for_saving_prototypes is not None:
                if prototype_img_filename_prefix is not None:
                    original_img_path = os.path.join(dir_for_saving_prototypes,
                            prototype_img_filename_prefix + '-original' + str(j) + '.png')
                    plt.imsave(original_img_path,
                    original_img_j,
                    vmin=0.0,
                    vmax=1.0)
            # rt = os.path.join(dir_for_saving_prototypes,
            #                 prototype_img_filename_prefix + 'bbox-original' + str(j) +'.png')
            save_prototype_original_img_with_bbox(dir_for_saving_prototypes, original_img_path,prototype_img_filename_prefix,j = j,
                                                  sub_patches = n_p,
                                                  indices = min_j_indice,
                                                  bound_box_j = proto_bound_boxes[j], color=(0, 255, 255))
            # rt_newvis = os.path.join(dir_for_saving_prototypes,
            #                 prototype_img_filename_prefix + '_newvis_' + str(j) +'.png')
            # proto_new_vis(rt_newvis, original_img_path,sub_patches= n_p,
            #                             slots = proto_slots_j,
            #                             indices = min_j_indice,
            #                             bound_box_j = proto_bound_boxes[j], color=(0, 255, 255))
            
    return None 



# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader,
                    pnet,
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
    """슬롯 기반 push 절차

    파라미터
    ---------
    dataloader : torch.utils.data.DataLoader
        push 단계에서 사용할 (정규화 전) 이미지 배치 공급자. `batch_size`는 전역 index 계산에 사용된다.
    pnet : PPNet
        슬롯 정보를 가진 ProtoViT 모델. `push_forward`/`patch_select` 등을 통해 프로토타입을 교체한다.
    class_specific : bool, 기본값 True
        True이면 각 프로토타입이 담당하는 클래스의 이미지에서만 최소 거리를 탐색한다.
        False로 두면 다른 클래스 이미지 패치로 덮어씌워질 수 있으므로 주의.
    preprocess_input_function : callable, optional
        push 전 입력 이미지를 정규화하는 함수. `update_prototypes_on_batch` 내부로 전달된다.
    prototype_layer_stride : int, 기본값 1
        특징맵 stride. 14×14 grid index를 feature map 좌표로 변환할 때 사용된다.
    root_dir_for_saving_prototypes : str, optional
        프로토타입 시각화/메타데이터를 저장할 루트 경로. None이면 저장을 생략한다.
    epoch_number : int, optional
        저장 시 `epoch-{n}` 하위 폴더를 생성해 버전을 구분한다.
    prototype_img_filename_prefix / prototype_self_act_filename_prefix : str, optional
        이미지/활성맵 파일 이름 prefix.
    proto_bound_boxes_filename_prefix : str, optional
        슬롯별 bbox numpy 파일 prefix. 지정되면 push 종료 후 저장한다.
    save_prototype_class_identity : bool
        True이면 bbox 배열에 클래스 ID 열을 추가해 기록한다.
    log : callable
        진행 로그 출력 함수 (기본 `print`).
    prototype_activation_function_in_numpy : callable, optional
        활성도 변환 함수. `update_prototypes_on_batch`로 전달된다.

    반환값
    -------
    없음. 함수는 `pnet.prototype_vectors`를 in-place로 갱신하고, 필요한 파일을 디스크에 저장한다.

    동작 요약
    ---------
    1. `global_min_proto_dist`/`global_min_fmap_patches` 버퍼를 초기화하고, slot 시그모이드를 +/-200으로 이진화한다.
    2. dataloader를 순회하면서 `update_prototypes_on_batch`를 호출해 슬롯별 최소 거리/패치/BBox를 누적한다.
    3. (옵션) bbox/이미지 아티팩트를 저장하고, 최종적으로 `prototype_vectors`를 numpy→torch로 복사해 push를 완료한다.
    """
    pnet.eval()
    log('\tpush')
    start = time.time()
    prototype_shape = pnet.prototype_shape  # (num_proto, feat_dim, num_slots)
    n_prototypes = pnet.num_prototypes # prototype 개수
    global_min_proto_dist = np.full(n_prototypes, np.inf) # 각 프로토타입별 최소 거리 저장소
    n_p = prototype_shape[2]
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         n_p])

    # 슬롯 indicator를 확실한 0/1 근처(±200)로 강제해 push 도중 slot pruning 상태를 고정한다.
    slots = torch.sigmoid(pnet.patch_select * pnet.temp).clone()
    slots_rounded = slots.round()  # 연속값(0~1)을 반올림해 0/1 근처로 정리
    result_tensor = torch.where(slots_rounded == 0,
                                torch.tensor(-1),
                                slots_rounded) * 200  # 0 슬롯 -> -200, 1 슬롯 -> +200 (sigmoid 후 ≈0/1)
    pnet.patch_select.data.copy_(torch.tensor(result_tensor.detach().cpu().numpy(), dtype=torch.float32).cuda())
    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    ex_dim:sub_patch component index
    '''
    if save_prototype_class_identity:
        proto_bound_boxes = np.full(shape=[n_prototypes, 5, n_p], fill_value=-1)
    else:
        proto_bound_boxes = np.full(shape=[n_prototypes, 4, n_p], fill_value=-1)
    if root_dir_for_saving_prototypes is not None:
        if epoch_number is not None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    num_classes = pnet.num_classes
    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        # 전역 이미지 index 오프셋 (0-based). bbox에 데이터셋 전체 index를 기록할 때 사용.
        start_index_of_search_batch = push_iter * search_batch_size
        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   pnet,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
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
    
    # push 이후 슬롯 활성 개수 통계를 출력해 pruning 결과를 빠르게 확인한다.
    slots_pushed = torch.sigmoid(pnet.patch_select * pnet.temp).squeeze(1).sum(-1) # [num_proto]
    unique_elements, counts = torch.unique(slots_pushed, return_counts=True) # 슬롯 활성 개수별 프로토타입 개수
    counter = dict(zip(unique_elements.tolist(), counts.tolist())) # {활성 슬롯 개수: 프로토타입 개수}
    log(str(counter))

    if proto_epoch_dir is not None and proto_bound_boxes_filename_prefix is not None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes) # save the bounding boxes

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape)) # [num_proto, feat_dim, n_p]
    # push prototype to latent feature 
    pnet.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())  # update the prototype vectors
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))
    return None 
