import torch
import numpy as np

import heapq

import matplotlib.pyplot as plt
import os
import time

import cv2
from helpers import makedir

def save_prototype_original_img_with_bbox(save_dir,
                                          img_rgb,
                                          sub_patches,
                                          bound_box_j,
                                          color=(0, 255, 255)):
    """슬롯별 bbox 시각화를 파일로 저장.

    Parameters
    ----------
    save_dir : str
        결과 이미지를 저장할 경로.
    img_rgb : np.ndarray
        `[H, W, 3]` 범위의 float RGB 이미지 (0~1). push/find_nearest 과정에서 torch 텐서를 numpy로 변환한 값.
    sub_patches : int
        슬롯 개수 (`prototype_shape[-1]`). 색상 팔레트 길이보다 길면 순환한다.
    bound_box_j : np.ndarray
        shape `[5(~6), sub_patches]`. 각 슬롯의 (y1, y2, x1, x2) 픽셀 좌표와 이미지 index 등을 담고, 비활성 슬롯은 -1.
    color : tuple
        기본 색상(BGR). 슬롯별 팔레트는 함수 내부에서 정의한다.
    """

    # matplotlib용 float RGB 이미지를 OpenCV BGR로 변환 후 bbox를 그린다.
    p_img_bgr = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)

    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for k in range(sub_patches):
        if bound_box_j[1, k] != -1:
            # -1이면 슬롯이 비활성화된 상태. 양수 값만 bbox로 사용한다.
            bbox_height_start_k = bound_box_j[1, k]
            bbox_height_end_k = bound_box_j[2, k]
            bbox_width_start_k = bound_box_j[3, k]
            bbox_width_end_k = bound_box_j[4, k]
            color = colors[k % len(colors)]  # 슬롯이 4개 초과면 팔레트 순환
            cv2.rectangle(p_img_bgr,
                          (bbox_width_start_k, bbox_height_start_k),
                          (bbox_width_end_k - 1, bbox_height_end_k - 1),
                          color,
                          thickness=2) # 두께 2 픽셀

    p_img_rgb = p_img_bgr[..., ::-1] # BGR → RGB
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(save_dir, p_img_rgb, vmin=0.0, vmax=1.0)

class ProtoImage:
    """프로토타입과 연관된 원본 이미지/슬롯 bbox 정보를 담는 힙 요소."""
    def __init__(self, bb_box_info,
                 label, activation,
                 original_img=None):
        self.bb_box_info = bb_box_info  # [5, num_slots]; 슬롯별 bbox 정보 (-1이면 미사용)
        self.label = label              # torch tensor → 저장 시 int로 변환됨
        self.activation = activation    # greedy_distance로 얻은 cosine 활성도 (값이 클수록 가까움)

        self.original_img = original_img
    def __lt__(self, other):
        # heapq는 __lt__ 비교를 사용하므로 activation 값으로 정렬된다.
        return self.activation < other.activation

    def __str__(self):
        return str(self.label) + str(self.activation)


class ImagePatch:
    """(사용 안 함) 패치 정보를 담는 구조체. 힙에서 코사인 활성 비교에 사용 가능."""

    def __init__(self, patch, label, activation,
                 original_img=None, act_pattern=None, patch_indices=None):
        self.patch = patch
        self.label = label
        self.activation = activation
        self.original_img = original_img
        self.patch_indices = patch_indices

    def __lt__(self, other):
        return self.activation < other.activation
    
class ImagePatchInfo:
    """(사용 안 함) 최소 정보 패치 구조체. activation 비교 연산만 제공."""

    def __init__(self, label, activation):
        self.label = label
        self.activation = activation

    def __lt__(self, other):
        return self.activation < other.activations

# find the nearest patches in the dataset to each prototype
def find_k_nearest_patches_to_prototypes(dataloader,
                                         ppnet,
                                         num_nearest_neighbors=5,
                                         preprocess_input_function=None,
                                         root_dir_for_saving_images='./nearest',
                                         log=print,
                                         prototype_layer_stride=1):
    """프로토타입별 최근접 패치를 찾아 저장한다.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        push 이후의 이미지 배치. 정규화 전 RGB 텐서를 `[0,1]` 범위로 제공해야 한다.
    ppnet : PPNet
        슬롯 구조를 포함한 ProtoViT 모델. `push_forward`와 `forward`(greedy_distance)를 활용한다.
    num_nearest_neighbors : int, default=5
        프로토타입당 저장할 최근접 이미지 수 K.
    preprocess_input_function : callable, optional
        datalo더에서 받은 이미지를 push용으로 정규화하는 함수 (예: Imagenet 평균/표준편차).
    root_dir_for_saving_images : str, default='./nearest'
        프로토타입별 최근접 이미지를 저장할 루트 폴더. `/<proto_id>/nearest-i_*.png`로 저장된다.
    log : callable, default=print
        진행 상황을 출력할 함수.
    prototype_layer_stride : int, default=1
        feature grid에서 픽셀 좌표로 변환할 때 사용하는 stride (ViT 백본은 1).

    Returns
    -------
    labels_all_prototype : np.ndarray
        shape `[num_proto, num_nearest_neighbors]`. 프로토타입별로 찾은 K개 이미지의 클래스 라벨.

    Notes
    -----
    1. push_forward를 이용해 슬롯별 patch index(0~195)를 얻고, greedy_distance가 반환하는 cosine 활성도를 사용한다.
    2. 각 프로토타입마다 힙(heapq)을 유지해 활성도가 높은 K개의 샘플만 저장한다.
    3. 힙을 정렬한 뒤 원본 이미지와 슬롯 bbox overlay를 PNG로 저장하고, 라벨 정보를 `class_id.npy`에 기록한다.
    """
    ppnet.eval()  # 분석 단계이므로 dropout/BN을 평가 모드로 고정
    log('find nearest patches')
    n_prototypes = ppnet.num_prototypes  # 전체 프로토타입 수 (예: 2000)
    prototype_shape = ppnet.prototype_shape  # (num_proto, feat_dim, num_slots)
    # 프로토타입마다 하나씩 max-heap을 유지 (heapq는 min-heap => activation 부호는 그대로, heapq가 작은 값부터 제거)
    heaps = [[] for _ in range(n_prototypes)]

    for index, (search_batch_input, search_y) in enumerate(dataloader):
        log(f'batch {index}')
        with torch.no_grad():
            # (선택) push용 전처리가 지정된 경우 적용 후 GPU로 이동
            if preprocess_input_function is not None:
                search_batch = preprocess_input_function(search_batch_input)
                search_batch = search_batch.cuda()
            else:
                search_batch = search_batch_input.cuda()
            n_p = prototype_shape[2]  # 슬롯 개수 (예: 4)
            slots_torch_raw = torch.sigmoid(ppnet.patch_select * ppnet.temp)  # [1, num_proto, num_slots]
            proto_slots = np.copy(slots_torch_raw.detach().cpu().numpy())      # numpy 사본 (CPU 기록용)
            factor = slots_torch_raw.sum(-1, keepdim=True) + 1e-10            # 슬롯 확률 합 (0 방지 epsilon)

            # push_forward: 특징맵/거리/슬롯 index를 반환 → greedy_distance 기반이라 slots와 동일
            protoL_input_torch, proto_dist_torch, proto_indices_torch = ppnet.push_forward(search_batch) # [B, feat_dim, 14, 14], [B, num_proto, num_slots], [B, num_proto, num_slots]
            _, _, values = ppnet(search_batch)  # forward: greedy_distance → evidence 출력과 슬롯 활성
        # 슬롯 확률을 합이 n_p가 되도록 보정한 후, greedy 활성값에 곱해 실제 활성도로 사용
        values_slot = values.clone() * (slots_torch_raw * n_p / factor)  # [B, num_proto, num_slots]
        cosine_act = values_slot.sum(-1)  # [B, num_proto]; ProtoPNet-style 활성도

        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())          # [B, num_proto]
        proto_indice_ = np.copy(proto_indices_torch.detach().cpu().numpy())     # [B, num_proto, num_slots]
        del protoL_input_torch, proto_dist_torch, proto_indices_torch, slots_torch_raw

        for b_idx, indices_b in enumerate(proto_indice_):
            cosine_act_b = cosine_act[b_idx] # [num_proto]
            # dataloader는 정규화 전 이미지를 제공하므로 복사해 HWC float로 변환.
            original_img = search_batch_input[b_idx].detach().cpu().numpy()
            original_img = np.transpose(original_img, (1, 2, 0))  # [H, W, 3]

            for j in range(n_prototypes):
                cosine_act_j = cosine_act_b[j]           # 프로토타입 j에 대한 활성도 스칼라
                indices_j = indices_b[j]                 # 슬롯마다 선택된 patch index (0~195)
                proto_slots_j = (proto_slots.squeeze())[j]  # [num_slots]; 0/1 슬롯 마스크
                proto_bound_boxes = np.full(shape=[5, n_p], fill_value=-1)  # (img_idx, y1, y2, x1, x2)

                min_j_indice = np.unravel_index(indices_j.astype(int), (14, 14))  # grid 좌표 (y, x)
                grid_width = 16
                for k in range(n_p):
                    if proto_slots_j[k] != 0: # 슬롯 k가 활성화된 경우에만 bbox 계산
                        fmap_height_start_index_k = min_j_indice[0][k] * prototype_layer_stride # 14x14 grid 좌표 → feature map 좌표
                        fmap_height_end_index_k = fmap_height_start_index_k + 1 # 1x1 patch이므로 +1
                        fmap_width_start_index_k = min_j_indice[1][k] * prototype_layer_stride
                        fmap_width_end_index_k = fmap_width_start_index_k + 1
                        bound_idx_k = np.array([[fmap_height_start_index_k, fmap_height_end_index_k],
                                                [fmap_width_start_index_k, fmap_width_end_index_k]]) # feature map 좌표
                        pix_bound_k = bound_idx_k * grid_width  # 14x14 grid → 224 픽셀 환산
                        proto_bound_boxes[0] = j
                        proto_bound_boxes[1, k] = pix_bound_k[0][0]
                        proto_bound_boxes[2, k] = pix_bound_k[0][1]
                        proto_bound_boxes[3, k] = pix_bound_k[1][0]
                        proto_bound_boxes[4, k] = pix_bound_k[1][1]

                highest_patch = ProtoImage(bb_box_info=proto_bound_boxes,
                                            label=search_y[b_idx],
                                            activation=cosine_act_j,
                                            original_img=original_img)

                # 각 프로토타입 힙에 활성도가 큰 순으로 최대 K개 유지
                if len(heaps[j]) < num_nearest_neighbors:
                    # 아직 K개가 안 찼으면 그대로 push (heap은 activation 오름차순 유지)
                    heapq.heappush(heaps[j], highest_patch)
                else:
                    # heappushpop: 새 patch를 넣은 뒤, heap에서 activation이 가장 작은 요소를 pop
                    # → 새 patch가 더 크면 교체되고, 작으면 즉시 버려지므로 항상 상위 K개만 남는다.
                    heapq.heappushpop(heaps[j], highest_patch)

    # 전체 데이터셋 순회 후 힙에는 K개의 최근접 샘플이 남아 있음
    for j in range(n_prototypes):
        heaps[j].sort()            # activation 오름차순 정렬
        heaps[j] = heaps[j][::-1]  # 가장 큰 activation부터 앞에 놓이도록 역순 변환
        dir_for_saving_images = os.path.join(root_dir_for_saving_images, str(j)) # prototype j 폴더
        makedir(dir_for_saving_images)
        for i, patch in enumerate(heaps[j]):
            # save the original image where the patch comes from
            plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_original.png'),
                           arr=patch.original_img,
                           vmin=0.0,
                           vmax=1.0) # [H, W, 3] float RGB
            bb_dir = os.path.join(dir_for_saving_images, 'nearest-' + str(i) + '_patch_with_box.png')
            save_prototype_original_img_with_bbox(bb_dir,
                                                  patch.original_img,
                                                  sub_patches=n_p,
                                                  bound_box_j=patch.bb_box_info,
                                                  color=(0, 255, 255)) # BGR color
        labels = np.array([patch.label for patch in heaps[j]])  # 각 프로토타입의 top-K 이미지 라벨
        np.save(os.path.join(dir_for_saving_images, 'class_id.npy'), labels)

    labels_all_prototype = np.array([[patch.label for patch in heaps[j]] for j in range(n_prototypes)]) # [num_proto, K]
    np.save(os.path.join(root_dir_for_saving_images, 'full_class_id.npy'), labels_all_prototype) # 전체 프로토타입 라벨
            
    return labels_all_prototype
