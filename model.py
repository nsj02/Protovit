"""ProtoViT 모델 정의 및 greedy slot 기반 프로토타입 갱신 로직."""

from xml.etree.ElementPath import xpath_tokenizer_re
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from tools.deit_features import deit_tiny_patch_features, deit_small_patch_features
from tools.cait_features import cait_xxs24_224_features

base_architecture_to_features = {'deit_small_patch16_224': deit_small_patch_features,
                                 'deit_tiny_patch16_224': deit_tiny_patch_features,
                                 #'deit_base_patch16_224':deit_base_patch16_224,
                                 'cait_xxs24_224': cait_xxs24_224_features,}

class PPNet(nn.Module):
    """ProtoViT/ProtoPNet 변형: ViT 백본 + 슬롯 기반 프로토타입을 묶어 예측 및 해석 제공."""

    def __init__(self, features, img_size, prototype_shape,
                 num_classes, init_weights=True,
                 prototype_activation_function='log',
                 sig_temp = 1.0,
                 radius = 3,
                 add_on_layers_type='bottleneck'):
        """생성자 매개변수 요약

        - `features`: timm/hf ViT 백본 모듈
        - `img_size`: 입력 이미지 한 변 크기 (예: 224)
        - `prototype_shape`: `(num_proto, feature_dim, num_slots)`
        - `num_classes`: 데이터셋 클래스 수
        - `prototype_activation_function`: 거리->유사도 변환 함수 타입
        - `sig_temp`: 슬롯 선택 시그모이드 temperature
        - `radius`: greedy 슬롯 선택에서 반경 제약
        """

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape # (num_prototypes, feature_dim, num_slots)
        self.num_prototypes = prototype_shape[0] # 총 프로토타입 수 = 클래스당 프로토타입 × 클래스 수
        self.num_classes = num_classes
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.epsilon = 1e-4
        self.normalizer = nn.Softmax(dim=1)
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function
        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        # 클래스별 균등 할당: 프로토타입 수가 클래스 수의 배수인지 확인하고 one-hot 행렬 생성.
        assert self.num_prototypes % self.num_classes == 0
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            # 프로토타입 j는 (j // num_prototypes_per_class) 클래스에 속한다.
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        #self.proto_layer_rf_info = proto_layer_rf_info

        # features 모듈은 self.features 이름으로 유지해야 사전학습 가중치 로딩과 호환된다.
        self.features = features

        # learnable 프로토타입 슬롯 텐서: 초기엔 랜덤, 이후 학습/푸시 단계에서 갱신됨.
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        self.radius = radius # greedy 선택 시 인접 패치 범위를 제어하는 radius
        # 슬롯 선택 파라미터 v: sigmoid(v * temp) -> [0,1] 근사 indicator
        self.patch_select = nn.Parameter(torch.ones(1, prototype_shape[0], prototype_shape[-1]) * 0.1,
                                         requires_grad=True)
        self.temp = sig_temp  # 슬롯 indicator를 hard하게 만들지 soft하게 만들지 결정하는 온도(temperature)
        # (미사용) 상수 텐서. 과거 슬롯 스케일링 실험 흔적으로 보이며 현재 로직에서는 참조되지 않는다.
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        # 마지막 evidence layer: 각 프로토타입의 출력(logits)을 클래스에 매핑 (bias 없음)
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False)
        features_name = str(features).upper()
        # 전달받은 ViT 백본 모듈의 문자열 표현으로 아키텍처 구분 (deit/cait).
        # 최신 timm 버전에서는 이름이 다를 수 있으니 필요하면 isinstance 체크 등으로 교체 권장.
        if features_name.startswith('VISION'):
            self.arc = 'deit'
        elif features_name.startswith('CAIT'):
            self.arc = 'cait'
        if init_weights:
            self._initialize_weights()  # ProtoPNet 방식으로 evidence layer 초기화

    def conv_features(self, x):
        """
        입력: x [B, 3, img_size, img_size]
        출력: feature_emb [B, feature_dim, 14, 14]

        1. timm PatchEmbed (16x16 패치, stride=16 conv) -> [B, 196, dim]
        2. 학습된 cls 토큰을 붙이고 위치 임베딩/드롭아웃/트랜스포머 블록 통과
        3. DeiT는 블록에서 cls 토큰과 패치를 함께 업데이트, CaiT는 cls-only 블록으로 한 번 더 갱신
        4. cls 토큰(전역 표현)을 제외한 196개 패치에서 cls 토큰을 빼 전역 대비 변화를 추출
        5. 패치 길이 196 = 14x14 격자이므로 conv feature map으로 reshape
        """
        x = self.features.patch_embed(x)
        # timm의 PatchEmbed 모듈은 Conv2d로 16x16 영역을 펼쳐 feature_dim 벡터로 선형 투영한다.
        # 결과 shape: [B, num_patches(=14*14), feature_dim]
        cls_token = self.features.cls_token.expand(x.shape[0], -1, -1)
        # cls_token은 학습 가능한 [1,1,dim] 파라미터이므로 배치 크기에 맞춰 복제해서 [B,1,dim]으로 맞춘다.
        if self.arc == 'deit':
            '''
            DeiT 백본 forward 경로 처리
            '''
            x = torch.cat((cls_token, x), dim=1)
            # cls 토큰 + 패치 토큰 연결 -> [B, 197, dim]
            x = self.features.pos_drop(x + self.features.pos_embed)
            # 학습된 1D position embedding을 더한 뒤 dropout 적용
            x = self.features.blocks(x)
            # DeiT 트랜스포머 블록(stack) 통과; self-attention/MLP 반복으로 패치+cls 동시 업데이트
            x = self.features.norm(x)
            # 마지막 LayerNorm -> [B, 197, dim]

        elif self.arc == 'cait':
            """
            CaiT 백본 forward 경로 처리
            """
            x = x + self.features.pos_embed
            # CaiT는 먼저 patch token에만 position embedding을 더한다 -> [B,196,dim]
            x = self.features.pos_drop(x)
            for blk in self.features.blocks:
                x = blk(x)
                # patch 토큰 전용 블록: cls 토큰 없이 패치 토큰들 간 self-attention 진행
            for blk in self.features.blocks_token_only:
                cls_token = blk(x, cls_token)
                # CaiT의 클래스-전용 블록: 패치 요약 정보를 cls 토큰에 모으는 cross-attention 단계
            x = torch.cat((cls_token, x), dim=1)
            # 최종적으로 cls 토큰을 다시 앞에 붙여 [B,197,dim]
            x = self.features.norm(x)
            # LayerNorm 공유

        # cls 토큰(전체 이미지 표현)을 각 패치에서 빼주어 "전역과의 차이"를 강조한다.
        # x[:, 0]: [B, dim] -> unsqueeze(1)로 [B,1,dim] 만들어 브로드캐스트
        x_2 = x[:, 1:] - x[:, 0].unsqueeze(1)
        # x[:,1:]: cls 제외 패치 196개 -> [B,196,dim], 따라서 x_2도 동일 shape
        fea_len = x_2.shape[1]
        B = x_2.shape[0]
        fea_width = fea_height = int(fea_len ** 0.5)  # 패치 격자의 한 변에 14개씩 배치된다고 가정 (img_size 224 기준)
        feature_emb = x_2.permute(0, 2, 1).reshape(B, -1, fea_width, fea_height)
        # permute 후 reshape -> [B, feature_dim, 14, 14]; 이후 conv 연산과 시각화에 사용
        return feature_emb
    
    def _cosine_convolution(self, x):
        """입력/출력 shape과 기능

        - 입력 `x`: `[batch, feature_dim, H, W]` float tensor (보통 `[B, dim, 14, 14]`)
        - 출력: `[batch, num_proto, H, W]` float tensor, dtype은 입력과 동일(기본은 float32)
        - 기능: 각 위치의 feature 벡터와 프로토타입 슬롯 벡터의 코사인 유사도를 계산한 뒤 부호를 뒤집어 "거리"로 반환
        - 주의: 슬롯 축이 1보다 클 때는 greedy_distance와 같이 슬롯별로 conv를 수행해야 하므로,
                이 함수는 projection 단계처럼 단일 슬롯(view=[num_proto, dim, 1, 1]) 상황에서 주로 사용
        """

        # conv 전에 채널 방향으로 L2 정규화해 모든 위치가 단위 벡터가 되도록 맞춘다.
        x = F.normalize(x, p=2, dim=1)  # [batch, dim, H, W]
        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)  # [num_proto, dim, num_slots]
        # F.conv2d는 weight를 [out_channels, in_channels, kH, kW]로 기대하므로
        # 슬롯 축이 1인 상태(예: projection 이후)에서는 `[num_proto, dim, 1, 1]` view로 동작한다.
        # 이렇게 계산된 값은 정규화된 dot product이므로 곧 코사인 유사도; 여기서는 부호만 반전해 "거리"로 쓴다.
        # 다중 슬롯일 때는 greedy_distance처럼 슬롯별 conv를 개별 처리해야 한다는 점에 유의.
        distances = F.conv2d(input=x, weight=now_prototype_vectors)  # [batch, num_proto, H, W] (slots=1 가정)
        # ProtoPNet 규약에 맞춰 "거리"로 사용하기 위해 부호를 뒤집는다. (유사도 높음 -> 거리는 작음)
        distances = -distances  # [batch, num_proto, H, W]

        return distances
    
    def _project2basis(self, x):
        """프로토타입 슬롯을 basis로 보고 정규화된 내적(=코사인 유사도)을 그대로 반환."""

        x = F.normalize(x, p=2, dim=1)  # [batch, dim, H, W]
        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)  # [num_proto, dim, num_slots]
        distances = F.conv2d(input=x, weight=now_prototype_vectors)  # [batch, num_proto, H, W]
        # 여기서는 부호를 유지해 양의 코사인 유사도를 그대로 사용한다. (push 단계 등에서 활용)
        return distances
    
    def prototype_distances(self, x):
        """주어진 배치에 대해 (projection 값, 거리 맵)을 함께 반환."""

        conv_features = self.conv_features(x)
        cosine_distances = self._cosine_convolution(conv_features)
        project_distances = self._project2basis(conv_features)
        return project_distances, cosine_distances
    
    def global_min_pooling(self, distances):
        """입력/출력과 용도

        - 입력 `distances`: `[batch, num_proto, H, W]` float tensor (ProtoNet distance map)
        - 출력: `[batch, num_proto]` float tensor, 각 프로토타입별로 최솟값만 남김
        - 기능: 공간 차원(H, W)에서 최소 거리를 뽑아 이미지마다 프로토타입별 핵심 거리 통계를 만든다
        """

        # 음수로 뒤집은 뒤 max pooling을 쓰면 min pooling 효과가 난다.
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        # shape: [batch, num_proto, 1, 1] -> [batch, num_proto]
        min_distances = min_distances.view(-1, self.num_prototypes)
        return min_distances

    def global_max_pooling(self, distances):
        """입력/출력과 용도: global_min_pooling과 동일하지만 부호를 뒤집지 않는다."""

        max_distances = F.max_pool2d(distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        max_distances = max_distances.view(-1, self.num_prototypes)

        return max_distances
    
    def subpatch_dist(self, x):
        """입력/출력과 기능

        - 입력 `x`: `[batch, 3, H, W]` 이미지 배치
        - 출력 tuple: `(conv_feature, dist_all)`
            * `conv_feature`: `[batch, feature_dim, 14, 14]` (ViT 특징맵)
            * `dist_all`: `[batch, num_proto, 196, num_slots]` float tensor
        - 기능: 각 슬롯(서브 프로토타입)에 대해 1x1 conv(정규화된 dot product)로 196개 패치 위치의 코사인 유사도 맵을 만들어 슬롯 차원으로 concat
        """

        dist_all = torch.FloatTensor().cuda()  # 초기 shape: [0]
        conv_feature = self.conv_features(x)  # [batch, dim, 14, 14]
        conv_features_normed = F.normalize(conv_feature, p=2, dim=1)  # [batch, dim, 14, 14]
        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)  # [num_proto, dim, num_slots]
        n_p = self.prototype_shape[-1]  # 슬롯 개수
        for i in range(n_p):
            proto_i = now_prototype_vectors[:, :, i].unsqueeze(-1).unsqueeze(-1)  # [num_proto, dim, 1, 1]
            dist_i = F.conv2d(input=conv_features_normed, weight=proto_i).flatten(2).unsqueeze(-1)
            # dist_i: [batch, num_proto, 196, 1] (196=14*14 패치 위치)
            dist_all = torch.cat([dist_all, dist_i], dim=-1) if dist_all.numel() else dist_i
            # dist_all 누적 shape: [batch, num_proto, 196, i+1]
        return conv_feature, dist_all
    
    def neigboring_mask(self, center_indices):
        """입력/출력과 기능

        - 입력 `center_indices`: `[batch, num_proto, 1]` (선택된 패치 위치, 0~195)
        - 출력: `[batch, num_proto, 196]` float tensor, 선택 패치와 반경(radius) 내 인접 패치만 1, 나머지 0
        - 기능: greedy 슬롯 선택 과정에서 새 후보 패치를 반경 내로 제한하는 마스크 생성
        """

        large_padded = (14 + self.radius * 2) ** 2  # 원래 14x14 격자 주변에 radius 만큼 padding을 둘러싼 뒤 평탄화한 크기
        large_matrix = torch.zeros(center_indices.shape[0], self.num_prototypes, large_padded).cuda()  # [B, P, (14+2r)^2]
        small_total = (2 * self.radius + 1) ** 2  # 커널로 추가할 패치 개수
        small_matrix = torch.ones(center_indices.shape[0], self.num_prototypes, small_total).cuda()  # [B, P, (2r+1)^2]
        batch_size, num_points, _ = center_indices.shape  # B, P, 1 -> B=배치, P=프로토타입 수
        small_size = int(small_matrix.shape[-1] ** 0.5)  # 커널 한 변 길이 (2r+1)
        large_size = int(large_matrix.shape[-1] ** 0.5)  # padding 포함 격자 한 변 길이 (14+2r)
        flat_idx = center_indices.squeeze(-1)          # [B, P]; 0~195, row-major로 flatten된 patch index
        center_row = flat_idx // 14  # [B, P]; 한 줄에 14개가 있으므로 몫은 행
        center_col = flat_idx % 14   # [B, P]; 나머지는 열
        start_row = torch.tensor(center_row + self.radius - small_size // 2, device=center_indices.device)
        start_col = torch.tensor(center_col + self.radius - small_size // 2, device=center_indices.device)
        # small_size // 2 == radius이므로 값 자체는 center_row/center_col과 같지만,
        # padding된 격자에서 커널의 top-left 좌표를 계산한 것이라는 의도를 살리기 위해 이 형태를 유지한다.
        start_row = torch.clamp(start_row, 0, large_size - small_size)  # padding 격자 안에서 커널 top-left가 유효하도록 제한
        start_col = torch.clamp(start_col, 0, large_size - small_size)
        # 결과적으로 중심이 격자 내부이면 start_row/col == center_row/col이 되고,
        # 가장자리에 가까우면 clamp에 의해 0 또는 14- (2r+1) 등으로 당겨져 창이 격자 밖으로 넘어가지 않는다.
        # (2r+1)x(2r+1) 슬라이딩 창을 padding 격자에 얹어 선택된 패치 주변을 1로 마킹한다.
        for i in range(small_size):
            for j in range(small_size):
                large_row = start_row + i       # [B, P]; padding 좌표계에서 커널의 현재 행
                large_col = start_col + j       # [B, P]; padding 좌표계에서 커널의 현재 열
                large_idx = large_row * large_size + large_col  # [B, P]; 2D -> flat 변환
                small_idx = i * small_size + j  # 스칼라; 커널 내부 인덱스
                # view + 고급 인덱싱을 쓰는 이유: 배치/프로토타입마다 서로 다른 large_idx 위치에 동시에 1을 더하기 위함.
                #   - torch.arange(batch_size)[:, None]  : [B,1]; 배치 축 인덱스
                #   - torch.arange(num_points)           : [P]; 프로토타입 축 인덱스
                #   - large_idx                          : [B,P]; padding 격자 내 target 위치 (각 배치/프로토타입마다 다름)
                # 이 조합으로 (B,P) 좌표마다 대응하는 1을 더하면 중심 패치 주변 (2r+1)^2 영역만 1이 된다.
                large_matrix.view(batch_size, num_points, -1)[
                    torch.arange(batch_size, device=large_matrix.device)[:, None],  # [B,1]
                    torch.arange(num_points, device=large_matrix.device),          # [P]
                    large_idx
                ] += small_matrix[..., small_idx]
        large_matrix_reshape = large_matrix.view(batch_size, num_points, large_size, large_size)  # padding 포함 2D 격자로 복원
        large_matrix_unpad = large_matrix_reshape[:, :, self.radius:-self.radius, self.radius:-self.radius]  # padding 제거 -> [B,P,14,14]
        large_matrix_unpad = large_matrix_unpad.reshape(batch_size, num_points, -1)  # 최종 [B, P, 196]
        # greedy_distance에서는 (1 - mask) * -1e5로 사용하므로 실제 값은 0/1여도 충분하다.
        return large_matrix_unpad
    

    def greedy_distance(self, x, get_f=False):
        """입력/출력과 기능

        - 입력 `x`: `[batch, 3, H, W]` 이미지 배치 (일반적으로 `[B, 3, 224, 224]`)
        - 출력 (기본):
            * `max_activation_slots`: `[batch, num_proto]`, 슬롯 확률이 반영된 활성 합
            * `min_distances`: `[batch, num_proto]`, 슬롯 수 - 활성 합으로 정의한 거리
            * `values_reordered`: `[batch, num_proto, num_slots]`, greedy 순서로 정렬된 슬롯 활성값
        - 출력 (`get_f=True`):
            * `conv_features`: `[batch, dim, 14, 14]`, ViT 백본 특징맵
            * `min_distances`: `[batch, num_proto]`
            * `indices_reordered`: `[batch, num_proto, num_slots]`, 선택된 패치 위치(0~195)
        - 기능: `subpatch_dist`에서 구한 `[B, num_proto, 196, num_slots]` 활성 맵을 사용해
                각 슬롯에 대해 가장 유사한 패치를 greedy하게 할당하고, radius 마스크로 인접성까지 반영한다.
        """
        conv_features, dist_all = self.subpatch_dist(x)  # conv_features:[B,dim,14,14], dist_all:[B,num_proto,196,num_slots]
        slots = torch.sigmoid(self.patch_select * self.temp)  # [1, num_proto, num_slots]; 슬롯 선택 확률(0~1)
        factor = (slots.sum(-1)).unsqueeze(-1) + 1e-10  # [1, num_proto, 1]; 슬롯 확률 합(0 방지 eps 포함)
        n_p = self.prototype_shape[-1]  # 슬롯 개수(num_slots)
        # 아래 마스크 텐서들은 greedy 선택에서 이미 사용된 패치/슬롯을 제외하기 위한 상태 저장 변수들이다.
        mask_act = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2])).cuda()      # [B,P,196]; 사용된 패치 위치를 0으로 만드는 마스크
        mask_subpatch = torch.ones((x.shape[0], self.num_prototypes, n_p)).cuda()              # [B,P,num_slots]; 사용된 슬롯(col)을 0으로 만드는 마스크
        mask_all = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2], n_p)).cuda()  # [B,P,196,num_slots]; 패치/슬롯 동시 제약 상태
        adjacent_mask = torch.ones((x.shape[0], self.num_prototypes, dist_all.shape[2])).cuda()  # [B,P,196]; radius 내 패치만 허용하는 마스크
        indices = torch.FloatTensor().cuda()  # [B,P,picked_slots]; 선택된 패치 인덱스(0~195) 누적 저장
        values = torch.FloatTensor().cuda()   # [B,P,picked_slots]; 선택된 슬롯의 코사인 활성값 누적 저장
        subpatch_ids = torch.LongTensor().cuda()  # [B,P,picked_slots]; greedy로 뽑힌 슬롯 번호 기록
        # --- Greedy 슬롯 매칭 루프 ---
        # 1) 사용 불가능한 위치(dist_all_masked)를 매우 작은 값(-1e5)으로 만들어 후보에서 제외
        # 2) 남은 위치 중 코사인 유사도가 최대인 패치를 찾고, 슬롯 및 패치 마스크를 갱신
        # 3) 선택 결과(슬롯 번호, 패치 index, 활성값)를 누적해 나중에 정렬/보정에 사용
        # 위 과정을 슬롯 개수(n_p)만큼 반복해 모든 서브 프로토타입에 패치를 할당한다.
        # --------------------------------
        # 슬롯 개수만큼 반복하며 각 슬롯에 가장 유사한 이미지 패치를 배정한다.
        for _ in range(n_p):
            # 이미 선택된 패치나 반경 밖 패치는 -1e5를 더해 softmax 없이도 선택되지 않게 만든다.
            dist_all_masked = dist_all + (1 - mask_all * adjacent_mask.unsqueeze(-1)) * (-1e5)
            # dist_all_masked: [B,P,196,num_slots]; 사용 불가(이미 선택 or radius 밖) 위치는 -1e5로 덮어써서 후보에서 제외
            # STEP1) dim=2(패치 축)에서 torch.max를 수행해 "슬롯마다" 가장 잘 맞는 패치 값을 얻는다.
            #       여기서 슬롯 축(num_slots)은 그대로 남아 있으므로, 각 슬롯이 자기 최고의 패치를 고른 상태가 된다.
            max_subs, max_subs_id = dist_all_masked.max(2)            # max_subs:[B,P,num_slots], max_subs_id:[B,P,num_slots]
            # STEP2) dim=-1(슬롯 축)에서 다시 torch.max를 수행해, 같은 프로토타입 내 여러 슬롯 중 최종으로 쓸 슬롯 하나를 고른다.
            #       프로토타입 축(P)은 남아 있으므로 다른 프로토타입과 섞이지 않는다.
            max_sub_act, max_sub_act_id = max_subs.max(-1)             # max_sub_act:[B,P], max_sub_act_id:[B,P]
            # STEP3) gather로 STEP1의 index 테이블에서 STEP2에서 고른 슬롯 위치만 골라, 실제 patch index(0~195)를 복원한다.
            max_patch_id = max_subs_id.gather(-1, max_sub_act_id.unsqueeze(-1))  # [B,P,1]
            # 선택된 패치를 중심으로 radius 이내 패치만 다음 후보로 남기기 위해 인접 마스크를 갱신한다.
            adjacent_mask = self.neigboring_mask(max_patch_id)
            # 중복 선택 방지 ①: 이번에 고른 patch index를 mask_act에 반영해 해당 위치를 0으로 만든다.
            mask_act = mask_act.scatter(index=max_patch_id, dim=2, value=0)
            # 중복 선택 방지 ②: 이번에 고른 슬롯 번호를 mask_subpatch에 반영해 해당 슬롯을 0으로 만든다.
            mask_subpatch = mask_subpatch.scatter(index=max_sub_act_id.unsqueeze(-1), dim=2, value=0)
            # 4D 마스크 갱신: patch 마스크와 슬롯 마스크를 결합해 사용 불가능한 위치를 모두 0으로 만든다.
            mask_all = mask_all * mask_act.unsqueeze(-1)  # patch 기준으로 이미 선택된 위치 제거
            mask_all = mask_all.permute(0, 1, 3, 2)
            mask_all = mask_all * mask_subpatch.unsqueeze(-1)  # 슬롯 기준으로 이미 사용한 슬롯 제거
            mask_all = mask_all.permute(0, 1, 3, 2)  # shape 복원 -> [B,P,196,num_slots]
            # 선택된 슬롯/패치 쌍의 (슬롯 번호, 패치 index, 활성값)을 누적 저장한다.
            max_sub_act = max_sub_act.unsqueeze(-1) # [B,P,1]
            subpatch_ids = torch.cat([subpatch_ids, max_sub_act_id.unsqueeze(-1)], dim=-1)  # 슬롯 선택 순서를 기록
            indices = torch.cat([indices, max_patch_id], dim=-1)  # 선택된 이미지 패치 위치
            values = torch.cat([values, max_sub_act], dim=-1)  # 슬롯별 활성값(코사인 유사도) 누적
        subpatch_ids = subpatch_ids.to(torch.int64)
        # greedy로 뽑힌 슬롯 번호(subpatch_ids)는 예컨대 [2,0,3,1]처럼 "선택 순서"로 기록되어 있다.
        # torch.sort는 (정렬된 값, 정렬에 필요한 인덱스)를 반환하므로 sub_indexes가 [1,3,0,2]처럼 나온다.
        # -> 이 인덱스를 torch.gather에 쓰면 values/indices를 "슬롯 번호 0,1,2,3" 순서로 재배열할 수 있다.
        _, sub_indexes = subpatch_ids.sort(-1)
        # torch.gather(..., index=sub_indexes) -> 슬롯 번호가 낮은 순서대로 값을 재배열한다.
        # 반환 shape은 index와 동일하므로 [B,P,num_slots]로 정렬된 활성값/패치 index가 나온다.
        values_reordered = torch.gather(values, -1, sub_indexes)
        indices_reordered = torch.gather(indices, -1, sub_indexes)
        # 슬롯 확률(slots)과 greedy 활성값을 곱해 최종 슬롯 가중치를 만든다.
        # slot 확률을 합이 n_p가 되도록 보정 후, 슬롯별 활성에 가중치로 곱해 최종 활성도를 만든다.
        values_slot = values_reordered.clone() * (slots * n_p / factor)  # [B,P,num_slots]
        max_activation_slots = values_slot.sum(-1)  # [B,P]; 가중된 슬롯 활성 합
        min_distances = n_p - max_activation_slots  # [B,P]; ProtoPNet distance (작을수록 가까움)
        if get_f:
            return conv_features, min_distances, indices_reordered  # conv_features:[batch,dim,14,14], indices:[batch,num_proto,num_slots]
        return max_activation_slots, min_distances, values_reordered  # max_activation/min_distances:[batch,num_proto], values:[batch,num_proto,num_slots]

    def push_forward_old(self, x):
        """입력/출력과 기능

        - 입력 `x`: `[batch, 3, H, W]`
        - 출력: `(conv_output, distances)`
            * `conv_output`: `[batch, dim, 14, 14]` 특징맵 (슬롯 개념 도입 전 사용)
            * `distances`: `[batch, num_proto, 14, 14]` cosine projection을 부호 반전한 거리 맵
        - 기능: slots 도입 이전 ProtoPNet push 단계와 동일하게, 특징맵과 거리 맵을 반환한다.
        """

        conv_output = self.conv_features(x)  # [batch, dim, 14, 14]
        distances = self._project2basis(conv_output)  # cosine 유사도
        distances = -distances  # ProtoPNet convention: 유사도를 부호 반전해 거리로 사용
        return conv_output, distances
    
    def push_forward(self, x):
        """입력/출력과 기능

        - 입력 `x`: `[batch, 3, H, W]`
        - 출력: `(conv_output, min_distances, indices)`
            * `conv_output`: `[batch, dim, 14, 14]`
            * `min_distances`: `[batch, num_proto]`
            * `indices`: `[batch, num_proto, num_slots]` (슬롯별 선택된 패치 index)
        - 기능: 최신 greedy 슬롯 로직을 사용해 push 단계에서 필요한 특징맵, 거리, 패치 위치를 반환한다.
        """

        conv_output, min_distances, indices = self.greedy_distance(x, get_f=True)
        return conv_output, min_distances, indices 
    
    def forward(self, x):
        """입력/출력과 기능

        - 입력 `x`: `[batch, 3, H, W]`
        - 출력: `(logits, min_distances, slot_values)`
            * `logits`: `[batch, num_classes]`, evidence layer 결과
            * `min_distances`: `[batch, num_proto]`, ProtoPNet distance
            * `slot_values`: `[batch, num_proto, num_slots]`, 정렬 이전 슬롯 활성값
        - 기능: greedy 슬롯 활성도를 계산해 evidence layer(`last_layer`)로 보내고, 거리/슬롯 정보를 함께 반환한다.
        """

        max_activation, min_distances, values = self.greedy_distance(x)
        logits = self.last_layer(max_activation)  # evidence layer: ProtoPNet-style linear classifier
        return logits, min_distances, values
    
    def __repr__(self):
        """입력/출력

        - 입력: 없음 (객체 자체 호출)
        - 출력: `str` 형식의 요약 정보
            * features: 백본 모듈 요약 문자열
            * img_size/prototype_shape/num_classes/epsilon 값
        역할: `print(ppnet)` 시 모델 주요 설정을 빠르게 확인할 수 있도록 한다.
        """

        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """입력/출력과 알고리즘

        - 입력: `incorrect_strength` (float, 예: -0.5)
        - 출력: 없음 (in-place로 `self.last_layer.weight`를 갱신)
        - 알고리즘:
            1. `prototype_class_identity` (shape `[num_proto, num_classes]`)를 transpose → `[num_classes, num_proto]`
               각 행이 한 클래스에 할당된 프로토타입 위치(one-hot).
            2. 정답 클래스 위치는 1, 나머지는 0이므로, (1 - mask)를 하면 "오답 클래스 위치"가 1이 된다.
            3. 정답 클래스에는 +1을, 오답 클래스에는 `incorrect_strength`를 곱해 증거층 가중치를 구성.
            4. `self.last_layer.weight`는 `[num_classes, num_proto]` shape이며, ProtoPNet 초깃값을 그대로 따른다.
        """

        positive_one_weights_locations = torch.t(self.prototype_class_identity)  # [num_classes, num_proto]; 정답 클래스에 속한 프로토타입 위치만 1
        negative_one_weights_locations = 1 - positive_one_weights_locations      # [num_classes, num_proto]; 오답 위치만 1

        correct_class_connection = 1.0
        incorrect_class_connection = incorrect_strength
        weight_template = (
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )
        # weight_template: [num_classes, num_proto]; 정답=+1, 오답=incorrect_strength
        # ProtoPNet 초기화 규칙을 사용하면 학습이 진행되면서 오답 연결이 0 근처로 수렴해 양의 증거만 사용하게 된다.
        self.last_layer.weight.data.copy_(weight_template)

    def _initialize_weights(self):
        """ProtoPNet 초기화 규칙을 따라 증거층 가중치를 설정."""

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)



def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 192, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    sig_temp = 1.0,
                    radius = 1,
                    add_on_layers_type='bottleneck'):
    """PPNet 생성 헬퍼

    - `base_architecture`: timm/cait 백본 이름 (`deit_small_patch16_224` 등)
    - `pretrained`: 백본 가중치 로딩 여부
    - `img_size`: 입력 한 변 크기 (224 권장)
    - `prototype_shape`: `(num_proto, feature_dim, num_slots)` 설정
    - `num_classes`: 클래스 수
    - `prototype_activation_function`: 거리->유사도 변환 함수 종류
    - `sig_temp`: 슬롯 시그모이드 temperature
    - `radius`: greedy 슬롯 반경
    - `add_on_layers_type`: 추가 레이어 타입(현재 미사용)
    """
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)

    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 radius = radius,
                 sig_temp = sig_temp,
                 add_on_layers_type=add_on_layers_type)
