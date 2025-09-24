"""ProtoViT 학습/해석 전반에서 공통으로 참조하는 기본 하이퍼파라미터 모음."""

# --- 백본 및 프로토타입 구성이 기대하는 입력 해상도/패치 정보를 정의 ---
base_architecture = 'deit_small_patch16_224'  # 사용할 ViT 백본 ID (timm 네이밍)
radius = 1  # 프로토타입이 커버하는 패치 반경 (patch 단위)
img_size = 224  # 백본이 받는 입력 이미지 한 변 크기

# 백본마다 최종 프로토타입 벡터의 채널 수와 slot 길이가 다르므로 조건 분기
if base_architecture == 'deit_small_patch16_224':
    prototype_shape = (2000, 384, 4)  # (총 프로토 수, 임베딩 차원, slot 개수)
elif base_architecture == 'deit_tiny_patch16_224':
    prototype_shape = (2000, 192, 4)
elif base_architecture == 'cait_xxs24_224':
    prototype_shape = (2000, 192, 4)

# 학습 기본 하이퍼파라미터 ---------------------------------------------------
dropout_rate = 0.0  # backbone/add-on에서 사용하는 드롭아웃 확률
num_classes = 200  # CUB-200 기준 클래스 수
prototype_activation_function = 'log'  # 프로토타입 활성 함수 타입
add_on_layers_type = 'regular'  # 추가 레이어 구성 방식(regular/last_layer 등)
experiment_run = 'exp1'  # 저장 경로 구분용 실험 식별자

# --- 데이터 경로: preprocess 단계에서 구축한 디렉터리 구조를 그대로 참조 ---
data_path = "./cub200_cropped/"  # 루트 경로 (필요 시 절대경로로 수정)
train_dir = data_path + 'train_cropped_augmented/'  # 증강된 학습 데이터
test_dir = data_path + 'test_cropped/'  # 검증/테스트 데이터
train_push_dir = data_path + 'train_cropped/'  # 프로토타입 push 단계에서 사용할 원본 학습 이미지

# --- DataLoader 배치 크기 ---
train_batch_size = 128
test_batch_size = 100
train_push_batch_size = 75  # push 단계는 자주 I/O가 발생하므로 약간 작게 유지

# --- 단계별 학습률 세팅 ---
# joint 단계: backbone과 프로토타입을 동시에 업데이트
joint_optimizer_lrs = {
    'features': 5e-5,  # backbone 파라미터 학습률 (주석의 값들은 실험 로그)
    'prototype_vectors': 3e-3,  # 프로토타입 벡터 학습률
}
joint_lr_step_size = 5  # schedular가 lr을 감소시키는 epoch 주기

# stage_2: patch select 모듈만 미세 조정
stage_2_lrs = {'patch_select': 5e-5}

# warm 단계: backbone 업데이트를 거의 멈추고 프로토타입만 조정
warm_optimizer_lrs = {
    'features': 1e-7,
    'prototype_vectors': 3e-3,
}

last_layer_optimizer_lr = 1e-4  # 최종 분류기(fully connected) 학습률

# --- 손실 항목 가중치 ---
coefs = {
    'crs_ent': 1,     # cross-entropy
    'clst': -0.8,     # prototype cluster loss (foreground 집중 유도)
    'sep': 0.1,       # separation loss (타 클래스에서 멀어지도록)
    'l1': 1e-2,       # 라스트 레이어 L1 규제
    'orth': 1e-3,     # 프로토타입 직교성 패널티
    'coh': 3e-3,      # slot coherence 관련 항
}

coefs_slots = {'coh': 1e-6}  # slot별 희소성/일관성 가중치

# --- 프로토타입 활성화 계산 관련 파라미터 ---
sum_cls = False  # 클래스별 activation을 합산할지 여부 (False면 max 기반)
k = 1  # cluster cost 계산 시 top-k를 고려하는지 (1이면 최대치만 사용)
sig_temp = 100  # 시그모이드 온도 (slot selection sharpness)

# --- 학습 기간 설정 ---
num_joint_epochs = 10
num_warm_epochs = 5
num_train_epochs = num_joint_epochs + num_warm_epochs  # 전체 에폭 수

slots_train_epoch = 5  # slot 학습 전용 반복 횟수
push_start = 10  # push 단계 시작 epoch (warm-up 이후)
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]  # 10 epoch마다 push 실행
