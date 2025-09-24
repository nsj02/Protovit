"""분석 스크립트(local/global)를 위한 공통 설정."""

# --- 분석 대상 모델 체크포인트 설정 ---
load_model_dir = 'saved model path'  # 예: './saved_models/deit_small/003/'
load_model_name = 'model_name'       # 예: 'finetuned0.9230.pth'

# --- 분석 결과 저장 경로/파일명 접두사 ---
save_analysis_path = 'saved_dir_rt'  # 해석 결과 아웃풋을 저장할 루트 디렉터리
img_name = 'prototype_vis_file'      # 시각화 이미지 저장 시 사용할 접두사 (폴더명/파일명)

# --- 평가에 사용할 데이터셋/옵션 ---
test_data = 'test_dir'  # 테스트 데이터 경로(설정에 맞춰 절대경로로 갱신 필요)
check_test_acc = False  # 분석 진행 중 테스트 정확도를 재계산할지 여부

# --- 특정 샘플만 분석하고 싶을 때 사용할 리스트 ---
check_list = [
    'list of test images',  # 예: "163_Mercedes-Benz SL-Class Coupe 2009/03123.jpg"
]
