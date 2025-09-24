"""위치 불일치(Location Misalignment) 실험을 위한 기본 경로 설정."""

# 공격/방어 실험에 사용할 학습된 모델(체크포인트)이 존재하는 디렉터리
load_model_path = "."  # 예: './saved_models/deit_small/robust_run/'

# 평가에 사용할 테스트 데이터 경로 (전처리된 cub200_cropped/test_cropped 등으로 교체)
test_dir = "./cub200_cropped/test_cropped"

# 공격 결과, 로그, 시각화 등을 저장할 출력 디렉터리
model_output_dir = "."  # 실험별로 서브폴더를 만들어 두는 것을 권장
