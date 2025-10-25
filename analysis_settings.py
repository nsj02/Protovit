"""분석 스크립트(local/global)를 위한 공통 설정.

Colab과 로컬 환경 모두에서 동일한 스크립트를 사용할 수 있도록 경로를
환경을 감지해 유연하게 지정한다.
"""

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# 기본 경로 정의
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / 'saved_models' / 'deit_small_patch16_224' / 'exp1'
# push 후 시각화 에셋(img/epoch-14)에 맞춘 기본 체크포인트
DEFAULT_MODEL_NAME = '14push0.8567.pth'
DEFAULT_IMG_SUBDIR = 'img/epoch-14'
DEFAULT_DRIVE_DATA_ROOT = PROJECT_ROOT / 'cub200_cropped'
COLAB_DATA_ROOT = Path('/content/cub200_cropped')
DRIVE_ANALYSIS_ROOT = PROJECT_ROOT / 'analysis_outputs'

# -----------------------------------------------------------------------------
# 모델 경로 설정 (환경 변수 우선)
# -----------------------------------------------------------------------------
model_path_env = os.getenv('PROTO_VIT_MODEL_PATH')
if model_path_env:
    model_path = Path(model_path_env).expanduser().resolve()
    if model_path.is_dir():
        load_model_dir = model_path.as_posix()
        load_model_name = os.getenv('PROTO_VIT_MODEL_NAME', DEFAULT_MODEL_NAME)
    else:
        load_model_dir = model_path.parent.as_posix()
        load_model_name = model_path.name
else:
    load_model_dir = DEFAULT_MODEL_DIR.as_posix()
    load_model_name = DEFAULT_MODEL_NAME

# `img` 시각화 폴더는 load_model_dir 기준 상대 경로로 지정
img_name = os.getenv('PROTO_VIT_IMG_SUBDIR', DEFAULT_IMG_SUBDIR)

# -----------------------------------------------------------------------------
# 분석 결과 저장 경로 (Colab: /content, 로컬: 프로젝트 내부)
# -----------------------------------------------------------------------------
analysis_out_env = os.getenv('PROTO_VIT_ANALYSIS_OUT')
if analysis_out_env:
    save_path = Path(analysis_out_env).expanduser()
else:
    save_path = DRIVE_ANALYSIS_ROOT / 'finished'

save_analysis_path = save_path.as_posix()

# -----------------------------------------------------------------------------
# 테스트 데이터 경로 (Colab 로컬 > 드라이브 순)
# -----------------------------------------------------------------------------
data_root_env = os.getenv('PROTO_VIT_DATA_ROOT')
if data_root_env:
    data_root = Path(data_root_env).expanduser()
else:
    data_root = COLAB_DATA_ROOT if COLAB_DATA_ROOT.exists() else DEFAULT_DRIVE_DATA_ROOT

test_data = (data_root / 'test_cropped').as_posix()

# -----------------------------------------------------------------------------
# 기타 옵션
# -----------------------------------------------------------------------------
check_test_acc = False
check_list = [
    '001_001_Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg',
    '001_001_Black_footed_Albatross/Black_Footed_Albatross_0002_55.jpg',
    '001_001_Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg',
    '002_002_Laysan_Albatross/Laysan_Albatross_0001_545.jpg',
    '002_002_Laysan_Albatross/Laysan_Albatross_0002_1027.jpg',
    '002_002_Laysan_Albatross/Laysan_Albatross_0004_930.jpg',
    '003_003_Sooty_Albatross/Sooty_Albatross_0002_796395.jpg',
    '003_003_Sooty_Albatross/Sooty_Albatross_0003_1078.jpg',
    '003_003_Sooty_Albatross/Sooty_Albatross_0004_796366.jpg',
]

