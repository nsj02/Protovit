# 진행 기록 (ProtoViT)

## 전체 플로우 요약
1. **데이터 정리** — `preprocess.py`, `img_aug.py`, `tools/preprocess.py`로 CUB/Stanford Cars 데이터를 정규화하고 증강하며, 결과를 `datasets/<task>` 경로에 배치합니다.
2. **환경/설정 준비** — `settings.py`, `analysis_settings.py`, `spatial_alignment_test/adv_setting.py`에서 경로·하이퍼파라미터·실험 옵션을 확정합니다.
3. **학습 실행** — `main.py` 또는 `train_and_test.py`로 학습/검증을 돌린 뒤, 필요 시 `push.py`, `push_greedy.py`로 프로토타입을 업데이트합니다.
4. **해석 및 분석** — `local_analysis.py`, `global_analysis.py`, `find_nearest.py`를 이용해 프로토타입 해석 결과와 샘플별 근접 이미지를 생성합니다.
5. **추가 실험** — `spatial_alignment_test/run_adv_test.py`로 위치 불일치 실험을 수행하고, 산출물은 `save.py` 혹은 지정한 출력 경로에 정리합니다.
6. **기록/정리** — `prompt/progress_log.md` 체크리스트와 노트를 갱신하고, 중요 변경 사항은 커밋 메시지 규칙에 맞춰 기록합니다.

## 사용 안내
- 이 문서는 Python 소스 파일을 읽으며 이해한 내용을 기록하기 위한 용도입니다.
- 각 항목의 체크박스를 읽기 완료 시 `[x]`로 갱신하고, `메모`에 핵심 요약이나 TODO를 짧게 남겨 주세요.
- 새로운 파일을 검토하면 동일 형식으로 항목을 추가하고, 필요하면 하위 Bullet로 세부 함수를 정리합니다.
- 노트에는 `YYYY-MM-DD` 형식으로 날짜를 적고, 후속 확인이 필요한 항목을 명시합니다.

## 진행 체크리스트

- [x] `preprocess.py` — 데이터셋 정규화(Imagenet mean/std) 및 역정규화 유틸.
  - 함수 요약: `preprocess`(RGB 채널별 정규화), `preprocess_input_function`(사본 반환), `undo_preprocess`(역정규화), `undo_preprocess_input_function`(복원 사본 제공).
  - 메모: 텐서 복제 전략과 채널 검증 로직 주석으로 명시.
- [x] `img_aug.py` — Augmentor 파이프라인으로 회전/스큐/시어 및 좌우반전 증강.
  - 함수 요약: `makedir`(출력 경로 생성), 메인 루프에서 `Pipeline` 구성(회전/스큐/시어 각각 10회 반복).
  - 메모: 클래스별 디렉터리 매핑, 확률/각도 파라미터, 루프 반복 횟수 설명.
- [x] `tools/preprocess.py` — tools 모듈용 전처리 함수 복제본.
  - 함수 요약: `preprocess`/`undo_preprocess` 페어와 wrapper 두 개가 동일 역할 수행.
  - 메모: tools 하위 스크립트에서 독립 사용 가능하도록 설계한 목적 기록.
- [x] `preprocess_sample_code/data_crop_sample.ipynb` — Stanford Cars 바운딩 박스 기반 크롭/리사이즈 샘플.
  - 단계: `scipy.io.loadmat`으로 annos 로드 -> 클래스별 폴더 생성 -> `PIL.Image.crop` 후 448×448로 리사이즈해 저장.
  - 메모: `stanford_car_raw` 구조 전제, bbox 크롭 루프까지 한국어 주석으로 세부 설명.
- [x] `preprocess_sample_code/data_split_sample.ipynb` — CUB-200 이미지/라벨 스플릿 예제.
  - 단계: `images.txt`, `image_class_labels.txt`, `train_test_split.txt` 읽기 -> 클래스별 디렉터리 생성 -> uncropped train/test 폴더로 복사.
  - 메모: 경로 하드코딩 주의, 코드 셀별 데이터 흐름 주석 추가 완료.

### 환경 & 설정
- [x] `settings.py` — 데이터 경로, 학습/추론 기본 하이퍼파라미터 정의.
  - 주요 변수: `base_architecture`/`prototype_shape`, 데이터 경로(`train_dir`,`test_dir`,`train_push_dir`), 배치 크기, 단계별 lr 딕셔너리, 손실 가중치(`coefs`), push 스케줄.
  - 메모: 학습 단계별 의미, slot 관련 파라미터, push 주기 등을 한국어 주석으로 정리.
- [x] `analysis_settings.py` — 분석 스크립트용 경로와 체크포인트 설정.
  - 주요 키: `load_model_dir`/`load_model_name`, `save_analysis_path`, `img_name`, `test_data`, `check_list`.
  - 메모: local/global 분석 스크립트에서 동일한 config를 읽도록 경로/옵션을 주석화. 실제 사용 시 체크포인트·데이터 경로를 환경에 맞게 갱신 필요.
- [x] `spatial_alignment_test/adv_setting.py` — 위치 불일치 실험 설정값.
  - 주요 키: `load_model_path`(공격 대상 모델 폴더), `test_dir`(평가 데이터), `model_output_dir`(결과 저장 위치).
  - 메모: 실험별로 폴더 분리 권장, 주석으로 교체해야 할 경로 명시.
- [x] `tools/deit_features.py` — DeiT 백본 로더.
  - 함수 요약: 권장 weight URL, positional embedding interpolation, timm 래퍼 설명.
- [x] `tools/cait_features.py` — CaiT 백본 로더.
  - 함수 요약: timm `create_model` 래퍼, head 제거 처리.

### 학습 & 프로토타입 관리
- [x] `main.py` — 전체 학습 파이프라인 시작, 설정 로딩 및 훈련 루프 호출.
  - 흐름: argparse로 GPU 고정 -> 데이터로더 구성 -> PPNet 생성 -> warm/joint/slot/finetune 단계 -> push_greedy -> last-layer finetune.
  - 주요 블록: optimizer 묶음(`joint_optimizer`, `joint_optimizer2`, `warm_optimizer`, `last_layer_optimizer`), 학습 루프 4단계, `save.save_model_w_condition`.
  - 메모: 각 단계 동작과 저장 조건을 한국어 주석으로 명시, push 시 원본 픽셀 사용 이유 추가.
- [x] `model.py` — ProtoViT 구조 및 greedy slot 선택 로직.
  - 함수 요약:
    - `conv_features(x)`: `[B,3,H,W]` -> `[B,dim,14,14]`; PatchEmbed, cls 토큰, DeiT/CaiT 블록, 전역 차이 추출 흐름 기록.
    - `_cosine_convolution(x)`: 정규화된 dot-product로 코사인 거리 맵 계산.
    - `_project2basis(x)`: push용 cosine 유사도 반환.
    - `prototype_distances(x)`: projection/거리 쌍 반환.
    - `global_min_pooling(distances)`: 공간 최소 거리 집계.
    - `subpatch_dist(x)`: 슬롯별 cosine 활성 맵 concat.
    - `neigboring_mask(center_indices)`: radius padding -> gather -> clamp -> (2r+1)^2 마스크 생성 과정을 상세 주석화.
    - `greedy_distance(x, get_f=False)`: 슬롯별 greedy 매칭, 마스크 갱신, 거리/활성 반환.
    - `push_forward(_old)`, `forward(x)`: push 및 추론 경로 설명.
  - 메모: 슬롯 마스크 및 gather 순서, radius 기반 padding 처리, clamp 의미 등을 한국어로 상세 정리. slot coherence, orth 손실 계산은 `train_and_test.py`에서 주석화됨.
- [x] `train_and_test.py` — 학습/검증 루프와 프로토타입 손실 계산.
  - 주요 함수: `_train_or_test`(공통 루프), `train`/`test` wrapper, `warm_only`/`joint`/`last_only` 단계별 grad 설정.
  - 메모: 손실 항목, 슬롯/직교/분리 계산, min_distances·slots 마스킹, EMA/메모리/로그 누적 로직까지 주석화 완료.
- [x] `push.py` — 프로토타입 업데이트(푸시) 루틴.
  - 함수 요약: `update_prototypes_on_batch`(배치별 최소 거리/RF 갱신), `push_prototypes`(global 최소 유지·저장).
  - 메모: 입력/출력 shape, numpy 갱신 흐름, 저장 옵션 경로 등을 주석으로 명확히 기술.
- [x] `push_greedy.py` — Greedy 방식 프로토타입 선택.
  - 함수 요약: `save_prototype_original_img_with_bbox`, `update_prototypes_on_batch`, `push_prototypes` 모두 슬롯별 입력/출력/shape 정리.
  - 메모: greedy push는 슬롯 index(`proto_indices`) 활용, slot rounding 및 bbox 저장 흐름 주석화.
- [x] `find_nearest.py` — 프로토타입 근접 이미지 조회.
  - 함수 요약: `save_prototype_original_img_with_bbox`, `find_k_nearest_patches_to_prototypes` 입력/출력/shape 주석화.
  - 메모: 슬롯 기반 코사인 활성→heap 정렬→이미지 저장 흐름 명확히 기록.
### 해석 & 분석
- [x] `global_analysis.py` — 프로토타입 전역 해석 시각화.
  - 함수 요약: 스크립트 상단에 입력/출력 목적, `save_prototype` 설명, train/test 최근접 탐색 흐름 주석화.
  - 메모: dataloader 구성·출력 경로·find_nearest 호출부 주석으로 정리 완료.
- [x] `local_analysis.py` — 개별 샘플 로컬 해석 및 시각화.
  - 함수 요약: `local_analysis`, `save_preprocessed_img`, `save_prototype_original_img_with_bbox` 입력/출력 및 슬롯 기반 bbox 표시 주석화.
  - 메모: 이미지 전처리, top-k class/슬롯 활성 시각화 흐름 설명 완료.

### 로깅 & 저장 유틸
- [x] `helpers.py` — 거리 계산 등 훈련 보조 함수 모음.
  - 함수 요약: `list_of_distances`, `make_one_hot`, `find_high_activation_crop` 등 입력/출력/shape 설명 추가.
  - 메모: 공통 유틸이 사용하는 numpy/torch 변환 및 용도 정리.
- [x] `log.py` — 학습 진행 상황 로깅 유틸리티.
  - 함수 요약: `create_logger` 입력/출력/동작 방식 주석화.
  - 메모: flush/fsync 주기 설명, 사용법 명시.
- [x] `save.py` — 체크포인트 저장/불러오기 로직.
  - 함수 요약: `save_model_w_condition` 입력/출력/저장 조건 설명.
  - 메모: 모델 전체를 저장하는 구조임을 주석으로 명확히 함.

### 추가 실험
- [x] `spatial_alignment_test/run_adv_test.py` — 위치 불일치 공격 실행.
  - 함수 요약: adversarial heatmap/마스크/지표 계산 함수에 입력/출력 주석 추가.
  - 메모: CleverHans PGD 활용 흐름과 데이터셋 순회 generator 동작 설명.

## 진행 노트
| 날짜 | 파일 | 핵심 요약 | 후속 작업 |
| --- | --- | --- | --- |
| 2025-09-19 | `preprocess.py`, `img_aug.py`, `tools/preprocess.py`, `preprocess_sample_code/*.ipynb`, `settings.py`, `analysis_settings.py`, `adv_setting.py`, `main.py`, `train_and_test.py` | 데이터/설정/학습 루프 스크립트 주석화 및 단계별 요약 정리 | Augmentor 설치 및 경로 권한 확인 |
| 2025-09-21 | 차이 분석 | 코드 단계 vs. 논문(ProtoViT 3.4) 비교: 현재 구현은 warm/joint/slot 단계 모두에서 `_train_or_test`가 동일 손실(Lce, Lclst, Lsep, Lcoh, Lorth, L1)을 합산하며, slot pruning 전용 손실(Lprune)이나 projection/last-layer stage 구분이 없다. slot sparsity/평균 추적 주석과 실제 계산도 일치하지 않아 참고용으로만 사용. | 코드 단계별 손실 분리 여부 검토, slot 통계 계산 방식 개선 |
| 2025-09-21 | `model.py` | `conv_features`와 거리/greedy 관련 보조 메서드까지 shape·마스크 로직을 한국어 주석으로 보강. | slot coherence 수식과 orth 손실 계산은 `train_and_test.py` 측 설명 재검토 |
| 2025-09-22 | `model.py` | `greedy_distance` 단계 설명, `forward`/`push_*`/`construct_PPNet` 입출력 주석 보완. `torch.sort`/`gather` 인덱스 재배치 의도 명시. | 추후 slots 재가중 로직(`values_slot`) 수식 검증 |
| 2025-09-22 | `push.py` | `update_prototypes_on_batch`/`push_prototypes` 입력/출력 및 numpy 기반 갱신 흐름 주석화. 저장 옵션·RF 계산 과정 상세 기록. | 이후 `push_greedy.py` 비교 검토 |
| 2025-09-22 | `push_greedy.py` | 슬롯 기반 push 함수 전반(`save_prototype_original_img_with_bbox`, `update_prototypes_on_batch`, `push_prototypes`)에 입력/출력/shape 및 greedy 슬롯 흐름 주석화. | greedy push와 기본 push 차이 분석 메모 참조 |
| 2025-09-22 | `find_nearest.py` | 최근접 탐색 함수 입력/출력·슬롯 기반 heap 정렬 과정 주석화, bbox 시각화 보조 함수 정리. | 저장 이미지 시각 품질 확인 |
| 2025-09-22 | `global_analysis.py` | 스크립트 전체 흐름(모델 로드, dataloader 준비, nearest 검색) 주석화 및 `save_prototype` 입력/출력 설명. | `local_analysis.py`로 이어서 확인 |
| 2025-09-22 | `local_analysis.py` | 로컬 해석 함수 입력/출력·슬롯 기반 시각화 과정 주석화. top-k 클래스/프로토타입 활성 흐름 명시. | `analysis_settings.py` 확인 완료 |
| 2025-09-22 | `helpers.py` | 공통 유틸 함수 docstring/입출력/shape 설명 추가. | 이후 `log.py` 확인 |
| 2025-09-22 | `log.py` | 단일 로거 생성 함수 입력/출력/flush 정책 주석화. | `save.py` 검토 예정 |
| 2025-09-22 | `save.py` | 조건부 체크포인트 저장 함수 입력/출력 정리. | 필요 시 state_dict 저장 버전 검토 |
| 2025-09-22 | `spatial_alignment_test/run_adv_test.py` | adversarial 분석 스크립트 주요 함수(docstring/입출력/shape) 정리. | 지표 해석 시각화 품질 확인 |
| 2025-09-22 | `tools/deit_features.py`, `tools/cait_features.py` | timm 백본 래퍼 docstring 및 입력/출력 설명 추가. positional embedding 보간 주석화. | 다른 백본 추가 시 참고 |
| 2025-09-23 | `push.py` | 시각화 저장 블록(원본/heatmap/RF/proto patch)과 `push_prototypes` 전 과정(배치 순회, 전역 최소 버퍼, dir 준비, bbox 저장)까지 도형·dtype·의미를 세분화해 주석 강화. | 동일 주석 스타일을 `push_greedy.py` 저장/루프 파트에도 반영 검토 |
| 2025-09-23 | `push_greedy.py` | `save_prototype_original_img_with_bbox` 인자 설명, 슬롯 시각화 루프, greedy push 전체 흐름(`push_prototypes` docstring/slot binarization/stat 로그)까지 한국어 주석 강화. | greedy push의 `global_min_fmap_patches` 업데이트 흐름 추가 점검 |
| 2025-09-23 | `find_nearest.py` | `save_prototype_original_img_with_bbox`, `find_k_nearest_patches_to_prototypes`에 입력/출력/shape 흐름 주석을 촘촘히 추가하고 힙 유지 로직·슬롯 bbox 계산 과정을 한국어로 정리. | 힙 결과 시각화(heatmap) 품질 확인 |
| 2025-09-23 | `global_analysis.py` | argparse/GPU 선택, 모델 로드, dataloader 구성, prototype bbox 복사 루프와 최근접 탐색 호출 부분에 줄단위 한국어 주석 추가. | recent 결과 폴더 구조 확인 |
| 2025-09-23 | `local_analysis.py` | helper 함수, `local_analysis`, `analyze` 흐름에 상세 한국어 주석(입력/출력/shape, 루프 단계) 추가. | slot bbox 좌표 시각화 품질 확인 |
