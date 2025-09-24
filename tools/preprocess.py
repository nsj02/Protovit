import torch

# tools 모듈에서 독립적으로 사용할 수 있도록 `preprocess.py` 내용을 복제한 버전.
# Imagenet 평균/표준편차 기반 정규화/역정규화 로직을 동일하게 제공한다.
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def preprocess(x, mean, std):
    # RGB 3채널 텐서만 정상 처리되므로 방어적으로 채널 수를 검사.
    assert x.size(1) == 3
    # 입력과 동일 크기의 버퍼에 결과를 쓰면 원본 텐서를 보존할 수 있다.
    y = torch.zeros_like(x)
    for i in range(3):
        # 채널별로 평균을 빼고 표준편차로 나눠 정규화 수행.
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y

def preprocess_input_function(x):
    '''
    pretrained 백본이 기대하는 정규화된 입력 텐서를 새로 생성해 반환.
    '''
    return preprocess(x, mean=mean, std=std)

def undo_preprocess(x, mean, std):
    # 정규화된 이미지를 원래 픽셀 분포로 되돌릴 때 사용.
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        # 표준편차를 곱하고 평균을 더하면 정규화 전 스케일로 복원된다.
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def undo_preprocess_input_function(x):
    '''
    정규화 해제된 이미지를 새 텐서로 반환해 후속 시각화/저장에 사용.
    '''
    return undo_preprocess(x, mean=mean, std=std)
