import torch

# Imagenet 통계값(평균/표준편차)을 기반으로 입력 이미지를 정규화하거나 다시 복원할 때 사용하는 헬퍼 모듈.
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def preprocess(x, mean, std):
    # 입력 텐서가 RGB 3채널인지 방어적으로 확인.
    assert x.size(1) == 3
    # 입력과 동일한 크기의 버퍼를 만들어 결과를 저장 (원본 훼손 방지).
    y = torch.zeros_like(x)
    for i in range(3):
        # 각 채널별로 평균을 빼고 표준편차로 나눠 정규화 수행.
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    '''
    pretrained 모델이 기대하는 방식으로 입력 이미지를 정규화한 새 텐서를 반환.
    '''
    return preprocess(x, mean=mean, std=std)

def undo_preprocess(x, mean, std):
    # 정규화된 이미지를 다시 원본 픽셀 분포로 복원.
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        # 채널별로 표준편차를 곱하고 평균을 더해 역정규화 수행.
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def undo_preprocess_input_function(x):
    '''
    정규화된 이미지를 원본 스케일의 텐서로 되돌린 사본을 반환.
    '''
    return undo_preprocess(x, mean=mean, std=std)
