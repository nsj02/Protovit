import Augmentor
import os

# CUB-200 데이터 학습 이미지를 증강해 `train_cropped_augmented/`에 저장하는 스크립트.
# 클래스별 디렉터리를 순회하면서 서로 다른 파이프라인을 구성해 회전/스큐/시어 변환을 반복 적용한다.
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    # 증강 결과를 저장할 디렉터리가 없을 경우 즉시 생성 (중첩 디렉터리 포함).
    if not os.path.exists(path):
        os.makedirs(path)

datasets_root_dir = './datasets/cub200_cropped/'
dir = datasets_root_dir + 'train_cropped/'
target_dir = datasets_root_dir + 'train_cropped_augmented/'

# 원본(train_cropped)과 증강(target) 디렉터리가 클래스별 폴더 구조를 동일하게 가지도록 준비.
makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]
    # --- 회전 기반 증강 파이프라인 ---
    # rotation
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    # 좌/우 15도 범위에서 항상 회전하도록 설정.
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    # 좌우 반전은 50% 확률로 적용해 시각적 다양성 확보.
    p.flip_left_right(probability=0.5)
    # 각 파이프라인은 10회 반복 실행하여 클래스별 샘플 수를 증가시킨다.
    for i in range(10):
        p.process()
    del p
    # --- 스큐 기반 증강 파이프라인 ---
    # skew
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    # 기하학적 왜곡(skew)을 항상 적용, magnitude=0.2는 약 45도에 해당.
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # --- 시어 기반 증강 파이프라인 ---
    # shear
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    # 좌/우 방향으로 최대 10도 기울이는 시어 변환.
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # random_distortion
    #p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    #p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    #p.flip_left_right(probability=0.5)
    #for i in range(10):
    #    p.process()
    #del p
