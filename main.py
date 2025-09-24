"""ProtoViT 학습 엔트리 포인트.

데이터 로더를 구성하고, PPNet을 초기화한 뒤 warm->joint->slot->finetune 순으로
학습 단계를 진행하며 프로토타입 push 및 체크포인트 저장까지 처리한다."""

import os
import shutil
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import push_greedy
import argparse
import re
import numpy as np
import random

from helpers import makedir
import model
#import push 
#import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')  # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
# 단일/복수 GPU ID를 문자열로 받아 CUDA_VISIBLE_DEVICES에 반영.
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

# 실험별로 체크포인트/시각화를 저장할 디렉터리 구성.
model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size,sig_temp, radius

normalize = transforms.Normalize(mean=mean,
                                 std=std)
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)  # Numpy module.
#     #random.seed(seed)  # Python random module.
#     torch.manual_seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

# reproducibility 목적: 무작위 시드를 고정(훈련마다 seed 로그 참고 가능).
seed = np.random.randint(10, 10000, size=1)[0]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#set_seed(seed)
# all datasets
# train set
# 증강 완료된 학습 셋을 로드해 표준화를 적용.
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False, drop_last=True)
# push set
# push 단계에서는 정규화 없이 원본 픽셀 값을 유지. (원본 이미지 가지고 있다가, 시각화/ 저장용으로 원본 이미지 가지고있기?)
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)
# test set
# 테스트 셋은 학습과 동일한 resize+정규화 파이프라인 사용.
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
# ProtoPNet 변형인 PPNet을 설정값에 맞춰 생성.
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              sig_temp = sig_temp,
                              radius = radius,
                              add_on_layers_type=add_on_layers_type)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(str(device))
ppnet = ppnet.to(device)
model_ema = None
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size, k,stage_2_lrs

# warm 이후 전체 네트워크를 공동 업데이트할 optimizer 구성.
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 #{'params': ppnet.patch_select, 'lr': joint_optimizer_lrs['patch_select']},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.AdamW(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

# to train the slots 
# 슬롯 선택 모듈만 업데이트할 stage2 optimizer 구성.
joint_optimizer_specs_stage2 =[{'params': ppnet.patch_select, 'lr': stage_2_lrs['patch_select']}]

joint_optimizer2 = torch.optim.AdamW(joint_optimizer_specs_stage2)
joint_lr_scheduler2 = torch.optim.lr_scheduler.StepLR(joint_optimizer2, step_size=joint_lr_step_size, gamma=0.1)


from settings import warm_optimizer_lrs
# warm 단계에서는 backbone 업데이트를 거의 멈추고 프로토타입 위주로 학습.
warm_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': warm_optimizer_lrs['features'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.AdamW(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
# 마지막 분류기만 업데이트하는 fine-tuning optimizer.
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.AdamW(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start,sum_cls, push_start, slots_train_epoch 

# train the model
log('start training')

log(f'weight coefs are: {coefs}')
import copy

# check_base = False 
# if check_base:
#     # check the performance of base-arch 
#     n_examples = 0
#     n_correct = 0
#     for i, (image, label) in enumerate(test_loader):
#         image = image.to(device)
#         label = label.to(device)
#         out = ppnet.features(image)
#         _, predicted = torch.max(out.data, 1)
#         n_examples += label.size(0)
#         n_correct += (predicted == label).sum().item()
#     log('base-arch acc: \t{0}'.format(n_correct / n_examples * 100))

#slots_epoch = num_warm_epochs +5 
# only_push = False 
# if not only_push:
# not ready for push yet
# 1) warm -> joint 학습을 수행하는 메인 training loop.
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        # warm 단계: backbone을 거의 동결하고 프로토타입만 조정해 초기 수렴을 안정화합니다.
        tnt.warm_only(model=ppnet, log=log)
        _, train_loss = tnt.train(model=ppnet, dataloader=train_loader, optimizer=warm_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k, sum_cls = sum_cls)
    else:
        # joint 단계: backbone과 프로토타입을 함께 업데이트합니다.
        tnt.joint(model=ppnet, log=log)
        # to train the model with no slots 
        _ , train_loss= tnt.train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k,sum_cls = sum_cls)
        joint_lr_scheduler.step()

    # 각 epoch 종료 후 검증 정확도를 확인하고 일정 임계값 이상이면 저장합니다.
    accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                    class_specific=class_specific, log=log, clst_k=k,sum_cls = sum_cls)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.75, log=log)
    # version 1, learn slots before push 
from settings import coefs_slots
# 슬롯 학습 단계(coh 손실)에 맞는 가중치를 덮어써서 사용.
coh_weight = coefs_slots['coh']
coefs['coh']  = coh_weight
log(f'Coefs for slots training: {coefs}')
# 2) 슬롯 선택 모듈 학습 전용 루프.
for epoch in range(slots_train_epoch):
    tnt.joint(model=ppnet, log=log)
    log('epoch: \t{0}'.format(epoch))
    _ , train_loss= tnt.train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer2,
                    class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k,sum_cls = sum_cls)
    joint_lr_scheduler2.step()
    accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                    class_specific=class_specific, log=log, clst_k=k, sum_cls = sum_cls)
    
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'slots', accu=accu,
                                target_accu=0.75, log=log)

# 3) 학습된 프로토타입을 최신 이미지에 맞춰 끌어오는 push 단계.
push_greedy.push_prototypes(
        train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
        pnet = ppnet, # pytorch network with prototype_vectors
        class_specific=class_specific,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        prototype_layer_stride=1,
        root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
        epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
        prototype_img_filename_prefix=prototype_img_filename_prefix,
        prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
        proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
        save_prototype_class_identity=True,
        log=log)
# push 이후 성능을 다시 평가하고 결과를 저장.
accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                class_specific=class_specific, log=log, clst_k = k,sum_cls = sum_cls)
save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                            target_accu=0.0, log=log)

# 4) 마지막 분류기만 재학습하는 finetune 단계.
# 4) 마지막 분류기만 재학습하는 finetune 단계. 작은 학습률로 분류기 안정화.
for epoch in range(15):
    tnt.last_only(model=ppnet, log=log)
    log('iteration: \t{0}'.format(epoch))
    _, train_loss = tnt.train(model=ppnet, dataloader=train_loader, optimizer=last_layer_optimizer,
                class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k, sum_cls = sum_cls)
    print('Accuracy is:')
    accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                        class_specific=class_specific, log=log, clst_k = k,sum_cls = sum_cls)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'finetuned', accu=accu,
                            target_accu=0.70, log=log)
logclose()
