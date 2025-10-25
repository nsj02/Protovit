"""ProtoViT 학습 엔트리 포인트.

데이터 로더를 구성하고, PPNet을 초기화한 뒤 warm->joint->slot->finetune 순으로
학습 단계를 진행하며 프로토타입 push 및 체크포인트 저장까지 처리한다."""

import os
import shutil
import copy
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms

# 기본 윈도우/주기 값이 비어 있을 때 사용할 기본값을 환경변수로 설정한다.
# 설정 단계 이후 settings.py에서 다시 가져다 쓰기 때문에 여기서도 일관된 값이 유지된다.

os.environ.setdefault("PROTOVIT_WINDOW", "20d")
os.environ.setdefault("PROTOVIT_IMAGE_FREQ", "week")

from datasets import ProtovitStockDataset, StockMemmapConfig, StockMemmapShard, FilterSpec
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

# ---------------------------------------------------------------------------
# 커맨드라인 인자 정의: 학습 시 사용할 GPU, 재개용 체크포인트, 체크포인트 저장 주기
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')  # python3 main.py -gpuid=0,1,2,3
parser.add_argument('--resume-checkpoint', type=str, default=None,
                    help='체크포인트 경로를 지정하면 해당 지점에서 학습을 재개합니다.')
parser.add_argument('--checkpoint-interval', type=int, default=1,
                    help='자동 체크포인트 저장 간격(epoch). 0이면 비활성화.')
parser.add_argument('--annual-topn', type=int, default=None,
                    help='연(샤드)별 6월 스냅샷 기준 시가총액 상위 N개 종목만 포함합니다.')
args = parser.parse_args()
# 단일/복수 GPU ID를 문자열로 받아 CUDA_VISIBLE_DEVICES에 반영.
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
# settings.py에 정의된 실험 설정을 모두 불러온다. 학습 파라미터의 싱크를 위해
# 다른 모듈에서도 동일한 상수를 사용하므로 여기서도 그대로 import 한다.
from settings import (
    add_on_layers_type,
    base_architecture,
    build_shard_paths,
    experiment_run,
    image_height,
    image_width,
    img_size,
    is_years,
    label_column,
    label_threshold,
    memmap_root,
    num_classes,
    oos_years,
    prototype_activation_function,
    prototype_shape,
    prefetch_factor,
    radius,
    pin_memory,
    persistent_workers,
    pth_target_accuracy,
    random_split_seed,
    sig_temp,
    test_batch_size,
    train_batch_size,
    train_push_batch_size,
    train_num_workers,
    train_size_ratio,
    eval_num_workers,
    window_size,
)

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

# ---------------------------------------------------------------------------
# 데이터셋 구성: settings에서 정의한 연도 구간에 맞춰 shard 경로를 만들고
# 학습/검증/테스트 DataLoader를 생성한다.
# ---------------------------------------------------------------------------
# load the data
train_shards = build_shard_paths(is_years)
test_shards = build_shard_paths(oos_years)

print("==========================================")
print("========= DATA SHARD SUMMARY =============")
print(f"  - IS years: {is_years[0]}-{is_years[-1]} ({len(train_shards)} shards)")
print(f"  - OOS years: {oos_years[0]}-{oos_years[-1]} ({len(test_shards)} shards)")
print("==========================================")
env_memmap_root = os.environ.get('PROTOVIT_MEMMAP_ROOT')
print(f"[ENV] PROTOVIT_MEMMAP_ROOT={env_memmap_root}")
print(f"[SETTINGS] memmap_root={memmap_root}")

seed = np.random.randint(10, 10000, size=1)[0]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True


def _loader_io_opts(num_workers: int) -> dict:
    workers = max(0, num_workers)
    opts = {
        'num_workers': workers,
        'pin_memory': pin_memory,
    }
    if workers > 0:
        opts['persistent_workers'] = persistent_workers
        opts['prefetch_factor'] = prefetch_factor
    return opts


def _build_config(shards):
    mapped = [StockMemmapShard(images_path=img, labels_path=lbl) for img, lbl in shards]
    return StockMemmapConfig(
        shards=mapped,
        window_size=window_size,
        height=image_height,
        width=image_width,
        label_column=label_column,
        target_threshold=label_threshold,
    )


train_config = _build_config(train_shards)
train_filter = FilterSpec(annual_topn=args.annual_topn) if args.annual_topn else None
full_train_dataset = ProtovitStockDataset(train_config, filter_spec=train_filter)
total_samples = len(full_train_dataset)
train_size = int(total_samples * train_size_ratio)
train_size = max(1, min(train_size, total_samples - 1))
validate_size = total_samples - train_size
if validate_size == 0:
    validate_size = 1
    train_size = total_samples - validate_size

split_generator = torch.Generator().manual_seed(random_split_seed)
train_subset, validate_subset = torch.utils.data.random_split(
    full_train_dataset, [train_size, validate_size], generator=split_generator
)

train_loader = torch.utils.data.DataLoader(
    train_subset,
    batch_size=train_batch_size,
    shuffle=True,
    drop_last=True,
    **_loader_io_opts(train_num_workers),
)

push_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(size=(img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])
train_push_dataset = ProtovitStockDataset(train_config, transform=push_transform, filter_spec=train_filter)
train_push_subset = torch.utils.data.Subset(train_push_dataset, train_subset.indices)
train_push_loader = torch.utils.data.DataLoader(
    train_push_subset,
    batch_size=train_push_batch_size,
    shuffle=False,
    **_loader_io_opts(eval_num_workers),
)

validate_loader = torch.utils.data.DataLoader(
    validate_subset,
    batch_size=test_batch_size,
    shuffle=False,
    **_loader_io_opts(eval_num_workers),
)

log(f"env PROTOVIT_MEMMAP_ROOT: {env_memmap_root}")
log(f"memmap_root: {memmap_root}")
log(f'In-sample train samples: {len(train_subset)} | validation samples: {len(validate_subset)}')

test_config = _build_config(test_shards)
oos_filter = FilterSpec(annual_topn=args.annual_topn) if args.annual_topn else None
oos_dataset = ProtovitStockDataset(test_config, filter_spec=oos_filter)
oos_loader = torch.utils.data.DataLoader(
    oos_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    **_loader_io_opts(eval_num_workers),
)
log(f'OOS samples: {len(oos_dataset)}')

# use validation loader inside training loop
test_loader = validate_loader


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

checkpoint_dir = os.path.join(model_dir, 'checkpoints')
makedir(checkpoint_dir)
checkpoint_interval = max(0, args.checkpoint_interval)
resume_checkpoint = args.resume_checkpoint
# PROTOVIT_WARM_RESUME: resume from best warm snapshot, PROTOVIT_RESUME: generic resume
resume_ckpt_env = os.environ.get('PROTOVIT_WARM_RESUME') or os.environ.get('PROTOVIT_RESUME')
if resume_ckpt_env:
    resume_checkpoint = resume_ckpt_env

# 각 단계의 진행 상황을 stage_state에 기록해 두면 중간에 재개할 때도 정확한 위치부터 이어진다.
stage_state = {
    'phase': 'warm_joint',
    'warm_joint_epoch': 0,
    'slots_epoch': 0,
    'finetune_epoch': 0
}

# 단계별 학습률과 업데이트 대상이 달라 optimizer를 분리해 둔다.
optimizers = {
    'warm': warm_optimizer,
    'joint': joint_optimizer,
    'joint_stage2': joint_optimizer2,
    'last_layer': last_layer_optimizer,
}

# optimizer와 동일한 키를 사용해 scheduler도 관리한다.
schedulers = {
    'joint': joint_lr_scheduler,
    'joint_stage2': joint_lr_scheduler2,
}

if resume_checkpoint:
    checkpoint = save.load_full_checkpoint(resume_checkpoint, device)

    # phase를 강제로 지정하면 warm/joint/slots/push/finetune 어느 지점부터든 재시작할 수 있다.
    resume_phase = os.environ.get('PROTOVIT_WARM_PHASE')  # override stage phase
    if resume_phase:
        checkpoint.setdefault('stage_state', {})
        checkpoint['stage_state']['phase'] = resume_phase
        if resume_phase == 'warm_joint':
            checkpoint['stage_state'].setdefault('warm_joint_epoch', checkpoint['stage_state'].get('warm_joint_epoch', 0))
        if resume_phase == 'slots':
            checkpoint['stage_state'].setdefault('slots_epoch', 0)
        if resume_phase == 'push':
            checkpoint['stage_state'].setdefault('slots_epoch', slots_train_epoch)
            checkpoint['stage_state'].setdefault('finetune_epoch', 0)
        if resume_phase == 'finetune':
            checkpoint['stage_state'].setdefault('finetune_epoch', 0)

    # 필요한 경우 warm/slots/finetune 각 단계의 epoch 오프셋을 덮어쓸 수 있다.
    resume_warm_epoch = os.environ.get('PROTOVIT_WARM_EPOCH')
    if resume_warm_epoch:
        checkpoint.setdefault('stage_state', {})
        checkpoint['stage_state']['warm_joint_epoch'] = int(resume_warm_epoch)

    resume_slots_epoch = os.environ.get('PROTOVIT_SLOTS_EPOCH')
    if resume_slots_epoch:
        checkpoint.setdefault('stage_state', {})
        checkpoint['stage_state']['slots_epoch'] = int(resume_slots_epoch)

    resume_finetune_epoch = os.environ.get('PROTOVIT_FINETUNE_EPOCH')
    if resume_finetune_epoch:
        checkpoint.setdefault('stage_state', {})
        checkpoint['stage_state']['finetune_epoch'] = int(resume_finetune_epoch)

    ppnet.load_state_dict(checkpoint['model_state'])
    for name, opt in optimizers.items():
        opt_state = checkpoint.get('optimizers', {}).get(name)
        if opt_state is not None:
            opt.load_state_dict(opt_state)
    for name, sched in schedulers.items():
        sched_state = checkpoint.get('schedulers', {}).get(name)
        if sched_state is not None:
            sched.load_state_dict(sched_state)
    stage_state = checkpoint.get('stage_state', stage_state)
    log(f"Resumed from checkpoint: {resume_checkpoint} (phase={stage_state.get('phase', 'warm_joint')})")


def maybe_save_checkpoint(filename: str) -> None:
    if checkpoint_interval <= 0:
        return
    path = os.path.join(checkpoint_dir, filename)
    save.save_full_checkpoint(ppnet, optimizers, schedulers, stage_state, path, log=log)


if stage_state.get('phase', 'warm_joint') == 'warm_joint':
    start_epoch = stage_state.get('warm_joint_epoch', 0)
    best_warm = None  # {'epoch': int, 'acc': float, 'path': str}
    for epoch in range(start_epoch, num_train_epochs):
        stage_state['phase'] = 'warm_joint'
        log('epoch: 	{0}'.format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet, log=log)
            _, train_loss = tnt.train(model=ppnet, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k, sum_cls = sum_cls)
        else:
            tnt.joint(model=ppnet, log=log)
            _ , train_loss = tnt.train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k,sum_cls = sum_cls)
            joint_lr_scheduler.step()

        accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                        class_specific=class_specific, log=log, clst_k = k,sum_cls = sum_cls)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=pth_target_accuracy, log=log)
        if (best_warm is None) or (accu > best_warm['acc']):
            candidate_path = os.path.join(checkpoint_dir, f'best_warm_epoch{epoch:03d}.pt')
            save.save_full_checkpoint(ppnet, optimizers, schedulers, stage_state, candidate_path, log=log)
            if best_warm and os.path.exists(best_warm['path']):
                try:
                    os.remove(best_warm['path'])
                except OSError:
                    pass
            best_warm = {'epoch': epoch, 'acc': accu, 'path': candidate_path}
        stage_state['warm_joint_epoch'] = epoch + 1  # 다음 반복에서 재개할 수 있도록 epoch 인덱스를 기록
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            maybe_save_checkpoint(f'warm_joint_epoch{epoch + 1:03d}.pt')  # 중간 재개용 full checkpoint

    if best_warm is not None:
        log(f"Reloading warm epoch {best_warm['epoch']} (acc={best_warm['acc']:.4f})")
        checkpoint = save.load_full_checkpoint(best_warm['path'], device)
        ppnet.load_state_dict(checkpoint['model_state'])
        for name, opt in optimizers.items():
            opt_state = checkpoint.get('optimizers', {}).get(name)
            if opt_state is not None:
                opt.load_state_dict(opt_state)
        for name, sched in schedulers.items():
            sched_state = checkpoint.get('schedulers', {}).get(name)
            if sched_state is not None:
                sched.load_state_dict(sched_state)
        stage_state['warm_joint_epoch'] = best_warm['epoch'] + 1
    stage_state['phase'] = 'slots'
    stage_state.setdefault('slots_epoch', 0)
    maybe_save_checkpoint('warm_joint_complete.pt')  # 슬롯 단계 진입 전 상태를 하나 저장해 둔다.

from settings import coefs_slots
coh_weight = coefs_slots['coh']
coefs['coh'] = coh_weight
log(f'Coefs for slots training: {coefs}')

if stage_state.get('slots_epoch', 0) >= slots_train_epoch and stage_state.get('phase') == 'slots':
    stage_state['phase'] = 'push'

if stage_state.get('phase') == 'slots':
    # 슬롯 훈련: patch selection 모듈만 업데이트하여 웜업에서 학습한 프로토타입을 정교화한다.
    start_slots = stage_state.get('slots_epoch', 0)
    for epoch in range(start_slots, slots_train_epoch):
        tnt.joint(model=ppnet, log=log)
        log('epoch: \t{0}'.format(epoch))
        _ , train_loss = tnt.train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer2,
                        class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k,sum_cls = sum_cls)
        joint_lr_scheduler2.step()
        accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                        class_specific=class_specific, log=log, clst_k=k, sum_cls = sum_cls)

        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'slots', accu=accu,
                                    target_accu=pth_target_accuracy, log=log)

        stage_state['slots_epoch'] = epoch + 1
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            maybe_save_checkpoint(f'slots_epoch{epoch + 1:02d}.pt')

    stage_state['phase'] = 'push'
    maybe_save_checkpoint('slots_complete.pt')

if stage_state.get('phase') == 'push':
    # push 단계: 프로토타입을 실제 이미지에 매핑하여 대표 패턴을 갱신한다.
    last_epoch_idx = max(stage_state.get('warm_joint_epoch', num_train_epochs) - 1, 0)
    push_greedy.push_prototypes(
            train_push_loader,
            pnet=ppnet,
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function,
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir,
            epoch_number=last_epoch_idx,
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)

    accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                    class_specific=class_specific, log=log, clst_k = k,sum_cls = sum_cls)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(last_epoch_idx) + 'push', accu=accu,
                                target_accu=0.0, log=log)

    stage_state['phase'] = 'finetune'
    stage_state.setdefault('finetune_epoch', 0)
    maybe_save_checkpoint('after_push.pt')  # push 완료 스냅샷을 남겨둔다.

num_finetune_epochs = 15
if stage_state.get('finetune_epoch', 0) >= num_finetune_epochs and stage_state.get('phase') == 'finetune':
    stage_state['phase'] = 'done'

if stage_state.get('phase') == 'finetune':
    # 마지막 분류기만 조정하여 전체 프로토타입이 고정된 상태에서 결정경계를 세밀하게 다듬는다.
    start_finetune = stage_state.get('finetune_epoch', 0)
    for epoch in range(start_finetune, num_finetune_epochs):
        tnt.last_only(model=ppnet, log=log)
        log('iteration: \t{0}'.format(epoch))
        _, train_loss = tnt.train(model=ppnet, dataloader=train_loader, optimizer=last_layer_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = k, sum_cls = sum_cls)
        print('Accuracy is:')
        accu, tst_loss = tnt.test(model=ppnet, dataloader=test_loader,
                            class_specific=class_specific, log=log, clst_k = k,sum_cls = sum_cls)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'finetuned', accu=accu,
                                target_accu=pth_target_accuracy, log=log)

        stage_state['finetune_epoch'] = epoch + 1
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            maybe_save_checkpoint(f'finetune_epoch{epoch + 1:02d}.pt')

    stage_state['phase'] = 'done'
    maybe_save_checkpoint('finished.pt')  # 전체 학습이 완료된 시점의 최종 체크포인트

# Evaluate out-of-sample performance after training
# OOS 데이터셋은 학습/검증에 포함되지 않았기 때문에 여기서 최종 일반화 성능을 확인한다.
accu, oos_loss = tnt.test(model=ppnet, dataloader=oos_loader,
                    class_specific=class_specific, log=log, clst_k = k, sum_cls = sum_cls)
log(f'Final OOS accuracy: {accu:.4f}, loss: {oos_loss:.4f}')

logclose()
