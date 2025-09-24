"""ProtoViT 학습/검증 루프와 단계별 gradient 제어를 담당하는 유틸 모듈."""

import time
import torch
from tqdm import tqdm
from helpers import list_of_distances, make_one_hot
import torch.nn.functional as F

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, ema = None, clst_k = 1,sum_cls = True):
    '''
    ProtoViT 학습/평가에 공통으로 사용되는 루프.

    optimizer가 주어지면 학습 모드로, None이면 평가 모드로 동작하며 아래 손실 구성요소를 모두 계산한다.
    - cross_entropy : 분류 손실
    - cluster/separation : 프로토타입이 정답 클래스에는 가깝고 다른 클래스에서는 멀어지도록 유도
    - orth/coherence : 슬롯 간 직교성과 다양성을 유지
    - l1 : 마지막 분류기 sparsity

    최종적으로 (정확도, 손실 딕셔너리)를 반환해 상위 루프(main.py)에서 로깅/저장을 쉽게 처리한다.
    '''
    is_train = optimizer is not None  # optimizer가 있으면 학습 단계, 없으면 평가 단계로 처리
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_orth_loss = 0 
    total_comp_loss = 0 
    total_loss = 0 
    for i, (image, label) in enumerate(dataloader):
        # 배치 단위로 이미지를 GPU에 올리고 손실/정확도를 누적
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        # 학습 시 gradient 계산, 평가 시 no_grad 컨텍스트 사용
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            # 모델은 (분류 logits), (프로토타입별 최소 거리), (slot별 활성값)을 반환한다.
            #  output: [batch, num_classes], min_distances: [batch, num_prototypes],
            #  values: [batch, num_prototypes, num_slots] (프로토타입-슬롯 유사도)
            output, min_distances, values = model(input)
            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)  # [1] 분류 손실 (cross entropy)

            if class_specific:              
                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                # to encourage the prototype to focus on foreground, we calcualted a weighted cluster loss
                # max_dist = (model.prototype_shape[1]
                #             * model.prototype_shape[2])
                #             #* model.prototype_shape[3]) # dim*1*1 
                # 정답 클래스에 속한 프로토타입만 골라 cluster/separation 손실 계산
                prototypes_of_correct_class = torch.t(model.prototype_class_identity[:,label]).cuda()
                # shape: [batch, num_prototypes] -> unsqueeze로 [batch, num_prototypes, 1] 마스크를 만든다.
                prototypes_of_correct_class = prototypes_of_correct_class.unsqueeze(-1)
                # min_distances는 슬롯 활성 합을 n_p에서 뺀 값으로, 작을수록 프로토타입과 가깝다는 의미.
                max_activations = -min_distances

                ### retrieve slots 
                # a soft approximation of 1, 0 s 
                slots = torch.sigmoid(model.patch_select * model.temp)  # shape: [1, num_proto, num_slots] (예: 1×2000×4) slot 선택 확률
                # slots 확률과 values(슬롯별 cosine 유사도)를 곱해 정답 클래스 활성도를 집계한다.
                #factor = ((slots.sum(-1))**0.5).unsqueeze(-1) # 2000, 1, 1
                if clst_k == 1:
                    if not sum_cls: 
                        correct_class_prototype_activations =  values * prototypes_of_correct_class # bsz, 2000, 4
                        correct_class_proto_act_max_sub_patch, _ = torch.max(correct_class_prototype_activations, dim = 2) # bsz, 2000
                        correct_class_prototype_activations, _ = torch.max(correct_class_proto_act_max_sub_patch, dim=1) # bsz 
                    else:
                        # sum_cls=True이면 슬롯 방향으로 먼저 합산해 전체 프로토타입 활성도를 얻는다.
                        correct_class_prototype_activations = (values.sum(-1)) * prototypes_of_correct_class.squeeze(-1) # bsz, 2000, 1
                        correct_class_prototype_activations, _ = torch.max(correct_class_prototype_activations, dim=1) 
                        
                    cluster_cost = torch.mean(correct_class_prototype_activations)  # [2] 클러스터 손실
                else:
                    # clst_k is a hyperparameter that lets the cluster cost apply in a "top-k" fashion:
                    # the original cluster cost is equivalent to the k = 1 case
                    # -> 여러 슬롯을 동시에 고려해 cluster cost를 완화하도록 설계됨.
                    correct_class_prototype_activations =  values * prototypes_of_correct_class # bsz, 2000, 4
                    correct_class_proto_act_max_sub_patch, _ = torch.max(correct_class_prototype_activations, dim = 2) # bsz, 2000
                    top_k_correct_class_prototype_activations, _ = torch.topk(correct_class_proto_act_max_sub_patch,
                                                                              k = clst_k, dim=1)
                    cluster_cost = torch.mean(top_k_correct_class_prototype_activations)

                # calculate separation cost
                prototypes_of_wrong_class = (1 - prototypes_of_correct_class.squeeze(-1)).unsqueeze(-1)
                # inverted_distances_to_nontarget_prototypes, _ = \
                #     torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                # separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
                
                if not sum_cls:
                    incorrect_class_prototype_activations_sub, _ = torch.max(values * prototypes_of_wrong_class, dim=2)# bsz, 2000
                    incorrect_class_prototype_activations, _ = torch.max(incorrect_class_prototype_activations_sub, dim=1) # bsz
                else:
                    #values_slot = (values.clone())*slots
                    incorrect_class_prototype_activations = (values.sum(-1)) * prototypes_of_wrong_class.squeeze(-1)
                    incorrect_class_prototype_activations, _ = torch.max(incorrect_class_prototype_activations, dim=1) 
                separation_cost = torch.mean(incorrect_class_prototype_activations)  # [3] 분리 손실
                
                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(values * prototypes_of_wrong_class, dim=1) / (values.shape[-1]*torch.sum(prototypes_of_wrong_class, dim=1))
                # [3-avg] 분리 손실의 슬롯 평균값: 오답 클래스와의 평균적 인접도가 크면 페널티가 커진다.
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                #optimize orthogonality of prototype_vector, borrowed from tesnet 
                # ortho loss version 1 
                #factor =  (model.prototype_shape[-1])**0.5
                # 직교성/정규성 페널티: 클래스별 프로토타입 슬롯이 서로 다른 방향을 향하도록 Gram matrix를 단위행렬에 가깝게 유지
                prototype_normalized = F.normalize(model.prototype_vectors, p=2, dim=1)  # p=2로 L2 정규화해 코사인 유사도/Gram 계산 시 방향 비교만 남긴다
                cur_basis_matrix = torch.squeeze(prototype_normalized)  # [num_prototypes, dim * num_slots]
                # 클래스별 [num_prototypes_per_class, dim*num_slots] basis로 재배열하여 Gram matrix(I)에 가깝도록 유지
                subspace_basis_matrix = cur_basis_matrix.reshape(model.num_classes, model.num_prototypes_per_class, -1)
                subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix, 1, 2)
                orth_operator = torch.matmul(subspace_basis_matrix, subspace_basis_matrix_T)  # 클래스별 Gram matrix (K×K); diag=1, off-diag≈0일 때 이상적
                I_operator = torch.eye(subspace_basis_matrix.size(1), subspace_basis_matrix.size(1)).cuda()  # 목표로 삼는 단위행렬
                difference_value = orth_operator - I_operator
                orth_cost = torch.sum(torch.relu(torch.norm(difference_value, p=1, dim=[1, 2]) - 0))  # [5] 직교 손실
   
                ### component / slot coherence loss
                """
                슬롯 간 cosine distance를 계산해 서로 지나치게 다른 방향을 보지 않도록 제약한다.
                - proto_norm_k             : [num_proto, feat_dim, num_slots]
                - cos_jk (각 반복에서)     : [num_proto]  -> 슬롯 j와 k 사이의 cosine distance
                - dist_init                : [num_proto, num_slots] 슬롯별 거리 목록
                - slots.squeeze()          : [num_proto, num_slots] (push 단계에서 선택된 슬롯 확률)
                - most_disimilar           : [num_proto]  -> 프로토타입별 가장 큰 슬롯 거리
                - avg_diff                 : 스칼라(coh 손실)
                """
                proto_norm_k = F.normalize(model.prototype_vectors, p=2, dim=1)  # 슬롯 벡터 정규화
                dist_jk = 1 - F.cosine_similarity(proto_norm_k[:, :, 0], proto_norm_k[:, :, 0])  # [num_proto]; 사실상 0 텐서 (shape 맞춤)
                dist_init = torch.tensor([]).cuda()
                for j in range(model.prototype_shape[-1]):  # 기준 슬롯 j
                    dist_jk = 1 - F.cosine_similarity(proto_norm_k[:, :, 0], proto_norm_k[:, :, 0])  # [num_proto] 초기화 (0)
                    for k in range(model.prototype_shape[-1]):  # 비교 슬롯 k
                        cos_jk = 1 - F.cosine_similarity(proto_norm_k[:, :, k], proto_norm_k[:, :, j])  # [num_proto]; 슬롯 j↔k 거리
                        dist_jk += cos_jk  # 슬롯 j 기준으로 모든 k 거리 누적 (shape 유지: [num_proto])
                    # cos_jk는 마지막 비교 슬롯(k 마지막)에서 계산된 거리. unsqueeze(-1) -> [num_proto, 1];
                    # 슬롯별로 붙이면 dist_init shape이 [num_proto, num_slots]가 된다.
                    dist_init = torch.concat((dist_init, cos_jk.unsqueeze(-1)), dim=-1)
                # dist_init: [num_proto, num_slots] (각 슬롯별 거리 목록), slots.squeeze(): 동일 shape
                dist_init_slots = dist_init * slots.squeeze()  # 선택된 슬롯 확률만 반영
                most_disimilar, _ = dist_init_slots.max(-1)  # 슬롯 간 가장 큰 거리 -> [num_proto]
                avg_diff = most_disimilar.sum()  # slot coherence 손실 값 (스칼라); sum이지만 기존 구현에서 이름을 avg_diff로 유지
                # avg_diff가 클수록 슬롯 간 방향 차이가 크다는 뜻 -> coefs['coh'] 가중치로 페널티 부여
                # l2 norm of slots to encourage sparsity 
                if use_l1_mask:
                    # prototype_class_identity: [num_proto, num_classes] -> transpose로 [num_classes, num_proto]
                    # 정답 클래스와 연결된 가중치는 0, 오답 연결만 1로 남겨 L1 페널티를 집중시킨다.
                    l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
                    l1 = (model.last_layer.weight * l1_mask).norm(p=1)  # 오답 연결에 대한 L1 규제
                else:
                    l1 = model.last_layer.weight.norm(p=1)  # 전체 가중치에 대해 L1 규제

            else:
                # 클래스별 구분이 없을 때는 모든 프로토타입 중 최소 거리를 사용한 단순 cluster 손실을 사용.
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)  # 누적된 샘플 수 (정확도 분모)
            n_correct += (predicted == target).sum().item()  # 누적 정답 수
            n_batches += 1
            # 각 손실 항목을 스칼라로 변환해 epoch 종료 후 평균을 계산한다.
            total_cross_entropy += cross_entropy.item()  # [1] cross-entropy 누적
            total_cluster_cost += cluster_cost.item()    # [2] cluster cost 누적
            total_separation_cost += separation_cost.item()  # [3] separation cost 누적
            total_avg_separation_cost += avg_separation_cost.item()  # [3-avg] separation 평균용
            total_orth_loss += orth_cost.item()          # [5] orthogonality 누적
            total_comp_loss += avg_diff.item()           # [6] slot coherence 누적
            # 활성화된 슬롯 총 개수를 세어 num_proto(2000)로 나눔 -> 프로토타입당 평균 활성 슬롯 수.
            avg_number_patch = (slots >= 0.5).sum()/slots.shape[1]
            # slots: [1, num_proto, num_slots]; squeeze(0) -> [num_proto, num_slots], 슬롯 확률을 합산 후 num_proto로 나눠 평균 확률 기록.
            avg_slots = slots.squeeze(0).sum(1)/slots.shape[1]
        # compute gradient and do SGD step
        if is_train:
            # 학습일 때만 손실 가중치 조합으로 backward 수행
            if class_specific:
                if coefs is not None:
                    # 총 손실 = [1]분류 + [2]클러스터 + [3]분리 + [4]L1 + [5]직교 + [6]slot coherence
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1
                          +coefs['orth']*orth_cost
                          +coefs['coh']*avg_diff
                          )
                          #+coefs['slots']*slots_loss)
                    # 로깅용으로 total_loss에 누적해 epoch 평균을 계산한다.
                    total_loss += loss.item()
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1  # 기본 가중치로 [1]+[2]+[3]+[4]
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1  # class_specific=False일 때 [1]+[2]+[4]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)

        del input
        del target
        del output
        del predicted
        #del weighted_min_distance
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))  # 한 epoch이 소요된 시간을 기록해 학습 속도를 추적.
    #log('\tlearning rate info: \t{0}'.format(optimizer))
    log('\ttotal loss: \t{0}'.format(total_loss / n_batches))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\torthogonal loss\t{0}'.format(total_orth_loss/n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\tslot of prototype 0: \t{0}'.format(slots.squeeze()[0]))
    log('\tEstimated avg number of subpatches: \t{0}'.format(avg_number_patch))
    log('\tEstimated avg slots logit: \t{0}'.format(avg_slots))
    
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\tcoherence loss: \t\t{0}%'.format(total_comp_loss / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.last_layer.weight.norm(p=1).item()))
    p = model.prototype_vectors.view(model.num_prototypes, -1).cpu()
    with torch.no_grad():
        # 프로토타입 간 평균 pairwise 거리를 기록해 다양성 유지 여부를 모니터링.
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
    # 호출 측(메인 스크립트)이 tensorboard/CSV 등에 손쉽게 기록할 수 있도록 Loss 요약 dict 반환.
    loss_values = {
    '[1] cross entropy': total_cross_entropy / n_batches,
    '[2] clst': total_cluster_cost / n_batches,
    '[3] sep': total_separation_cost / n_batches,
    '[3-avg] sep_avg': total_avg_separation_cost / n_batches,
    '[4] l1': model.last_layer.weight.norm(p=1).item(),
    '[5] orth': total_orth_loss / n_batches,
    '[6] coh': total_comp_loss / n_batches,
    'acc': n_correct / n_examples * 100}
    return (n_correct / n_examples), loss_values


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print, ema = None, clst_k = 1,sum_cls = True):
    """학습 모드 wrapper. optimizer/EMA/손실 가중치를 전달해 `_train_or_test` 결과를 반환한다."""
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, ema = ema, clst_k =clst_k,sum_cls = sum_cls)


def test(model, dataloader, class_specific=False, log=print, ema = None, clst_k = 1, sum_cls = True):
    """평가용 wrapper. optimizer=None으로 `_train_or_test`를 호출해 (accuracy, loss_dict) 반환."""
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, ema = ema, clst_k = clst_k,sum_cls = sum_cls)


def last_only(model, log=print):
    """백본·프로토타입을 고정하고 마지막 분류기만 학습하도록 gradient 플래그를 설정한다."""
    for p in model.features.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')

# def slots_only(model, log=print):
#     for p in model.features.parameters():
#         p.requires_grad = False


def warm_only(model, log=print):
    """warm 단계: 백본·프로토타입·classifier 모두 requires_grad=True."""
    for p in model.features.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    """warm 이후 joint 단계: 모든 파라미터에 gradient 허용 (warm과 동일)."""
    for p in model.features.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    log('\tjoint')
