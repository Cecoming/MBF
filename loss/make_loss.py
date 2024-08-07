# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from __future__ import print_function
import torch.nn.functional as F
import torch
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss



import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets): 
        batch_size = text_features.shape[0] # [64, 512]
        batch_size_N = image_features.shape[0] # [64, 512]
        # 该掩码标记出text_features和image_features中对应的标签相等的位置
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device)     # [64, 64]
        # 计算text_features和image_features的内积
        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability, 防止浮点值溢出
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() 
        # 计算logits的softmax
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        # 正样本的平均log概率
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # [64]
        loss = - mean_log_prob_pos.mean()

        return loss
    
def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 2) +
                     epsilon, 0.5).unsqueeze(2).expand_as(feature)
    return torch.div(feature, norm)
    
def orthonomal_loss(w):
    B, K, C = w.shape
    w_norm = featureL2Norm(w)
    WWT = torch.matmul(w_norm, w_norm.transpose(1, 2))
    return F.mse_loss(WWT - torch.eye(K).unsqueeze(0).cuda(), torch.zeros(B, K, K).cuda(), size_average=False) / (K*K)

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 768 * 5
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, vis=None, de=False):
            if vis == None: # normal training loss
                if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                    if cfg.MODEL.IF_LABELSMOOTH == 'on':
                        if isinstance(score, list): 
                            ID_LOSS = [xent(scor, target) for scor in score[1:]]    
                            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                            ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                        else:
                            ID_LOSS = xent(score, target)

                        if isinstance(feat, list):
                                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                        else:
                                TRI_LOSS = triplet(feat, target)[0]

                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    else:
                        if isinstance(score, list):
                            if de:
                                PART_ID_LOSS = [F.cross_entropy(scor, target) for scor in score]
                                PART_ID_LOSS = sum(PART_ID_LOSS) / len(PART_ID_LOSS)
                                ID_LOSS = 0.5 * PART_ID_LOSS
                            else:
                                # global loss
                                GLOBAL_ID_LOSS = [F.cross_entropy(scor, target) for scor in score[:1]]
                                GLOBAL_ID_LOSS = sum(GLOBAL_ID_LOSS)
                                # part loss
                                PART_ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                                PART_ID_LOSS = sum(PART_ID_LOSS) / len(PART_ID_LOSS)
                                # PART_ID_LOSS = sum(PART_ID_LOSS)
                                ID_LOSS = 0.5 * PART_ID_LOSS + 0.5 * GLOBAL_ID_LOSS
                        else:
                            ID_LOSS = F.cross_entropy(score, target)

                        if isinstance(feat, list):
                            if de:
                                PART_TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                                PART_TRI_LOSS = sum(PART_TRI_LOSS) / len(PART_TRI_LOSS)
                                TRI_LOSS = 0.5 * PART_TRI_LOSS
                            else:
                                GLOBAL_TRI_LOSS = [triplet(feats, target)[0] for feats in feat[:1]]
                                GLOBAL_TRI_LOSS = sum(GLOBAL_TRI_LOSS)
                                PART_TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                                PART_TRI_LOSS = sum(PART_TRI_LOSS) / len(PART_TRI_LOSS)
                                TRI_LOSS = PART_TRI_LOSS * 0.5 + GLOBAL_TRI_LOSS * 0.5
                        else:
                            TRI_LOSS = triplet(feat, target)[0]

                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    print('expected METRIC_LOSS_TYPE should be triplet'
                        'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
            else:# use vis to get optinal training loss. vis [B, 1+K]
                if isinstance(score, list):
                    # global loss
                    GLOBAL_ID_LOSS = [F.cross_entropy(scor, target) for scor in score[:1]]
                    GLOBAL_ID_LOSS = sum(GLOBAL_ID_LOSS)
                    # part loss
                    part_id_loss = []
                    for i, scor in enumerate(score[1:], start=1):
                        loss = F.cross_entropy(scor, target, reduction='none')
                        loss = loss * vis[:, i]  # 按照vis的值来控制loss是否计算
                        part_id_loss.append(loss.sum() / (vis[:, i].sum() + 1e-12))  # 平均损失，避免除以0
                    PART_ID_LOSS = sum(part_id_loss) / len(part_id_loss)

                    ID_LOSS = 0.5 * PART_ID_LOSS + 0.5 * GLOBAL_ID_LOSS
                else:
                    ID_LOSS = F.cross_entropy(score, target)

                if isinstance(feat, list):
                    GLOBAL_TRI_LOSS = [triplet(feats, target)[0] for feats in feat[:1]]
                    GLOBAL_TRI_LOSS = sum(GLOBAL_TRI_LOSS)
                    
                    part_tri_loss = []
                    for i, feats in enumerate(feat[1:], start=1):
                        loss = triplet(feats, target)[0]
                        loss = loss * vis[:, i]  # 按照vis的值来控制loss是否计算
                        part_tri_loss.append(loss.sum() / (vis[:, i].sum() + 1e-12))  # 平均损失，避免除以0
                    PART_TRI_LOSS = sum(part_tri_loss) / len(part_tri_loss)
                    TRI_LOSS = PART_TRI_LOSS * 0.5 + GLOBAL_TRI_LOSS * 0.5
                else:
                    TRI_LOSS = triplet(feat, target)[0]

                return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


