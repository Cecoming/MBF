import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch_mbf import vit_base_patch16_224_ETransReID, vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID, vit_base_patch16_224_MBF
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss, MArcface

def get_background_features(background_masks, idx, feat_patch, feat_global):
    """
    Args:
    - background_masks: Tensor of shape [B, P], initially False
    - idx: Tensor of indices with shape [B, T]
    - feat_patch: Tensor of features with shape [B, P, D]
    - feat_global: Tensor of global features with shape [B, D]
    
    Returns:
    - The sum of cosine similarities between selected patch features and global features.
    """
    # 设置background_masks中idx指定的位置为True
    B, T = idx.shape
    P = background_masks.shape[1]
    arange_b = torch.arange(B)[:, None].expand(B, T)
    background_masks[arange_b, idx] = True
    
    # 选择background_masks为True的位置对应的feat_patch
    selected_feat_patch = feat_patch[background_masks]
    
    # 重塑feat_global以便进行广播操作
    feat_global_expanded = feat_global.unsqueeze(1).expand(-1, P, -1)
    selected_feat_global = feat_global_expanded[background_masks]
    
    # 计算余弦相似度
    cos_sim = F.cosine_similarity(selected_feat_patch, selected_feat_global, dim=-1)
    
    # 返回余弦相似度的绝对值的平均值
    return cos_sim.abs().mean()

class ConvTransform(nn.Module):
    def __init__(self, query_num, patch_num):
        super(ConvTransform, self).__init__()
        self.query_num = query_num
        self.patch_num = patch_num
        # 定义一个1x1的卷积层
        self.conv1d = nn.Conv1d(in_channels=query_num, out_channels=patch_num, kernel_size=1)

    def forward(self, x):
        # x shape: [query_num, 768]
        # x = x.unsqueeze(0)  # 添加batch维度，使其变为 [1, query_num, 768]
        x = self.conv1d(x)   # 应用卷积层，输出形状为 [1, patch_num, 768]
        # x = x.squeeze(0)      # 去除batch维度，变为 [patch_num, 768]
        return x
    
class Gpart_Block(nn.Module):
    def __init__(self, norm_layer, block_layer):
        super(Gpart_Block, self).__init__()
        self.norm_layer = norm_layer
        self.block_layer = block_layer

    def forward(self, x):
        x, _, _ = self.block_layer(x)
        x = self.norm_layer(x)
        return x

########################################
#             Classifiers              #
########################################

class PatchToPartClassifier(nn.Module):
    def __init__(self, dim_output, parts_num):
        super(PatchToPartClassifier, self).__init__()
        self.bn = torch.nn.BatchNorm2d(dim_output)
        self.classifier = nn.Conv2d(in_channels=dim_output, out_channels=parts_num + 1, kernel_size=1, stride=1, padding=0)
        self._init_params()

    def forward(self, x):
        # [B,H,W,D] -> [B,D,H,W]
        x = x.permute(0, 3, 1, 2)
        x = self.bn(x)
        x = self.classifier(x)
        # [B,K,H,W] -> [B,H,W,D]
        # x = x.permute(0, 2, 3, 1)
        return x    # [B,K,H,W]

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm1 = self.norm1.requires_grad_()
        self.norm2 = nn.LayerNorm(d_model)
        self.norm2 = self.norm2.requires_grad_()
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, global_feat, prototype, pos, query_pos):
        prototype = self.with_pos_embed(prototype, query_pos)

        out_prototype = self.multihead_attn(query=prototype, key=global_feat, value=global_feat)[0]    
        prototype = prototype + self.dropout2(out_prototype)
        prototype = self.norm2(prototype)
        prototype = self.linear2(self.dropout(self.activation(self.linear1(prototype))))
        prototype = prototype + self.dropout3(prototype)
        prototype = self.norm3(prototype)
        return prototype
    
    def forward(self, global_feat, prototype, pos=None, query_pos=None):
        return self.forward_post(global_feat, prototype, pos, query_pos)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): #nn.GELU
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, prototype, global_feat,
                pos = None, query_pos = None):
        output = global_feat
        intermediate = []
        for layer in self.layers:
            output = layer(output, prototype, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):# drop_loc, base_keep_rate
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        #########################evit##################################
        drop_loc = (10,)    # hyper-parameters
        base_keep_rate = cfg.MODEL.EFFECTIVE_RADIO   # hyper-parameters

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT, attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        base_keep_rate=base_keep_rate, drop_loc=drop_loc)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier_middle = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_middle.apply(weights_init_classifier)
        self.bottleneck_middle = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_middle.bias.requires_grad_(False)
        self.bottleneck_middle.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None, keep_rate=None, get_idx=False):
        if not get_idx:
            [feat_middle, global_feat] = self.base(x, cam_label=cam_label, view_label=view_label, keep_rate=keep_rate)  # (b, 768
        else:
            [feat_middle, global_feat], idx = self.base(x, cam_label=cam_label, view_label=view_label, keep_rate=keep_rate, get_idx=get_idx)

        feat = self.bottleneck(global_feat)
        feat_middle = self.bottleneck_middle(feat_middle)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_middle_score = self.classifier_middle(feat_middle)

            if not get_idx:
                return [cls_score,cls_middle_score], global_feat  # global feature for triplet loss
            else:
                return [cls_score,cls_middle_score], global_feat, idx  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                # TODO:add feat_middle
                if get_idx:
                    return global_feat, idx
                else:
                    return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0
        drop_loc = (10,)    # hyper-parameters
        base_keep_rate = cfg.MODEL.EFFECTIVE_RADIO   # hyper-parameters
        ############# memory ###############

        ############# decoder ##############
         # decoderlayer config
        self.num_head = cfg.MODEL.NUM_HEAD
        self.dim_forward = 2048
        self.decoder_drop = cfg.MODEL.DECODER_DROP_RATE
        self.drop_first = cfg.MODEL.DROP_FIRST

        # decoder config
        self.decoder_numlayer = cfg.MODEL.NUM_DECODER_LAYER
        self.decoder_norm = nn.LayerNorm(self.in_planes)
        
        # query setting
        self.num_query = cfg.MODEL.QUERY_NUM
        # self.num_query = 128
        self.query_embed = nn.Embedding(self.num_query, self.in_planes).weight
        f_size = ((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0]+1, (cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1]+1)
        self.query_trasform = ConvTransform(self.num_query, f_size[0]*f_size[1])

        # part view based decoder
        self.transformerdecoderlayer = TransformerDecoderLayer(self.in_planes, self.num_head, self.dim_forward, self.decoder_drop, "relu", self.drop_first)
        self.transformerdecoder = TransformerDecoder(self.transformerdecoderlayer, self.decoder_numlayer, self.decoder_norm)
        ######################################

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, 
                                                        local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        base_keep_rate=base_keep_rate, drop_loc=drop_loc)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.last_block = copy.deepcopy(block)
        self.last_norm = copy.deepcopy(layer_norm)
        self.local_block = copy.deepcopy(block)
        self.local_norm = copy.deepcopy(layer_norm)
        self.gpart_block = Gpart_Block(self.local_norm, self.local_block)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

            self.classifier_1_decoder = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1_decoder.apply(weights_init_classifier)
            self.classifier_2_decoder = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2_decoder.apply(weights_init_classifier)
            self.classifier_3_decoder = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3_decoder.apply(weights_init_classifier)
            self.classifier_4_decoder = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4_decoder.apply(weights_init_classifier)


        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.patch_classifier = PatchToPartClassifier(self.in_planes, 4)
        self.bottleneck_1_decoder = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1_decoder.bias.requires_grad_(False)
        self.bottleneck_1_decoder.apply(weights_init_kaiming)
        self.bottleneck_2_decoder = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2_decoder.bias.requires_grad_(False)
        self.bottleneck_2_decoder.apply(weights_init_kaiming)
        self.bottleneck_3_decoder = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3_decoder.bias.requires_grad_(False)
        self.bottleneck_3_decoder.apply(weights_init_kaiming)
        self.bottleneck_4_decoder = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4_decoder.bias.requires_grad_(False)
        self.bottleneck_4_decoder.apply(weights_init_kaiming)


    def forward(self, x, label=None, cam_label= None, view_label=None, keep_rate=None, get_idx=False, f_size=(16, 8), is_occ=True):
        global_feat, idx, attn, attns = self.base(x, cam_label=cam_label, view_label=view_label, keep_rate=keep_rate, get_idx=get_idx)

        # out_feature: 11th feat, 12th feat
        feat_global = self.base.norm(global_feat[:,0]) # [B, C]
        feat_patch = global_feat[:,1:]  # [B, P, C]

        feat_patch_reshaped = feat_patch.reshape(feat_patch.shape[0], f_size[0], f_size[1], -1) #[B, K+1, fH, fW]
        patch_cls_scores = self.patch_classifier(feat_patch_reshaped)  # [B, K+1, fH, fW]
        patch_cls_scores = F.softmax(patch_cls_scores, dim=1)  # [B, K+1, fH, fW], return for segment loss

        patch_cls_scores_reshaped = patch_cls_scores.reshape(patch_cls_scores.shape[0], patch_cls_scores.shape[1], -1).contiguous()  # [B, K+1, P]
        background_masks = patch_cls_scores_reshaped[:, 0]  # [B, P]
        parts_masks = patch_cls_scores_reshaped[:, 1:]   # [B, K, P]

        THRESHOLD = 0.4
        # 将uneffective patch设为背景, 返回对应背景特征，for 正交化损失
        _idx = idx[0]
        background_threshold = background_masks > THRESHOLD
        loss_orth = get_background_features(background_threshold, _idx, feat_patch, feat_global)

        # 衡量距离时加入可视得分权重
        use_visiable_matrix = False
        if use_visiable_matrix:
            visiable_matrix, _ = torch.max(parts_masks, dim=2)  # [B, P]
            visiable_matrix[visiable_matrix >= THRESHOLD] = 1
            visiable_matrix[visiable_matrix < THRESHOLD] = 0
            global_weight = torch.ones(visiable_matrix.shape[0], 1, device=visiable_matrix.device)
            visiable_matrix = torch.cat((global_weight, visiable_matrix), dim=1)    # [B, 1+P]

        part_threshold = parts_masks > THRESHOLD   # [B, K, P]

        ## 2. global-local patch get part_features
        # 将 `threshold` 的形状从 `[B, K, P]` 转换为 `[B, K, P, C]`
        part_threshold = part_threshold.unsqueeze(-1).expand(-1, -1, -1, feat_patch.shape[-1])
        # # 从 `feat_patch` 中提取 K 个部位的 Patch，其余部位置为 0
        patches = torch.where(part_threshold > 0, feat_patch.unsqueeze(1), torch.zeros_like(feat_patch).unsqueeze(1))

        # # 将形状从 `[B, K, P, C]` 转换为 `[K, B, P, D]`
        patches = patches.permute(1, 0, 2, 3)  # [K, B, P, D]

        token = feat_global.clone().unsqueeze(1)                # [B, 1, D]
        b1_local_feat = torch.cat((token, patches[0]), dim=1)   # [B, K+1, 768]
        b1_local_feat = self.gpart_block(b1_local_feat)         # [B, K+1, 768]
        local_feat_1 = b1_local_feat[:, 0]                      # [64, 768]

        b2_local_feat = torch.cat((token, patches[1]), dim=1)   # [B, K+1, 768]
        b2_local_feat = self.gpart_block(b2_local_feat)         # [B, K+1, 768]
        local_feat_2 = b2_local_feat[:, 0]                      # [64, 768]

        b3_local_feat = torch.cat((token, patches[2]), dim=1)   # [B, K+1, 768]
        b3_local_feat = self.gpart_block(b3_local_feat)         # [B, K+1, 768]
        local_feat_3 = b3_local_feat[:, 0]                      # [64, 768]

        b4_local_feat = torch.cat((token, patches[3]), dim=1)   # [B, K+1, 768]
        b4_local_feat = self.gpart_block(b4_local_feat)         # [B, K+1, 768]
        local_feat_4 = b4_local_feat[:, 0]                      # [64, 768]

        feat_global_bn = self.bottleneck(feat_global)           # [64, 768]
        local_feat_1_bn = self.bottleneck_1(local_feat_1)       # [64, 768]
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)
        en_feat_bn = [feat_global_bn, local_feat_1_bn/4, local_feat_2_bn/4, local_feat_3_bn/4, local_feat_4_bn/4]
        en_feat_inf = [feat_global, local_feat_1/4, local_feat_2/4, local_feat_3/4, local_feat_4/4]
        #----------------------------------------------------------------------------------
        if is_occ:
            # decoder
            # part views
            parts_masks = torch.sum(parts_masks, dim=1).unsqueeze(-1)   # [bs, P]
            decoder_value = feat_patch * parts_masks                    # [bs, 128, 768]
            
            decoder_value = decoder_value.permute(1,0,2)                # [128, bs, 768]
            query_embed = self.query_embed                              # [4, 768]
            bs = global_feat.shape[0]
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)     # [4, bs, 768]
            prototype = torch.zeros_like(query_embed)
            
            # part-view based decoder
            de_out = self.transformerdecoder(prototype, decoder_value, query_pos=query_embed)  # [128, bs, 768]
            de_out = de_out.permute(1, 0, 2)                # [bs, 4, 768]
            de_out = self.query_trasform(de_out)            # [bs, 128, 768]
            de_patch = de_out                                       # [bs, 128, 768]
            
            with torch.no_grad():
                # decoder part mask
                de_patch_reshaped = de_patch.reshape(de_patch.shape[0], f_size[0], f_size[1], -1) #[B, K+1, fH, fW]
                de_patch_cls_scores = self.patch_classifier(de_patch_reshaped)  # [B, K+1, fH, fW]
                de_patch_cls_scores = F.softmax(de_patch_cls_scores, dim=1)  # [B, K+1, fH, fW], return for segment loss
                
                de_patch_cls_scores_reshaped = de_patch_cls_scores.reshape(de_patch_cls_scores.shape[0], de_patch_cls_scores.shape[1], -1).contiguous()  # [B, K+1, P]
                de_parts_masks = de_patch_cls_scores_reshaped[:, 1:]   # [B, K, P]
                de_part_threshold = de_parts_masks > THRESHOLD   # [B, K, P]
                de_part_threshold = de_part_threshold.unsqueeze(-1).expand(-1, -1, -1, de_patch.shape[-1])
                # 从 `feat_patch` 中提取 K 个部位的 Patch，其余部位置为 0
                # TODO:debug 查看de_part_threshold
                de_patches = torch.where(de_part_threshold > 0, de_patch.unsqueeze(1), torch.zeros_like(de_patch).unsqueeze(1))
                # # 将形状从 `[B, K, P, C]` 转换为 `[K, B, P, D]`
                de_patches = de_patches.permute(1, 0, 2, 3)  # [K, B, P, D]           # [B, 1, D]
                de_token = token

            de_b1_local_feat = torch.cat((de_token, de_patches[0]), dim=1)   # [B, K+1, 768]
            de_b1_local_feat = self.gpart_block(de_b1_local_feat)         # [B, K+1, 768]
            de_local_feat_1 = de_b1_local_feat[:, 0]                      # [64, 768]

            de_b2_local_feat = torch.cat((de_token, de_patches[1]), dim=1)   # [B, K+1, 768]
            de_b2_local_feat = self.gpart_block(de_b2_local_feat)         # [B, K+1, 768]
            de_local_feat_2 = de_b2_local_feat[:, 0]                      # [64, 768]

            de_b3_local_feat = torch.cat((de_token, de_patches[2]), dim=1)   # [B, K+1, 768]
            de_b3_local_feat = self.gpart_block(de_b3_local_feat)         # [B, K+1, 768]
            de_local_feat_3 = de_b3_local_feat[:, 0]                      # [64, 768]

            de_b4_local_feat = torch.cat((de_token, de_patches[3]), dim=1)   # [B, K+1, 768]
            de_b4_local_feat = self.gpart_block(de_b4_local_feat)         # [B, K+1, 768]
            de_local_feat_4 = de_b4_local_feat[:, 0]                      # [64, 768]
            
            de_local_feat_1_bn = self.bottleneck_1_decoder(de_local_feat_1)   # [64, 768]
            de_local_feat_2_bn = self.bottleneck_2_decoder(de_local_feat_2)
            de_local_feat_3_bn = self.bottleneck_3_decoder(de_local_feat_3)
            de_local_feat_4_bn = self.bottleneck_4_decoder(de_local_feat_4)

            
            de_feat_inf = [de_local_feat_1/4, de_local_feat_2/4, de_local_feat_3/4, de_local_feat_4/4]
            de_feat_bn = [de_local_feat_1_bn/4, de_local_feat_2_bn/4, de_local_feat_3_bn/4, de_local_feat_4_bn/4]
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_global_bn, label)
            else:
                if is_occ:
                    de_cls_score_1 = self.classifier_1_decoder(de_local_feat_1_bn)
                    de_cls_score_2 = self.classifier_2_decoder(de_local_feat_2_bn)
                    de_cls_score_3 = self.classifier_3_decoder(de_local_feat_3_bn)
                    de_cls_score_4 = self.classifier_4_decoder(de_local_feat_4_bn)

                cls_score = self.classifier(feat_global_bn)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            
            if use_visiable_matrix:
                return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4
                    ], en_feat_bn, patch_cls_scores, idx, loss_orth, visiable_matrix
            else:
                if is_occ:
                    return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4
                    ], en_feat_bn, [de_cls_score_1, de_cls_score_2, de_cls_score_3, de_cls_score_4
                    ], de_feat_bn, patch_cls_scores, idx, loss_orth, de_patch, attn, attns
                else:
                    return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4
                    ], en_feat_bn, patch_cls_scores, idx, loss_orth, feat_patch, attn, attns

        else:
            if self.neck_feat == 'after':
                if is_occ:
                    inf_feat = en_feat_bn + de_feat_bn#
                    return torch.cat(inf_feat, dim=1)
                else:
                    return torch.cat(en_feat_bn, dim=1)
            else:
                if is_occ:
                    inf_feat = en_feat_inf + de_feat_inf
                    return torch.cat(inf_feat, dim=1)
                else:
                    return torch.cat(en_feat_inf, dim=1)
                    

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_MBF': vit_base_patch16_224_MBF,
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
