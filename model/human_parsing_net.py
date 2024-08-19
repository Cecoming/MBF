# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
import torchvision
import skimage.measure as M
from scipy import ndimage
from .human_parsing.seg_hrnet import get_seg_model

from PIL import Image

class HumanParsingNet():
    def __init__(self, cfg):
        super(HumanParsingNet, self).__init__()
        self.model = get_seg_model(cfg)
        model_state_file = cfg.HUMAN_PARSING.MODEL_FILE
        pretrained_dict = torch.load(model_state_file)
        model_dict = self.model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.base_size = (cfg.HUMAN_PARSING.IMAGE_SIZE[1], cfg.HUMAN_PARSING.IMAGE_SIZE[0])

    def to(self, device):
        self.model = self.model.to(device)

    def make_parallel(self):
        self.model = nn.DataParallel(self.model)

    def make_distributedParallel(self, device_ids):
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=device_ids, find_unused_parameters=True)

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def largest_connect_component(self, img):
        area_thr = 100
        
        mask = np.zeros_like(img)
        mask[img > 0] = 1
        labeled_mask, num = M.label(mask, connectivity=2, background=0, return_num=True)
        if num <= 1:
            return img
        else:
            for i in range(1, num + 1):
                if np.sum(labeled_mask == i) < area_thr:
                    mask[labeled_mask == i] = 0
            img = img * mask
            return img

    def combine_label(self, img):
        img[img == 5] = 2
        img[img == 6] = 1   
        img[img == 7] = 1
        img[img == 8] = 2
        img[img == 9] = 0
        return img
        # 0 -> background
        # 1 -> head(hair)
        # 2 -> upper
        # 3 -> lower
        # 4 -> shoes
        # 5 -> bag
        # 6 -> head(hat)
        # 7 -> head(face)
        # 8 -> arm
        # 9 -> leg

    def remove_small_area(self, img):
        area_thr = 80
        # import skimage.measure as M
        label_set = set(img.flatten())
        for label in label_set:
            if label == 0:
                continue
            mask = np.zeros_like(img)
            mask[img == label] = 1
            labeled_mask, num = M.label(mask, connectivity=2, background=0, return_num=True)

            for i in range(1, num + 1):
                if np.sum(labeled_mask == i) >= area_thr:
                    mask[labeled_mask == i] = 0
            img[mask > 0] = 0
        return img

    def remove_duplicate_area(self, img):
        num_thr = 2
        # import skimage.measure as M
        label_set = set(img.flatten())
        for label in label_set:
            if label == 0:
                continue
            mask = np.zeros_like(img)
            mask[img == label] = 1
            labeled_mask, num = M.label(mask, connectivity=2, background=0, return_num=True)
            if num <= num_thr:
                continue

            area = [-np.sum(labeled_mask == i) for i in range(1, num + 1)]
            match = np.argsort(area) + 1
            for i in match[:num_thr]:
                mask[labeled_mask == i] = 0
            img[mask > 0] = 0
        return img

    def get_batch_part_masks(self, image_list, sv_pred=False):
        # start_time = time.time()
        with torch.no_grad():
            size = image_list.size()
            hrnet_size = self.base_size # HRNet input size
            image_list = F.interpolate(image_list, size=hrnet_size, mode='bilinear', align_corners=False)
            preds = self.model(image_list)
            # pred_time = time.time()
            
            # preds = F.interpolate(preds, size=(size[-2], size[-1]), mode='bilinear', align_corners=False)
            preds = F.interpolate(preds, size=hrnet_size, mode='bilinear', align_corners=False)
            preds = preds.exp()

            if preds.size()[-2] != size[0] or preds.size()[-1] != size[1]:
                preds = F.interpolate(preds, size=(size[-2], size[-1]), mode='bilinear', align_corners=False)
            
            # interpolate_time = time.time()

            preds = torch.argmax(preds, dim=1).type(torch.uint8) # (b, 285, 113), 找出预测结果中每个位置概率最大的类别，并将结果转换为 uint8 类型的numpy数组
            preds = preds.cpu().numpy()  # (b, 10, 285, 113)
            # numpy_time = time.time()

            for i in range(preds.shape[0]):
                preds[i] = self.largest_connect_component(preds[i])
                preds[i] = self.combine_label(preds[i])           
                preds[i] = self.remove_small_area(preds[i])    
                preds[i] = self.remove_duplicate_area(preds[i])   

            return preds
    
    
    def custom_maxfrequent_downsample(self, image, num_parts, stride):
        batch_size, height, width = image.size()
        kernel_size = 16

        output_height = (height - kernel_size) // stride + 1
        output_width = (width - kernel_size) // stride + 1

        patches = image.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        patches = patches.contiguous().view(batch_size, output_height, output_width, -1)  # [B, OH, OW, K*K]

        counts = torch.zeros(batch_size, output_height, output_width, num_parts + 1, device=image.device)

        for part in range(num_parts + 1):
            counts[..., part] = (patches == part).sum(dim=-1)

        output_tensor = counts.argmax(dim=-1)

        return output_tensor
    
    def get_occ_mask(self, h_image_part_mask, o_mask, image_shape, stride_size):
        kernel_size = 16
        (H, W) = image_shape
        b, fh, fw = h_image_part_mask.shape

        occ_mask = h_image_part_mask.clone()

        for i in range(b):
            mask = o_mask[i]
            if mask is None:
                continue
            
            top, left, mask_h, mask_w = mask
            mask_matrix = torch.zeros((H, W), dtype=torch.uint8, device=h_image_part_mask.device)
            mask_matrix[top:top + mask_h, left:left + mask_w] = 1
            
            unfolded_mask = mask_matrix.unfold(0, kernel_size, stride_size).unfold(1, kernel_size, stride_size)
            
            mask_counts = unfolded_mask.sum(dim=(-1, -2))
            mask_ratios = mask_counts / (kernel_size * kernel_size)

            mask_ratios = mask_ratios[:fh, :fw]
            
            occ_mask[i] = torch.where(mask_ratios > 0.7, torch.tensor(0, dtype=occ_mask.dtype, device=occ_mask.device), occ_mask[i])
        return occ_mask

