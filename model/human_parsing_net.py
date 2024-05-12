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
from .human_parsing.seg_hrnet import get_seg_model
from .human_parsing import occduke

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
        import skimage.measure as M
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
        img[img == 5] = 0   # for 4 part test
        img[img == 6] = 1   
        img[img == 7] = 1
        # img[img == 8] = 0
        img[img == 8] = 2
        img[img == 9] = 0
        return img

    def remove_small_area(self, img):
        area_thr = 80
        import skimage.measure as M
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
        import skimage.measure as M
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
        self.model.eval()
        with torch.no_grad():
            size = image_list.size()
            hrnet_size = self.base_size # HRNet input size
            image_list = F.interpolate(image_list, size=hrnet_size, mode='bilinear', align_corners=False)
            preds = self.model(image_list)
            
            # preds = F.interpolate(preds, size=(size[-2], size[-1]), mode='bilinear', align_corners=False)
            preds = F.interpolate(preds, size=hrnet_size, mode='bilinear', align_corners=False)
            preds = preds.exp()

            if preds.size()[-2] != size[0] or preds.size()[-1] != size[1]:
                preds = F.interpolate(preds, size=(size[-2], size[-1]), mode='bilinear', align_corners=False)
                
            # test_dataset.save_pred(pred, sv_path, name)
            preds = preds.cpu().numpy().copy()  # (b, 10, 285, 113)
            preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)    # (b, 285, 113), 找出预测结果中每个位置概率最大的类别，并将结果转换为 uint8 类型的numpy数组
            for i in range(preds.shape[0]): # 遍历batch中每一张图片
                preds[i] = self.largest_connect_component(preds[i]) # 保留最大的连通区域
                preds[i] = self.combine_label(preds[i])             # 合并某些类别的标签
                # if preds[i].max() > 5:
                #     print('max:', preds[i].max())
                pred = self.remove_small_area(preds[i])         # 移除面积小于一定阈值的区域。
                pred = self.remove_duplicate_area(preds[i])     # 移除重复区域

            if sv_pred:
                palette = self.get_palette(256)
                save_img = Image.fromarray(pred)
                save_img.putpalette(palette)
                save_img.save('./test_results12.png')

            return preds
    
    def custom_maxtimes_downsample(self, image, K, block_size=(16, 16)):
        # 获取图像的高度和宽度
        batch, height, width = image.shape

        # 计算下采样后的高度和宽度
        down_height = height // block_size[0]
        down_width = width // block_size[1]

        if down_height * block_size[0] != height or down_width * block_size[1] != width:
            # 裁剪图像以便它可以被均匀地分割成块
            crop_height = down_height * block_size[0]
            crop_width = down_width * block_size[1]
            image = image[:, :crop_height, :crop_width]

        # 重新调整 input_array 的形状以便于计算每个 block 的 mode
        reshaped = image.reshape(batch, down_height, block_size[0], down_width, block_size[1])

        # 初始化一个新的数组来存储每个 block 的 histogram
        hist = torch.zeros((batch, down_height, down_width, K+1))

        # 计算每个 block 的 histogram
        for k in range(0, K+1):
            hist[..., k] = torch.sum(reshaped == k, axis=(2, 4))

        # 找到每个 block 中出现次数最多的元素
        block_mode = torch.argmax(hist, axis=-1)

        return block_mode
    
    def visualization_batch(self, iter_index, img, image_part_mask, masked_imgs, eimage_part_mask, path='./visualization0.8'):  # visualization for test
        if img.shape[0] < 64:
            print('batch' + str(iter_index+1) +'\'s size is small than 64, skip visualization.')
            return

        # 检查路径是否存在
        if not os.path.exists(path):
            # 如果路径不存在，则创建
            os.makedirs(path)

        ##1. image
        # 假设 img 是一个形状为 [B, 3, 256, 128] 的张量
        # 使用 torchvision.utils.make_grid 将图片拼接成一张大图, nrow 参数设置每行的图片数量
        grid_img = torchvision.utils.make_grid(img, nrow=8)
        # 将张量转换为 numpy 数组，并调整通道的顺序
        grid_img = grid_img.cpu().numpy().transpose(1, 2, 0)
        # 将 numpy 数组的值范围从 [-1, 1] 调整为 [0, 255]
        grid_img = (grid_img + 1) / 2 * 255
        # 将 numpy 数组转换为 PIL 图片
        grid_img_pil = Image.fromarray(grid_img.astype('uint8'))
        # 保存图片
        grid_img_pil.save(path + '/batch'+str(iter_index+1)+'_images.png')

        ##2. part mask
        image_part_mask = image_part_mask.unsqueeze(dim=1)
        grid_mask = torchvision.utils.make_grid(image_part_mask, nrow=8)
        grid_mask = grid_mask.cpu().numpy()[0]

        # 将 numpy 数组转换为 PIL 图片
        grid_mask = Image.fromarray(grid_mask.astype('uint8'))
        palette = get_palette(256)
        grid_mask.putpalette(palette)
        grid_mask.save(path + '/batch'+str(iter_index+1)+'_masks.png')

        ##3. emaskd image
        masked_grid_img = torchvision.utils.make_grid(masked_imgs, nrow=8)
        # 将张量转换为 numpy 数组，并调整通道的顺序
        masked_grid_img = masked_grid_img.cpu().numpy().transpose(1, 2, 0)
        # 将 numpy 数组的值范围从 [-1, 1] 调整为 [0, 255]
        masked_grid_img = (masked_grid_img + 1) / 2 * 255
        # 将 numpy 数组转换为 PIL 图片
        masked_grid_img_pil = Image.fromarray(masked_grid_img.astype('uint8'))
        # 保存图片
        masked_grid_img_pil.save(path + '/batch'+str(iter_index+1)+'_eimages.png')

        # 4. emaskd image1
        eimage_part_mask = eimage_part_mask.unsqueeze(dim=1)
        e_grid_mask = torchvision.utils.make_grid(eimage_part_mask, nrow=8)
        e_grid_mask = e_grid_mask.cpu().numpy()[0]

        # 将 numpy 数组转换为 PIL 图片
        e_grid_mask = Image.fromarray(e_grid_mask.astype('uint8'))
        e_palette = get_palette(256)
        e_grid_mask.putpalette(e_palette)
        e_grid_mask.save(path + '/batch'+str(iter_index+1)+'_eimages_masks.png')



def get_palette(n):
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

# import argparse
# from config import config
# from config import update_config

# def parse_args():
#     parser = argparse.ArgumentParser(description='Test segmentation network')
    
#     parser.add_argument('--cfg',
#                         help='experiment configure file name',
#                         default='/home/ssy/occluded/HumanParsing/seg_hrnet_w48_occduke.yaml',
#                         type=str)
#     parser.add_argument('opts',
#                         help="Modify config options using the command-line",
#                         default=None,
#                         nargs=argparse.REMAINDER)

#     args = parser.parse_args()
#     update_config(config, args)

    def get_simgle_part_masks(self, image, model, size):
        model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0)
            _size = image.size()
            image = image.cuda()
            preds = model(image)
            preds = F.interpolate(preds, size=(_size[-2], _size[-1]), mode='bilinear', align_corners=False)
            preds = preds.exp()
            
            if preds.size()[-2] != size[0] or preds.size()[-1] != size[1]:
                preds = F.interpolate(preds, size=(size[-2], size[-1]), mode='bilinear', align_corners=False)
            
            # test_dataset.save_pred(pred, sv_path, name)
            preds = preds.cpu().numpy().copy()  # (1, 10, 285, 113)
            preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)    # (1, 285, 113), 找出预测结果中每个位置概率最大的类别，并将结果转换为 uint8 类型的numpy数组
            for i in range(preds.shape[0]): # 遍历batch中每一张图片
                pred = preds[i]
                pred = self.largest_connect_component(pred) # 保留最大的连通区域
                pred = self.combine_label(pred)             # 合并某些类别的标签
                pred = self.remove_small_area(pred)         # 移除面积小于一定阈值的区域。
                pred = self.remove_duplicate_area(pred)     # 移除重复区域

            # save
            palette = self.get_palette(256)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save('./test_results12.png')

            return pred


    def test(self, testloader, model, sv_dir='', sv_pred=True):
        model.eval()
        with torch.no_grad():
            for batch in tqdm(testloader):
            # for batch in testloader:
                image, _, size, name = batch
                # image: torch.Size([1, 3, 384, 128])
                size = size[0]  # tensor([285, 113])
                _size = image.size()
                image= image.cuda()
                preds = model(image)
                preds = F.interpolate(preds, size=(_size[-2], _size[-1]), mode='bilinear', align_corners=False)
                preds = preds.exp()
                
                
                if preds.size()[-2] != size[0] or preds.size()[-1] != size[1]:
                    preds = F.interpolate(preds, size=(size[-2], size[-1]), mode='bilinear', align_corners=False)
                    
                if sv_pred:
                    sv_path = os.path.join(sv_dir,'test_results')
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)

                    palette = self.get_palette(256)
                    preds = preds.cpu().numpy().copy()  # (1, 10, 285, 113)
                    preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)    # (1, 285, 113), 找出预测结果中每个位置概率最大的类别，并将结果转换为 uint8 类型的numpy数组
                    for i in range(preds.shape[0]): # 遍历batch中每一张图片
                        pred = preds[i]
                        pred = self.largest_connect_component(pred) # 保留最大的连通区域
                        pred = self.combine_label(pred)             # 合并某些类别的标签
                        pred = self.remove_small_area(pred)         # 移除面积小于一定阈值的区域。
                        pred = self.remove_duplicate_area(pred)     # 移除重复区域

                        # end
                        save_img = Image.fromarray(pred)
                        save_img.putpalette(palette)
                        dataset_name, file_name = name[i].split('/')
                        path = os.path.join(sv_path, dataset_name)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        save_img.save(os.path.join(path, file_name.replace('.jpg', '.png')))
            
