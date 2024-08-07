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
        # img[img == 5] = 0   # for 4 part test
        img[img == 5] = 2   # for 4 part test
        img[img == 6] = 1   
        img[img == 7] = 1
        # img[img == 8] = 0
        img[img == 8] = 2
        img[img == 9] = 0
        # img[img == 9] = 3   # for occluded-REID
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

            for i in range(preds.shape[0]): # 遍历batch中每一张图片
                preds[i] = self.largest_connect_component(preds[i]) # 保留最大的连通区域
                preds[i] = self.combine_label(preds[i])             # 合并某些类别的标签
                # pred = self.remove_small_area(preds[i])         # 移除面积小于一定阈值的区域。
                # pred = self.remove_duplicate_area(preds[i])     # 移除重复区域
                preds[i] = self.remove_small_area(preds[i])         # 移除面积小于一定阈值的区域。
                preds[i] = self.remove_duplicate_area(preds[i])     # 移除重复区域
            
            # end_time = time.time()

            # print('pred_time:', pred_time - start_time)
            # print('interpolate_time:', interpolate_time - pred_time)
            # print('numpy_time:', numpy_time - interpolate_time)
            # print('end_time:', end_time - numpy_time)
            # print('__________________________________________________')

            # if sv_pred:
            #     palette = self.get_palette(256)
            #     save_img = Image.fromarray(pred)
            #     save_img.putpalette(palette)
            #     save_img.save('./test_results12.png')

            return preds
    
    # def custom_maxtimes_downsample(self, image, K, block_size=(16, 16)):
    #     # 获取图像的高度和宽度
    #     batch, height, width = image.shape

    #     # 计算下采样后的高度和宽度
    #     down_height = height // block_size[0]
    #     down_width = width // block_size[1]

    #     if down_height * block_size[0] != height or down_width * block_size[1] != width:
    #         # 裁剪图像以便它可以被均匀地分割成块
    #         crop_height = down_height * block_size[0]
    #         crop_width = down_width * block_size[1]
    #         image = image[:, :crop_height, :crop_width]

    #     # 重新调整 input_array 的形状以便于计算每个 block 的 mode
    #     reshaped = image.reshape(batch, down_height, block_size[0], down_width, block_size[1])

    #     # 初始化一个新的数组来存储每个 block 的 histogram
    #     hist = torch.zeros((batch, down_height, down_width, K+1)).to(image.device)

    #     # 计算每个 block 的 histogram
    #     for k in range(0, K+1):
    #         hist[..., k] = torch.sum(reshaped == k, axis=(2, 4))

    #     # 找到每个 block 中出现次数最多的元素
    #     block_mode = torch.argmax(hist, axis=-1)

    #     return block_mode
    
    def custom_maxfrequent_downsample(self, image, num_parts, stride):
        batch_size, height, width = image.size()
        kernel_size = 16

        # 计算经过卷积操作后特征图的大小
        output_height = (height - kernel_size) // stride + 1
        output_width = (width - kernel_size) // stride + 1

        # unfold操作
        patches = image.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
        patches = patches.contiguous().view(batch_size, output_height, output_width, -1)  # [B, OH, OW, K*K]

        # 创建存储每个part的计数
        counts = torch.zeros(batch_size, output_height, output_width, num_parts + 1, device=image.device)

        for part in range(num_parts + 1):
            counts[..., part] = (patches == part).sum(dim=-1)

        # 获取每个区域出现次数最多的部位数字
        output_tensor = counts.argmax(dim=-1)

        return output_tensor
    
    def get_occ_mask(self, h_image_part_mask, o_mask, image_shape, stride_size):
        kernel_size = 16
        (H, W) = image_shape
        b, fh, fw = h_image_part_mask.shape

        # 初始化occ_mask为原始的h_image_part_mask
        occ_mask = h_image_part_mask.clone()

        for i in range(b):
            mask = o_mask[i]
            if mask is None:
                continue
            
            top, left, mask_h, mask_w = mask
            # 生成一个全零矩阵，用于记录每个位置是否在mask区域
            mask_matrix = torch.zeros((H, W), dtype=torch.uint8, device=h_image_part_mask.device)
            mask_matrix[top:top + mask_h, left:left + mask_w] = 1
            
            # 用unfold将原图的mask展开为卷积块
            unfolded_mask = mask_matrix.unfold(0, kernel_size, stride_size).unfold(1, kernel_size, stride_size)
            
            # 计算每个卷积块中mask的占比
            mask_counts = unfolded_mask.sum(dim=(-1, -2))
            mask_ratios = mask_counts / (kernel_size * kernel_size)

            # 确保mask_ratios和occ_mask[i]的尺寸匹配
            mask_ratios = mask_ratios[:fh, :fw]
            
            # 将占比大于70%的位置设为0
            occ_mask[i] = torch.where(mask_ratios > 0.7, torch.tensor(0, dtype=occ_mask.dtype, device=occ_mask.device), occ_mask[i])
        return occ_mask

    
#     def visualization_batch(self, iter_index, img, image_part_mask, masked_imgs, eimage_part_mask, emask=None, path='./visualization0.8'):  # visualization for test
    def visualization_batch(self, iter_index, img, masked_imgs,  emask=None, path='./visualization0.8'):  # visualization for test
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

        # ##2. part mask
        # image_part_mask = image_part_mask.unsqueeze(dim=1)
        # grid_mask = torchvision.utils.make_grid(image_part_mask, nrow=8)
        # grid_mask = grid_mask.cpu().numpy()[0]

        # # 将 numpy 数组转换为 PIL 图片
        # grid_mask = Image.fromarray(grid_mask.astype('uint8'))
        # palette = get_palette(256)
        # grid_mask.putpalette(palette)
        # grid_mask.save(path + '/batch'+str(iter_index+1)+'_masks.png')

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

        # # 4. emaskd image1
        # eimage_part_mask = eimage_part_mask.unsqueeze(dim=1)
        # e_grid_mask = torchvision.utils.make_grid(eimage_part_mask, nrow=8)
        # e_grid_mask = e_grid_mask.cpu().numpy()[0]

        # # 将 numpy 数组转换为 PIL 图片
        # e_grid_mask = Image.fromarray(e_grid_mask.astype('uint8'))
        # e_palette = get_palette(256)
        # e_grid_mask.putpalette(e_palette)
        # e_grid_mask.save(path + '/batch'+str(iter_index+1)+'_eimages_masks.png')

        # 5. emask
        if emask is not None:
            emask = emask.unsqueeze(dim=1)
            e_grid = torchvision.utils.make_grid(emask, nrow=8)
            e_grid = e_grid.cpu().numpy()[0]

            # 将 numpy 数组转换为 PIL 图片
            e_grid = Image.fromarray(e_grid.astype('uint8'))
            emask_palette = get_palette(256)
            e_grid.putpalette(emask_palette)
            e_grid.save(path + '/batch'+str(iter_index+1)+'_trainmasks.png')



def visualization_batch_paper(iter_index, img, image_part_mask, masked_imgs, emask=None, ba_mask=None, hp_ori=None, path='./visualization0.8', is_occ=False):  # visualization for test
#         def visualization_batch(self, iter_index, img, masked_imgs,  emask=None, path='./visualization0.8'):  # visualization for test
        # if img.shape[0] < 64:
        #     print('batch' + str(iter_index+1) +'\'s size is small than 64, skip visualization.')
        #     return

        # 检查路径是否存在
        if not os.path.exists(path):
            # 如果路径不存在，则创建
            os.makedirs(path)

        ##1. original image
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
        if is_occ:
            grid_img_pil.save(path + '/batch'+str(iter_index+1)+'o_images.png')
        else:
            grid_img_pil.save(path + '/batch'+str(iter_index+1)+'h_images.png')


        ##2. original human parsing mask
        image_part_mask = image_part_mask.unsqueeze(dim=1)
        grid_mask = torchvision.utils.make_grid(image_part_mask, nrow=8)
        grid_mask = grid_mask.cpu().numpy()[0]
        # 将 numpy 数组转换为 PIL 图片
        grid_mask = Image.fromarray(grid_mask.astype('uint8'))
        palette = get_palette(256)
        grid_mask.putpalette(palette)
        if is_occ:
            grid_mask.save(path + '/batch'+str(iter_index+1)+'o_hpmasks.png')
        else:
            grid_mask.save(path + '/batch'+str(iter_index+1)+'h_hpmasks.png')

        ##2.1 original human parsing mask 256*128
        if hp_ori is not None:
            hp_ori = hp_ori.unsqueeze(dim=1)
            grid_mask256 = torchvision.utils.make_grid(hp_ori, nrow=8)
            grid_mask256 = grid_mask256.cpu().numpy()[0]
            # 将 numpy 数组转换为 PIL 图片
            grid_mask256 = Image.fromarray(grid_mask256.astype('uint8'))
            palette = get_palette(256)
            grid_mask256.putpalette(palette)
            if is_occ:
                grid_mask256.save(path + '/batch'+str(iter_index+1)+'o_hpmasks256.png')
            else:
                grid_mask256.save(path + '/batch'+str(iter_index+1)+'h_hpmasks256.png')
        


        ##3. effective patch image
        masked_grid_img = torchvision.utils.make_grid(masked_imgs, nrow=8)
        # 将张量转换为 numpy 数组，并调整通道的顺序
        masked_grid_img = masked_grid_img.cpu().numpy().transpose(1, 2, 0)
        # 将 numpy 数组的值范围从 [-1, 1] 调整为 [0, 255]
        masked_grid_img = (masked_grid_img + 1) / 2 * 255
        # 将 numpy 数组转换为 PIL 图片
        masked_grid_img_pil = Image.fromarray(masked_grid_img.astype('uint8'))
        # 保存图片
        if is_occ:
            masked_grid_img_pil.save(path + '/batch'+str(iter_index+1)+'o_eimages.png')
        else:
            masked_grid_img_pil.save(path + '/batch'+str(iter_index+1)+'h_eimages.png')


        # 4. final mask
        if emask is not None:
            emask = emask.unsqueeze(dim=1)
            e_grid = torchvision.utils.make_grid(emask, nrow=8)
            e_grid = e_grid.cpu().numpy()[0]

            # 将 numpy 数组转换为 PIL 图片
            e_grid = Image.fromarray(e_grid.astype('uint8'))
            emask_palette = get_palette(256)
            e_grid.putpalette(emask_palette)
            if is_occ:
                e_grid.save(path + '/batch'+str(iter_index+1)+'o_finalhpmasks.png')
            else:
                e_grid.save(path + '/batch'+str(iter_index+1)+'h_finalhpmasks.png')

        # 5. BA mask
        if ba_mask is not None:
            # ba_mask = ba_mask.unsqueeze(dim=1)
            # ba_grid = torchvision.utils.make_grid(ba_mask, nrow=8)
            # ba_mask = ba_mask.cpu().numpy()[0]

            # # 将 numpy 数组转换为 PIL 图片
            # ba_grid = Image.fromarray(ba_grid.astype('uint8'))
            # bamask_palette = get_palette(256)
            # ba_grid.putpalette(bamask_palette)
            # ba_grid.save(path + '/batch'+str(iter_index+1)+'bamasks.png')
            ba_mask = ba_mask.unsqueeze(dim=1)
            ba_grid = torchvision.utils.make_grid(ba_mask, nrow=8)
            ba_grid = ba_grid.cpu().numpy()[0]

            # 将 numpy 数组转换为 PIL 图片
            ba_grid = Image.fromarray(ba_grid.astype('uint8') * 255)  # 乘以255将1转换为255（白色），0保持为0（黑色）
            ba_grid = ba_grid.convert('L')  # 转换为灰度图像
            ba_grid.save(path + '/batch'+str(iter_index+1)+'o_bamasks.png')


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
            
