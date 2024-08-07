from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path.split('/')[-1]

class ImageDatasetWithOriAndMask(ImageDataset):
    def __init__(self, dataset, transform=None, transform_ori=None, transform_2=None):
        self.dataset = dataset
        self.transform = transform
        self.transform_ori = transform_ori
        self.transform_2 = transform_2
        # # 生成mask位置
        # if mask_radio is not None and f_size is not None:
        #     self.mask = generate_random_masks(f_size, len(dataset), mask_radio)
        # else:
        #     self.mask = None

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        # 保存原始图像
        img = self.transform_ori(img)

        if self.transform is not None:
            img_eras, mask = self.transform(img.clone())

        if self.transform_2 is not None:
            img_trans2, mask2 = self.transform_2(img)

        # return img_ori, img_eras, mask, img_trans2, mask2, pid, camid, trackid, img_path.split('/')[-1]
        return img, img_eras, mask, pid, camid, trackid, img_path.split('/')[-1]
        # return img, pid, camid, trackid, img_path.split('/')[-1]

def generate_random_masks(f_size, L, radio):
    H, W = f_size
    area = int(radio * H * W)
    
    # 生成随机起始点
    start_points = np.random.randint(0, [H, W], size=(L, 2))
    
    # 生成随机长宽比，但确保面积等于radio * H * W
    ratios = np.random.rand(L)
    heights = np.sqrt(area / ratios).astype(int)
    widths = (area / heights).astype(int)
    
    # 限制长宽在图像尺寸范围内
    heights = np.clip(heights, 1, H)
    widths = np.clip(widths, 1, W)
    
    # 创建掩码矩阵
    masks = np.zeros((L, area), dtype=int)
    
    # 获取每个mask的起始和结束位置的索引
    start_h = start_points[:, 0]
    start_w = start_points[:, 1]
    end_h = np.minimum(start_h + heights, H)
    end_w = np.minimum(start_w + widths, W)
    
    # 生成每个mask的索引
    for i in range(L):
        mask_indices = np.array([(x * W + y) for x in range(start_h[i], end_h[i]) for y in range(start_w[i], end_w[i])])
        masks[i, :len(mask_indices)] = mask_indices[:area]
    
    return masks