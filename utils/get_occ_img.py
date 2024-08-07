import torch
from torchvision import transforms
import random
import cv2

def find_diff_id_images(img):
    # 确定每组有4张图像
    batch_size = img.shape[0]
    group_size = 4
    num_groups = batch_size // group_size
    
    # 生成图像索引
    indices = torch.arange(batch_size)
    # 生成组ID矩阵
    group_ids = torch.repeat_interleave(torch.arange(num_groups), group_size)
    # 扩展维度以便比较
    group_ids_expanded = group_ids.unsqueeze(1)
    # 生成不同组的掩码矩阵
    diff_group_mask = group_ids_expanded != group_ids.unsqueeze(0)
    # 创建随机索引
    random_indices = torch.empty(batch_size, dtype=torch.long)
    
    for i in range(batch_size):
        # 获取当前图像的不同组索引
        diff_indices = indices[diff_group_mask[i]]
        # 随机选择一个不同组的图像索引
        random_indices[i] = diff_indices[torch.randint(len(diff_indices), (1,))]
    
    # 获取对应的不同ID的图像
    img_other = img[random_indices]
    return random_indices, img_other


def calculate_f_size(img_shape, patch_size, stride_size):
    """计算patch化后的高和宽"""
    height, width = img_shape[:2]
    fH = (height - patch_size) // stride_size[0] + 1
    fW = (width - patch_size) // stride_size[1] + 1
    return fH, fW


def patchify(img, patch_size, stride_size):
    """将图像划分成patch"""
    batch_size, height, width, channels = img.shape
    fH, fW = calculate_f_size((height, width), patch_size, stride_size)
    patches = img.unfold(1, patch_size, stride_size[0]).unfold(2, patch_size, stride_size[1])   # torch.Size([64, 16, 8, 3, 16, 16])
    patches = patches.contiguous().view(batch_size, fH * fW, -1)
    return patches


def unpatchify(patches, img_shape, patch_size, stride_size):
    """将patch重新组合成图像"""
    batch_size, num_patches, patch_dim = patches.shape
    height, width, channels = img_shape
    fH, fW = calculate_f_size((height, width), patch_size, stride_size)
    patches = patches.view(batch_size, fH, fW, channels, patch_size, patch_size)        # torch.Size([64, 16, 8, 3, 16, 16])
    patches = patches.permute(0, 1, 4, 2, 5, 3).contiguous()                            # torch.Size([64, 16, 16, 8, 16, 3])
    img_ = torch.zeros((batch_size, height, width, channels), device=patches.device)    # torch.Size([64, 256, 128, 3])
    
    for i in range(fH):
        for j in range(fW):
            img_[:, i*stride_size[0]:i*stride_size[0]+patch_size, j*stride_size[1]:j*stride_size[1]+patch_size,: ] = patches[:, i, :, j, :, :]

    img_ = img_.contiguous().permute(0, 3, 1, 2)   # (B, H, W, C) -> (B, C, H, W)
    return img_


def cover_patches(img, other_img, img_other_idx, stride_size, patch_size=16):
    """将other_img的有用patch随机覆盖到img的左侧、右侧或下侧"""

    # 将图像划分成patch
    fH, fW = calculate_f_size(img.shape[2:], patch_size, stride_size)
    img = img.permute(0, 2, 3, 1)   # (B, C, H, W) -> (B, H, W, C)
    other_img = other_img.permute(0, 2, 3, 1)   # (B, C, H, W) -> (B, H, W, C)
    img_patches = patchify(img, patch_size, stride_size)
    other_patches = patchify(other_img, patch_size, stride_size)
    
    # 获取有用patch的索引
    batch_size, useful_patch_num = img_other_idx.shape
    # 生成随机覆盖位置
    locations = torch.randint(0, 3, (batch_size,), device=img.device)  # 0: left, 1: right, 2: bottom
    # 初始化返回序列
    covered_idx = torch.zeros_like(img_other_idx)
    
    # 生成目标位置索引
    target_indices = torch.zeros_like(img_other_idx)
    for b in range(batch_size):
        useful_patches = img_other_idx[b]
        useful_patches_h = useful_patches // fW
        useful_patches_w = useful_patches % fW

        target_h = useful_patches_h
        target_w = useful_patches_w

        if locations[b] == 0:       # 覆盖左侧
            target_w -= (fW // 2)
        elif locations[b] == 1:     # 覆盖右侧
            target_w += (fW // 2)
        elif locations[b] == 2:     # 覆盖下侧
            target_h += (fH // 2)
        # else:
        #     target_h -= (fH // 2)
    
        # 更新img_patches
        valid_indices = (target_h >= 0) & (target_h < fH) & (target_w >= 0) & (target_w < fW)
        target_indices[b, valid_indices] = target_h[valid_indices] * fW + target_w[valid_indices]
        source_indices = useful_patches[valid_indices]
        img_patches[b, target_indices[b, valid_indices]] = other_patches[b, source_indices]
        covered_idx[b, valid_indices] = target_indices[b, valid_indices]
    
    # 将patch重新组合成图像
    img_occ_person = unpatchify(img_patches, img.shape[1:], patch_size, stride_size)
    # img_occ_person = img_occ_person.permute(0, 3, 1, 2)   # (B, H, W, C) -> (B, C, H, W)
    return img_occ_person, covered_idx

def cover_pixels(img_, other_img, other_img_pmask, other_img_pmask_patch, stride_size, beta=0.5, patch_size=16, img_other_idx=None):
    """
    将other_img的human随机覆盖到img的左侧、右侧或下侧; 弱关注patch覆盖到上下左右侧。
    args:   img, other_img:(B, C, H, W)
            othet_img_pmask:(B, H, W)
            othet_img_pmask_idx:(B, fH, fW)
    """
    assert 0 <= beta <= 1, "beta must be between 0 and 1"

    fH, fW = calculate_f_size(img_.shape[2:], patch_size, stride_size)
    img = img_.clone()
    other_img_pmask = other_img_pmask > 0               # (B, H, W)
    other_img_pmask_patch = other_img_pmask_patch > 0   # (B, fH, fW)
    
    # batch_size, height, width, channel = img.shape
    batch_size, channel, height, width = img.shape
    n1 = (fH// 2) * fW
    n2 = fH * (fW // 2)
    useful_patch_num = max(n1, n2) + fW
    PATCH_NUM = fH * fW

    # 生成选择覆盖方式的随机数
    random_number = torch.rand(batch_size)
    # 生成随机覆盖位置
    person_locations = torch.randint(0, 3, (batch_size,))  # 0: left, 1: right, 2: bottom
    occlussion_locations = torch.randint(0, 4, (batch_size,))  # 0: left, 1: right, 2: bottom
    # 生成目标位置索引
    target_patches_indices = torch.zeros(batch_size, useful_patch_num, device=img.device, dtype=torch.int64)

    # 创建全集索引数组
    full_idx = torch.arange(PATCH_NUM).unsqueeze(0).repeat(batch_size, 1)
    # 生成整个特征图的索引
    indices = torch.arange(fH * fW, device=img.device).view(fH, fW)
    left_indices = indices[:, :fW // 2].flatten()   # 110
    right_indices = indices[:, fW // 2:].flatten()  # 132
    top_indices = indices[:fH // 2, :].flatten()    # 121
    bottom_indices = indices[fH // 2:, :].flatten() # 121
    to_pil_transform = transforms.ToPILImage()
    to_tensor_transform = transforms.ToTensor()
    resize_transformH = transforms.Resize((height//2, width))
    resize_transformW = transforms.Resize((height, width//2))

    for b in range(batch_size):
        if random_number[b] <= beta:
            # non-target person
            true_patches_indices = torch.where(other_img_pmask_patch[b])
            useful_patches_h = true_patches_indices[0]
            useful_patches_w = true_patches_indices[1]

            true_indices = torch.where(other_img_pmask[b])
            useful_h = true_indices[0]
            useful_w = true_indices[1]

            target_patches_h = useful_patches_h
            target_patches_w = useful_patches_w
            target_h = useful_h.clone()
            target_w = useful_w.clone()
            source_h = useful_h.clone()
            source_w = useful_w.clone()
            
            if person_locations[b] == 0:    # 覆盖左侧
                target_w -= (width // 2)
                target_patches_w -= (fW // 2)
            elif person_locations[b] == 1:  # 覆盖右侧
                target_w += (width // 2)
                target_patches_w += (fW // 2)
            elif person_locations[b] == 2:  # 覆盖下侧
                target_h += (height // 2)
                target_patches_h += (fH // 2)

            # 更新img
            valid_indices = (target_h >= 0) & (target_h < height) & (target_w >= 0) & (target_w < width)
            img[b, :, target_h[valid_indices], target_w[valid_indices]] = other_img[b, :, source_h[valid_indices], source_w[valid_indices]]

            # 生成目标idx
            valid_patches_indices = (target_patches_h >= 0) & (target_patches_h < fH) & (target_patches_w >= 0) & (target_patches_w < fW)
            target_patches = target_patches_h[valid_patches_indices] * fW + target_patches_w[valid_patches_indices]
            len_target_patches = min(len(target_patches), target_patches_indices.shape[1])
            target_patches_indices[b, :len_target_patches] = target_patches[:len_target_patches]
        else:
            # occlussion
            if img_other_idx is not None:
            ###### use ineffective patches ######
                mask = torch.ones(PATCH_NUM, dtype=torch.bool)
                mask[img_other_idx[b]] = False  # 标记出现过的索引为False
                useless_idx = full_idx[b][mask]  # 获取未出现的索引
            else:
                ###### use random all patches ######
                useless_idx = full_idx[b]

            random_index = torch.randint(0, len(useless_idx), (1,)).item()  # 生成一个随机索引
            useless_idx = useless_idx[random_index]  # 使用随机索引从useless_idx中选择一个元素
            useless_h = useless_idx // fW
            useless_w = useless_idx % fW
            # useless_patch_img = other_img[b, :, useless_h * stride_size[0]:(useless_h + 1) * stride_size[0], useless_w * stride_size[1]:(useless_w + 1) * stride_size[1]].clone()
            useless_patch_img = other_img[b, :, useless_h * stride_size[0]:useless_h * stride_size[0]+16, useless_w * stride_size[1]:useless_w * stride_size[1]+16].clone()
            # useless_patch_img = to_pil_transform(useless_patch_img)  # Tensor到PIL图像
            

            if occlussion_locations[b] == 0:    # 覆盖左侧
                useless_patch_img = resize_transformW(useless_patch_img)
                # useless_patch_img = to_tensor_transform(useless_patch_img)  # PIL图像到Tensor
                img[b, :, 0:height, 0:width//2] = useless_patch_img
                target_patches_indices[b, :len(left_indices)] = left_indices
                
            elif occlussion_locations[b] == 1:  # 覆盖右侧
                useless_patch_img = resize_transformW(useless_patch_img)
                #useless_patch_img = to_tensor_transform(useless_patch_img)
                img[b, :, 0:height, width//2:width] = useless_patch_img
                target_patches_indices[b, :len(right_indices)] = right_indices
                
            elif occlussion_locations[b] == 2:  # 覆盖下侧
                useless_patch_img = resize_transformH(useless_patch_img)
                # useless_patch_img = to_tensor_transform(useless_patch_img)
                img[b, :, height//2:height, 0:width] = useless_patch_img
                target_patches_indices[b, :len(bottom_indices)] = bottom_indices
            else:
                useless_patch_img = resize_transformH(useless_patch_img)
                # useless_patch_img = to_tensor_transform(useless_patch_img)
                img[b, :, 0:height//2, 0:width] = useless_patch_img
                target_patches_indices[b, :len(top_indices)] = top_indices
                
    return img, target_patches_indices


def cover_pixels1(img_, other_img, other_img_pmask, other_img_pmask_patch, stride_size, beta=0.5, patch_size=16):
    """
    将other_img的有用patch随机覆盖到img的左侧、右侧或下侧
    args:   img, other_img:(B, C, H, W)
            othet_img_pmask:(B, H, W)
            othet_img_pmask_idx:(B, fH, fW)
    """
    assert 0 <= beta <= 1, "beta must be between 0 and 1"

    fH, fW = calculate_f_size(img_.shape[2:], patch_size, stride_size)
    img = img_.clone()
    other_img_pmask = other_img_pmask > 0               # (B, H, W)
    other_img_pmask_patch = other_img_pmask_patch > 0   # (B, fH, fW)
    
    # batch_size, height, width, channel = img.shape
    batch_size, channel, height, width = img.shape
    n1 = (fH// 2) * fW
    n2 = fH * (fW // 2)
    useful_patch_num = max(n1, n2) + fW
    PATCH_NUM = fH * fW

    # 生成选择覆盖方式的随机数
    random_number = torch.rand(batch_size)
    # 生成随机覆盖位置
    person_locations = torch.randint(0, 3, (batch_size,))  # 0: left, 1: right, 2: bottom
    occlussion_locations = torch.randint(0, 4, (batch_size,))  # 0: left, 1: right, 2: bottom
    # 生成目标位置索引
    target_patches_indices = torch.zeros(batch_size, useful_patch_num, device=img.device, dtype=torch.int64)

    # 创建全集索引数组
    full_idx = torch.arange(PATCH_NUM).unsqueeze(0).repeat(batch_size, 1)
    # 生成整个特征图的索引
    indices = torch.arange(fH * fW, device=img.device).view(fH, fW)
    left_indices = indices[:, :fW // 2].flatten()   # 110
    right_indices = indices[:, fW // 2:].flatten()  # 132
    top_indices = indices[:fH // 2, :].flatten()    # 121
    bottom_indices = indices[fH // 2:, :].flatten() # 121
    # to_pil_transform = transforms.ToPILImage()
    # to_tensor_transform = transforms.ToTensor()
    resize_transformH = transforms.Resize((height//2, width))
    resize_transformW = transforms.Resize((height, width//2))

    for b in range(batch_size):
        if random_number[b] <= beta:
            # non-target person
            true_patches_indices = torch.where(other_img_pmask_patch[b])
            useful_patches_h = true_patches_indices[0]
            useful_patches_w = true_patches_indices[1]

            true_indices = torch.where(other_img_pmask[b])
            useful_h = true_indices[0]
            useful_w = true_indices[1]

            target_patches_h = useful_patches_h
            target_patches_w = useful_patches_w
            target_h = useful_h.clone()
            target_w = useful_w.clone()
            source_h = useful_h.clone()
            source_w = useful_w.clone()
            
            if person_locations[b] == 0:    # 覆盖左侧
                target_w -= (width // 2)
                target_patches_w -= (fW // 2)
            elif person_locations[b] == 1:  # 覆盖右侧
                target_w += (width // 2)
                target_patches_w += (fW // 2)
            elif person_locations[b] == 2:  # 覆盖下侧
                target_h += (height // 2)
                target_patches_h += (fH // 2)

            # 更新img
            valid_indices = (target_h >= 0) & (target_h < height) & (target_w >= 0) & (target_w < width)
            img[b, :, target_h[valid_indices], target_w[valid_indices]] = other_img[b, :, source_h[valid_indices], source_w[valid_indices]]

            # 生成目标idx
            valid_patches_indices = (target_patches_h >= 0) & (target_patches_h < fH) & (target_patches_w >= 0) & (target_patches_w < fW)
            target_patches = target_patches_h[valid_patches_indices] * fW + target_patches_w[valid_patches_indices]
            len_target_patches = min(len(target_patches), target_patches_indices.shape[1])
            target_patches_indices[b, :len_target_patches] = target_patches[:len_target_patches]
        else:
            # occlussion
            useless_patch_img = random_crop_square(other_img[b])
            useless_patch_img = to_pil_transform(useless_patch_img)  # Tensor到PIL图像

            if occlussion_locations[b] == 0:    # 覆盖左侧
                useless_patch_img = resize_transformW(useless_patch_img)
                useless_patch_img = to_tensor_transform(useless_patch_img)  # PIL图像到Tensor
                img[b, :, 0:height, 0:width//2] = useless_patch_img
                target_patches_indices[b, :len(left_indices)] = left_indices
                
            elif occlussion_locations[b] == 1:  # 覆盖右侧
                useless_patch_img = resize_transformW(useless_patch_img)
                useless_patch_img = to_tensor_transform(useless_patch_img)
                img[b, :, 0:height, width//2:width] = useless_patch_img
                target_patches_indices[b, :len(right_indices)] = right_indices
                
            elif occlussion_locations[b] == 2:  # 覆盖下侧
                useless_patch_img = resize_transformH(useless_patch_img)
                useless_patch_img = to_tensor_transform(useless_patch_img)
                img[b, :, height//2:height, 0:width] = useless_patch_img
                target_patches_indices[b, :len(bottom_indices)] = bottom_indices
            else:
                useless_patch_img = resize_transformH(useless_patch_img)
                useless_patch_img = to_tensor_transform(useless_patch_img)
                img[b, :, 0:height//2, 0:width] = useless_patch_img
                target_patches_indices[b, :len(top_indices)] = top_indices
                
    return img, target_patches_indices

def random_crop_square(img):
    C, H, W = img.shape
    # 确定可能的最大正方形边长
    max_square_side = min(H, W)
    
    # 随机选择正方形的边长，设最小边长为16
    square_side = random.randint(16, max_square_side)
    
    # 随机确定正方形左上角的位置
    top = random.randint(0, H - square_side)
    left = random.randint(0, W - square_side)
    
    # 裁剪出正方形图像
    img_crop = img[:, top:top+square_side, left:left+square_side].clone()
    
    return img_crop
