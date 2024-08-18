import torch
from torchvision import transforms
import random

def find_diff_id_images(img):
    batch_size = img.shape[0]
    group_size = 4
    num_groups = batch_size // group_size
    
    indices = torch.arange(batch_size)
    group_ids = torch.repeat_interleave(torch.arange(num_groups), group_size)
    group_ids_expanded = group_ids.unsqueeze(1)
    diff_group_mask = group_ids_expanded != group_ids.unsqueeze(0)
    random_indices = torch.empty(batch_size, dtype=torch.long)
    
    for i in range(batch_size):
        diff_indices = indices[diff_group_mask[i]]
        random_indices[i] = diff_indices[torch.randint(len(diff_indices), (1,))]
    
    img_other = img[random_indices]
    return random_indices, img_other


def calculate_f_size(img_shape, patch_size, stride_size):
    """calculate height and width after patchlizing"""
    height, width = img_shape[:2]
    fH = (height - patch_size) // stride_size[0] + 1
    fW = (width - patch_size) // stride_size[1] + 1
    return fH, fW


def cover_pixels(img_, other_img, other_img_pmask, other_img_pmask_patch, stride_size, beta=0.5, patch_size=16):
    """
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

    # generate random number for choosing human or region occlussion
    random_number = torch.rand(batch_size)
    # generate person occlusion location and region occlussion location
    person_locations = torch.randint(0, 3, (batch_size,))  # 0: left, 1: right, 2: bottom
    occlussion_locations = torch.randint(0, 4, (batch_size,))  # 0: left, 1: right, 2: bottom
    # generate target patches indices
    target_patches_indices = torch.zeros(batch_size, useful_patch_num, device=img.device, dtype=torch.int64)

    # generate full index
    full_idx = torch.arange(PATCH_NUM).unsqueeze(0).repeat(batch_size, 1)
    # get indices
    indices = torch.arange(fH * fW, device=img.device).view(fH, fW)
    left_indices = indices[:, :fW // 2].flatten()   
    right_indices = indices[:, fW // 2:].flatten()  
    top_indices = indices[:fH // 2, :].flatten()    
    bottom_indices = indices[fH // 2:, :].flatten() 
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
            
            if person_locations[b] == 0:    # cover left
                target_w -= (width // 2)
                target_patches_w -= (fW // 2)
            elif person_locations[b] == 1:  # cover right
                target_w += (width // 2)
                target_patches_w += (fW // 2)
            elif person_locations[b] == 2:  # cover bottom
                target_h += (height // 2)
                target_patches_h += (fH // 2)

            # update img
            valid_indices = (target_h >= 0) & (target_h < height) & (target_w >= 0) & (target_w < width)
            img[b, :, target_h[valid_indices], target_w[valid_indices]] = other_img[b, :, source_h[valid_indices], source_w[valid_indices]]

            # generatre target idx
            valid_patches_indices = (target_patches_h >= 0) & (target_patches_h < fH) & (target_patches_w >= 0) & (target_patches_w < fW)
            target_patches = target_patches_h[valid_patches_indices] * fW + target_patches_w[valid_patches_indices]
            len_target_patches = min(len(target_patches), target_patches_indices.shape[1])
            target_patches_indices[b, :len_target_patches] = target_patches[:len_target_patches]
        else:
            # occlussion
            useless_idx = full_idx[b]

            random_index = torch.randint(0, len(useless_idx), (1,)).item()  # generate random index
            useless_idx = useless_idx[random_index]  # get random index
            useless_h = useless_idx // fW
            useless_w = useless_idx % fW
            useless_patch_img = other_img[b, :, useless_h * stride_size[0]:useless_h * stride_size[0]+16, useless_w * stride_size[1]:useless_w * stride_size[1]+16].clone()
            useless_patch_img = to_pil_transform(useless_patch_img)  # Tensor -> PIL image

            if occlussion_locations[b] == 0:    # cover left
                useless_patch_img = resize_transformW(useless_patch_img)
                useless_patch_img = to_tensor_transform(useless_patch_img)  # PIL image -> Tensor
                img[b, :, 0:height, 0:width//2] = useless_patch_img
                target_patches_indices[b, :len(left_indices)] = left_indices
                
            elif occlussion_locations[b] == 1:  # cover right
                useless_patch_img = resize_transformW(useless_patch_img)
                useless_patch_img = to_tensor_transform(useless_patch_img)
                img[b, :, 0:height, width//2:width] = useless_patch_img
                target_patches_indices[b, :len(right_indices)] = right_indices
                
            elif occlussion_locations[b] == 2:  # cover bottom
                useless_patch_img = resize_transformH(useless_patch_img)
                useless_patch_img = to_tensor_transform(useless_patch_img)
                img[b, :, height//2:height, 0:width] = useless_patch_img
                target_patches_indices[b, :len(bottom_indices)] = bottom_indices
            else:   # cover top
                useless_patch_img = resize_transformH(useless_patch_img)
                useless_patch_img = to_tensor_transform(useless_patch_img)
                img[b, :, 0:height//2, 0:width] = useless_patch_img
                target_patches_indices[b, :len(top_indices)] = top_indices
                
    return img, target_patches_indices
