import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2


def create_heatmap_image(image, heatmap, iter_index, epoch, sv_dir='', threshold=0.4, is_occ=True, attn=None, attns=None):
    B, K, fH, fW = heatmap.shape
    B, C, H, W = image.shape
    
    # 将heatmap上采样至图像的尺寸 (B, K, H, W)
    upscaled_heatmaps = torch.nn.functional.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False)  # (B, K, H, W)

    final_concatenated_images = []
    add_text = False

    for b in range(B):
        original_image = image[b].cpu().detach().permute(1, 2, 0).numpy()    # (H, W, 3)
        original_image = (original_image + 1) / 2 * 255
        original_image_pil = Image.fromarray(original_image.astype(np.uint8))
        # original_image_pil.save('./visualization/heatmap/batch'+str(iter_index+1)+'.png')

        # 准备横向拼接的图像列表，首先添加原图
        image_row = [original_image_pil]

        # 遍历每个heatmap并叠加
        for k in range(K):
            heatmap_single = upscaled_heatmaps[b, k].cpu().detach().numpy()
            heatmap_single_f = heatmap[b, k].cpu().detach().numpy()
            # 将单个heatmap转换成颜色图像
            heatmap_color = plt.cm.jet(heatmap_single)[:, :, :3]  # 取RGB通道
            heatmap_color = np.uint8(heatmap_color * 255)
            heatmap_pil = Image.fromarray(heatmap_color)
            
            # 将heatmap叠加到原图
            blended_image = Image.blend(original_image_pil, heatmap_pil, alpha=0.5)
            
            # 得分可视化
            if add_text:
                # 在热力图上标注注意力得分
                heatmap_with_scores = np.array(blended_image)
                font_scale = 0.2  # 字体大小
                for y in range(fH):
                    for x in range(fW):
                        score = heatmap_single_f[y, x]
                        cv2.putText(heatmap_with_scores, f"{score:.2f}", (x * (W // fW), y * (H // fH) + (H // fH)//2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

                heatmap_with_scores_pil = Image.fromarray(heatmap_with_scores)
                image_row.append(heatmap_with_scores_pil)
            else:
                image_row.append(blended_image)

        # 将当前行的图片横向拼接
        concatenated_image_row = Image.new('RGB', ((1 + K) * W, H))
        x_offset = 0
        for img in image_row:
            concatenated_image_row.paste(img, (x_offset, 0))
            x_offset += W

        # 将当前行的图片添加到最终纵向拼接列表
        final_concatenated_images.append(concatenated_image_row)

    # 纵向拼接所有行
    total_width = (1 + K) * W
    total_height = B * H
    final_image = Image.new('RGB', (total_width, total_height))
    y_offset = 0
    for img in final_concatenated_images:
        final_image.paste(img, (0, y_offset))
        y_offset += H

    if sv_dir:
        save_dir = os.path.join(sv_dir, 'visualization')
        os.makedirs(save_dir, exist_ok=True)
        if is_occ:
            final_image.save(os.path.join(save_dir, 'occ_epoch'+str(epoch)+'_batch'+str(iter_index+1)+'.png'))
        else:
            final_image.save(os.path.join(save_dir, 'hol_epoch'+str(epoch)+'_batch'+str(iter_index+1)+'.png'))
    
    if attn is not None:
        attn = attn.view(B, fH, fW)
        
        # 将注意力得分上采样到与图像相同大小
        attn_resized = torch.nn.functional.interpolate(attn.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        
        # 准备一个列表来保存所有可视化后的图像
        visualized_images = []
        
        for i in range(B):
            img = image[i].cpu().numpy()  # (C, H, W)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            
            attn_map = attn_resized[i].cpu().numpy()
            
            # 归一化注意力得分到[0, 255]范围
            # 归一化注意力得分到[0, 1]范围
            attn_map_normalized = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

            # 反转归一化后的注意力得分
            attn_map_normalized = 1 - attn_map_normalized
            # attn_map_normalized = 1 - attn_map

            # 将归一化的注意力得分映射到[0, 255]范围
            attn_map = np.uint8(attn_map_normalized * 255)
            
            # 将单通道的热力图转换为三通道
            heatmap_color = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
            
            # 叠加注意力得分到原图
            superimposed_img = cv2.addWeighted(img.transpose(1, 2, 0), 0.6, heatmap_color, 0.4, 0)
            
            if add_text:
                # 在图像上标注注意力得分
                step_h = H // fH
                step_w = W // fW
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.2  # 字体大小，调小字体以适应更密集的标注
                thickness = 1
                for y in range(fH):
                    for x in range(fW):
                        score = attn[i, y, x].item()
                        text = f"{score:.2f}"
                        position = (x * step_w, y * step_h + step_h // 2)
                        cv2.putText(superimposed_img, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # 将图像转换为PIL格式并保存
            visualized_images.append(superimposed_img)
        
        # 将所有图像拼接成一张大图
        num_images = len(visualized_images)
        num_cols = 8  # 每行显示的图片数量
        num_rows = (num_images + num_cols - 1) // num_cols  # 计算需要的行数
        
        # 创建一个空白画布
        canvas_width = W * num_cols
        canvas_height = H * num_rows
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # 将所有图像填充到画布上
        for idx, img in enumerate(visualized_images):
            row = idx // num_cols
            col = idx % num_cols
            y_start = row * H
            x_start = col * W
            canvas[y_start:y_start + H, x_start:x_start + W, :] = img
        
        # 将画布转换为PIL格式并保存
        canvas_pil = Image.fromarray(canvas)
        if is_occ:
            canvas_pil.save(os.path.join(save_dir, 'occ_epoch'+str(epoch)+'_batch'+str(iter_index+1)+'_attn.png'))
        else:
            canvas_pil.save(os.path.join(save_dir, 'hol_epoch'+str(epoch)+'_batch'+str(iter_index+1)+'_attn.png'))

    if attns is not None:
        attns = attns.view(B, fH, fW)
        
        # 将注意力得分上采样到与图像相同大小
        attns_resized = torch.nn.functional.interpolate(attns.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        
        # 准备一个列表来保存所有可视化后的图像
        visualized_images = []
        
        for i in range(B):
            img = image[i].cpu().numpy()  # (C, H, W)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            
            attns_map = attns_resized[i].cpu().numpy()
            
            # 归一化注意力得分到[0, 255]范围
            # 归一化注意力得分到[0, 1]范围
            attns_map_normalized = (attns_map - attns_map.min()) / (attns_map.max() - attns_map.min())

            # 反转归一化后的注意力得分
            attns_map_normalized = 1 - attns_map_normalized
            # attns_map_normalized = 1 - attns_map

            # 将归一化的注意力得分映射到[0, 255]范围
            attns_map = np.uint8(attns_map_normalized * 255)
            
            # 将单通道的热力图转换为三通道
            heatmap_color = cv2.applyColorMap(attns_map, cv2.COLORMAP_JET)
            
            # 叠加注意力得分到原图
            superimposed_img = cv2.addWeighted(img.transpose(1, 2, 0), 0.6, heatmap_color, 0.4, 0)
            
            # 在图像上标注注意力得分
            step_h = H // fH
            step_w = W // fW
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.2  # 字体大小，调小字体以适应更密集的标注
            thickness = 1
            for y in range(fH):
                for x in range(fW):
                    score = attns[i, y, x].item()
                    text = f"{score:.2f}"
                    position = (x * step_w, y * step_h + step_h // 2)
                    cv2.putText(superimposed_img, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # 将图像转换为PIL格式并保存
            visualized_images.append(superimposed_img)
        
        # 将所有图像拼接成一张大图
        num_images = len(visualized_images)
        num_cols = 8  # 每行显示的图片数量
        num_rows = (num_images + num_cols - 1) // num_cols  # 计算需要的行数
        
        # 创建一个空白画布
        canvas_width = W * num_cols
        canvas_height = H * num_rows
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # 将所有图像填充到画布上
        for idx, img in enumerate(visualized_images):
            row = idx // num_cols
            col = idx % num_cols
            y_start = row * H
            x_start = col * W
            canvas[y_start:y_start + H, x_start:x_start + W, :] = img
        
        # 将画布转换为PIL格式并保存
        canvas_pil = Image.fromarray(canvas)
        if is_occ:
            canvas_pil.save(os.path.join(save_dir, 'occ_epoch'+str(epoch)+'_batch'+str(iter_index+1)+'_attns.png'))
        else:
            canvas_pil.save(os.path.join(save_dir, 'hol_epoch'+str(epoch)+'_batch'+str(iter_index+1)+'_attns.png'))        


# if __name__ == "__main__":
    # Assuming img and heatmap are PyTorch tensors with the correct shape
    # img: (B, 3, H, W)
    # heatmap: (B, K, fH, fW)

    # Call the function to perform all operations and save the image
    # save_concatenated_heatmap_images(img, heatmap)
