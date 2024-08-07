import os
import torch
from torchvision.utils import save_image
from einops import rearrange


@torch.no_grad()
def visualize_mask(data_loader, model, device, output_dir, n_visualization, fuse_token, keep_rate=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visualize:'
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).reshape(3, 1, 1)

    # switch to evaluation mode
    model.eval()

    ii = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        B = images.size(0)

        with torch.cuda.amp.autocast():
            output, idx = model(images, keep_rate, get_idx=True)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # denormalize
        images = images * std + mean

        idxs = get_real_idx(idx, fuse_token)
        for jj, idx in enumerate(idxs):
            masked_img = mask(images, patch_size=16, idx=idx)
            save_img_batch(masked_img, output_dir, file_name='img_{}' + f'_l{jj}.jpg', start_idx=world_size * B * ii + rank * B)

        save_img_batch(images, output_dir, file_name='img_{}_a.jpg', start_idx=world_size * B * ii + rank * B)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.synchronize_between_processes()
        ii += 1
        if world_size * B * ii >= n_visualization:
            break

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def mask(x, idx, patch_size):
    """
    Args:
        x: input image, shape: [B, 3, H, W]
        idx: indices of masks, shape: [B, T], value in range [0, h*w)
    Return:
        out_img: masked image with only patches from idx postions
    """
    h = x.size(2) // patch_size
    x = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=patch_size, q=patch_size)  # [64, 768, 128]
    output = torch.zeros_like(x) - 1 # for non-mask black   # [64, 768, 128]
    idx1 = idx.unsqueeze(1).expand(-1, x.size(1), -1)       # [64, 768, 103]
    extracted = torch.gather(x, dim=2, index=idx1)  # [b, c p q, T], # [64, 768, 103], 从x中提取idx1指定的元素
    scattered = torch.scatter(output, dim=2, index=idx1, src=extracted) # [64, 768, 128], 将 extracted 中的元素放回output中idx1指定的位置，得到scattered
    out_img = rearrange(scattered, 'b (c p q) (h w) -> b c (h p) (w q)', p=patch_size, q=patch_size, h=h)
    return out_img


def get_deeper_idx(idx1, idx2):
    """
    Args:
        idx1: indices, shape: [B, T1]
        idx2: indices to gather from idx1, shape: [B, T2], T2 <= T1
    """
    return torch.gather(idx1, dim=1, index=idx2)


def get_real_idx(idxs, fuse_token):
    # nh = img_size // patch_size
    # npatch = nh ** 2

    # gather real idx
    for i in range(1, len(idxs)):
        tmp = idxs[i - 1]
        if fuse_token:
            B = tmp.size(0)
            tmp = torch.cat([tmp, torch.zeros(B, 1, dtype=tmp.dtype, device=tmp.device)], dim=1)
        idxs[i] = torch.gather(tmp, dim=1, index=idxs[i])
    return idxs


def save_img_batch(x, path, file_name='img{}', start_idx=0):
    for i, img in enumerate(x):
        save_image(img, os.path.join(path, file_name.format(start_idx + i)))
