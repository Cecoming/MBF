import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

from model.human_parsing_net import HumanParsingNet
from loss.part_mask_loss import KLabelSmoothedCrossEntropyLoss
from utils.get_occ_img import find_diff_id_images, cover_pixels



def do_train(cfg,
             model,
             # center_criterion,
             train_loader,
             val_loader,
             optimizer,
             # optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    segment_model = HumanParsingNet(cfg)
    if device:
        model.to(local_rank)
        segment_model.to(local_rank)

        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            segment_model.make_parallel()
            model = nn.DataParallel(model)
    class_weights = torch.tensor([0.5, 1.0, 0.8, 0.8, 1.0])
    segment_loss = KLabelSmoothedCrossEntropyLoss(epsilon=0.1, class_weights=class_weights)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    f_size = ((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0]+1, (cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1]+1)
    stride_size = cfg.MODEL.STRIDE_SIZE[0]
    K=4
    get_idx = True
    logger.info('stride_size = {}'.format(stride_size))
    logger.info('class_weights = {}'.format(class_weights))

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, feature_dim=768)
    scaler = amp.GradScaler()

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, (img, img_occ, o_mask, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            # optimizer_center.zero_grad()
            img = img.to(device)
            img_occ = img_occ.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            # -----------------change start------------------
            with amp.autocast(enabled=True):
                h_score, h_feat, h_patch_cls_score, h_idxs, loss_orth_h, feat_patch_h = model(img, target, cam_label=target_cam, view_label=target_view,
                                                                                get_idx=get_idx, f_size=f_size, is_occ=False)
                ################from segment model get part masks
                # 1. holistic image part mask
                with torch.no_grad():
                    h_image_part_mask_ = segment_model.get_batch_part_masks(img)
                    h_image_part_mask_ = torch.from_numpy(h_image_part_mask_).to(device)    #[B, H, W]
                    h_image_part_mask = segment_model.custom_maxfrequent_downsample(h_image_part_mask_, K, stride_size)

                    # occluded patch mask
                    idx = h_idxs[0]  # shape: [B, T]
                    mask_ = torch.zeros_like(h_image_part_mask)  # shape: [B, fH, fW]
                    mask_ = mask_.view(mask_.shape[0], -1)  # shape: [B, fH*fW]
                    mask_.scatter_(1, idx, 1)   # 将掩码中idx索引位置的值设置为1
                    mask_ = mask_.view(mask_.shape[0], h_image_part_mask.shape[1], h_image_part_mask.shape[2])
                    # 5. 将image_part_mask与掩码相乘
                    h_image_part_mask = h_image_part_mask * mask_

                loss_seg_h = segment_loss(h_patch_cls_score,  h_image_part_mask) 
                loss_reid_h = loss_fn(h_score, h_feat, target, target_cam)

                ####################################################################################
                # 2. occluded image part mask
                random_indices, img_other = find_diff_id_images(img)
                img_occ_person, covered_idx = cover_pixels(img_occ, img_other, h_image_part_mask_[random_indices], h_image_part_mask[random_indices], beta=cfg.MODEL.BETA, stride_size=cfg.MODEL.STRIDE_SIZE, patch_size=16)

                del img_other, random_indices

                image_shape = (img.shape[2], img.shape[3])
                image_part_mask = segment_model.get_occ_mask(h_image_part_mask, o_mask, image_shape, stride_size)
                ###将covered_idx指向的位置在image_part_mask置为0
                # 创建一个与image_part_mask形状相同的全1掩码
                covered_mask = torch.ones_like(h_image_part_mask)
                covered_mask = covered_mask.view(covered_mask.shape[0], -1)  # shape: [B, fH*fW]
                covered_mask.scatter_(1, covered_idx, 0)
                covered_mask = covered_mask.view(covered_mask.shape[0], h_image_part_mask.shape[1], h_image_part_mask.shape[2])  # shape: [B, fH, fW]
                # 将image_part_mask中covered_idx指定位置置为0
                image_part_mask = image_part_mask * covered_mask


                score, feat, de_score, de_feat, patch_cls_score, idxs, loss_orth_o, feat_patch_o = model(img_occ_person, target, cam_label=target_cam, view_label=target_view,
                                                                                      get_idx=get_idx, f_size=f_size, is_occ=True)
                with torch.no_grad():
                    # occluded patch mask
                    idx = idxs[0]  # shape: [B, T]
                    mask_ = torch.zeros_like(image_part_mask)  # shape: [B, fH, fW]
                    mask_ = mask_.view(mask_.shape[0], -1)  # shape: [B, fH*fW]
                    mask_.scatter_(1, idx, 1)   # 将掩码中idx索引位置的值设置为1
                    mask_ = mask_.view(mask_.shape[0], image_part_mask.shape[1], image_part_mask.shape[2])
                    # 5. 将image_part_mask与掩码相乘
                    image_part_mask = image_part_mask * mask_


                loss_seg_o = segment_loss(patch_cls_score,  image_part_mask) 
                loss_reid_o = loss_fn(score, feat, target, target_cam)
                loss_reid_o_de = loss_fn(de_score, de_feat, target, target_cam, de=True)


                mse_loss = nn.MSELoss()
                loss_mse = mse_loss(feat_patch_h, feat_patch_o)

                loss_orth = (loss_orth_h + loss_orth_o)
                loss_seg = loss_seg_h + loss_seg_o
                loss = loss_reid_h + loss_reid_o + loss_seg + loss_reid_o_de + loss_orth + loss_mse

            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (h_score[0].max(1)[1] == target).float().mean() # top-1 accuracy
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])   
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
            # -----------------change end------------------
        # 
        
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, vis = model(img, cam_label=camids, view_label=target_view, get_idx=get_idx)
                            evaluator.update((feat, vid, camid, vis))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view, get_idx=get_idx, f_size=f_size)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                logger.info('Best result: {} {:.1%} {:.1%}' .format(top[0], top[1], top[2]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    f_size = ((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0]+1, (cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1]+1)
    get_idx = True

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view, get_idx=get_idx, f_size=f_size)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


