import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import random

def miou(x, mmix, H, W, num):
    B, K = mmix.shape
    inter = torch.sum(mmix, 1)
    x  = x.reshape(1, H, W)
    for i in range(W):
        if i == 0:
            x_roll = x.reshape(1, -1)
            intra_matrix = torch.mul(x_roll, mmix)
            intra = torch.matmul(x_roll, mmix.t())
            iou_x = intra / inter
            iou = iou_x
        else:
            x_roll = torch.roll(x, i, 2)
            x_roll = x_roll.reshape(1, -1)
            intra_matrix = torch.cat((intra_matrix, torch.mul(x_roll, mmix)), 0)
            intra = torch.matmul(x_roll, mmix.t())
            iou = torch.cat((iou, intra / inter), 1)
    iou_list, rank_list = torch.sort(iou, descending=True, dim=1)
    range_end = int(B*0.1)
    for attempt in range(100):
        select_rank = random.sample(range(1,range_end), 1)
        select = rank_list[:, select_rank]
        real_max = iou_list[:, select_rank]
        intra_select = intra_matrix[select,:]
        shift = math.floor(select / B)
        select = select - shift * B
        real = iou_x[:, select]
        if select != num and real> 0.7:
            return select, shift
    select_rank = random.sample(range(0, 1), 1)
    select = rank_list[:,select_rank]
    shift = math.floor(select / B)
    select = select - shift * B
    return select, shift

         
def SelectMix(mmix, num, slabel, label, label_new, probability=0.0):
    B, N = mmix.shape
    if random.uniform(0, 1) > probability:
        slabel[num] = B
        return slabel, label_new
    else:
        W = int((N / 2.0) ** 0.5)
        H = int(N / W)
        mmix_select = mmix[num,:].unsqueeze(0)
        select, shift = miou(mmix_select, mmix, H, W, num)
        slabel[num] = select[0,0]
        label_new[num] = label[select[0,0]]
        return slabel, label_new
        
def do_train(cfg,
             model,
             model_pt,
             n_class,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
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
    if device:
        model.to(local_rank)
        model_pt.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            model_pt = torch.nn.parallel.DistributedDataParallel(model_pt, device_ids=[local_rank], find_unused_parameters=True)
            
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        model_pt.eval()
        
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            one_hot = torch.index_select(torch.eye(n_class), dim = 0, index = vid)
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            slabel = torch.zeros(target.shape).to(device)
            one_hot = one_hot.to(device)
            target_new = target.clone()
            
            add_mask = 0
            
            with torch.no_grad():
                mmix = model_pt(img, img, add_mask, target, cam_label=target_cam, view_label=target_view)
            mmix_ones = torch.ones(mmix.shape).to(device)
            mmix_zeros = torch.zeros(mmix.shape).to(device)
            mmix = torch.where(mmix>=0.5, mmix_ones, mmix_zeros)
            
            for i in range (target.shape[0]):
                slabel, target_new = SelectMix(mmix, i, slabel, target, target_new)
            
            one_hotN = torch.index_select(torch.eye(n_class), dim = 0, index = target_new.cpu()).to(device)
            sub_hot = torch.mul(one_hot, one_hotN)
            sub_flag =  torch.sum(sub_hot, 1).unsqueeze(1)
            sub_hot = one_hotN - (sub_flag * one_hot)
            sub_tri = torch.matmul(one_hotN, one_hot.t()) * (1.0 - sub_flag)
            ##################################################################################
            optimizer.zero_grad()
            with amp.autocast(enabled=True):
                
                score, feat = model(img, mmix, slabel, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, sub_hot, sub_tri, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

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
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
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
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
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
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


