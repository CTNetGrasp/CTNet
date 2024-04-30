from utils.loss import compute_loss
from loguru import logger
import torch
from torch.utils.data import DataLoader
from args import parse_args
from datasets.referRGBDataset import RefOCIDGrasp
from models.clip_corse2fine import CTNet
import torch.optim as optim
import os
import shutil
from utils.evaluation import caculate_seg_iou, count_grasp_correct

from tqdm import tqdm
import numpy as np

from utils import utils_

from utils.draw import draw_img_grasp

from utils.post_process import post_process_output
from utils.evaluation import detect_grasps


import torch.distributed as dist
import wandb

def train_one_epoch(args, epoch, network, train_data, optimizer, lr_scheduler):

    network.train()
    metric_logger = utils_.MetricLogger(delimiter="  ")

    header = 'Epoch: [{}]'.format(epoch)
    batch_idx = 0

    # for img, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, idx in tqdm(train_data, desc="train"):
    for img, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, idx in metric_logger.log_every(train_data, args.print_freq, header):
        batch_idx += 1
        img = img.cuda(non_blocking=True)
        word_embeddings = word_embeddings.cuda(non_blocking=True)
        word_attention_mask = word_attention_mask.cuda(non_blocking=True)
        pos_img = pos_img.cuda(non_blocking=True)
        cos_img = cos_img.cuda(non_blocking=True)
        sin_img = sin_img.cuda(non_blocking=True)
        width_img = width_img.cuda(non_blocking=True)
        refmask = refmask.cuda(non_blocking=True)
        
        pred_seg, pos, cos, sin, width = network(img, word_embeddings, word_attention_mask)

        loss = compute_loss(pred_seg, refmask, [pos, cos, sin, width], [pos_img, cos_img, sin_img, width_img])

        optimizer.zero_grad()
        loss.backward()


        optimizer.step()
        lr_scheduler.step()
        torch.cuda.synchronize()

        if args.local_rank == 0:
            metric_logger.update(loss=loss)
            wandb.log({'loss':loss,'epoch': epoch})

        if batch_idx % args.print_freq == 0:

            MIoU_list, I_list, U_list = caculate_seg_iou(pred_seg, refmask)
            bs_miou = np.mean(np.array(MIoU_list))

            grasp_num, max_iou_list = count_grasp_correct([pos, cos, sin, width], refgrasps)

            bs_gacc =  grasp_num / (args.batch_size//utils_.get_world_size())
            bs_giou = np.mean(max_iou_list)

            metric_logger.update(gacc=bs_gacc, giou=bs_giou, miou=bs_miou)
            if args.local_rank == 0:
                wandb.log({'bs_gacc':bs_gacc, 'bs_giou':bs_giou, 'bs_miou':bs_miou})

            if args.local_rank == 0 and batch_idx == 10 and args.visualize:
                pos_gt, angle_gt, width_gt = post_process_output(pos_img, cos_img, sin_img, width_img)
                draw_img_grasp(img[0], refgrasps[0], refmask[0], [pos_gt[0], angle_gt[0], width_gt[0]], f'train/train_gt_{epoch}')

                pos_pre, angle_pre, width_pre = post_process_output(pos, cos, sin, width)

                pred_mask = np.array(pred_seg.detach().cpu().numpy() > 0.35)

                pre_grasp = detect_grasps(pos_pre[0], angle_pre[0], width_pre[0])[0]

                draw_img_grasp(img[0], pre_grasp, pred_mask[0], [pos_pre[0], angle_pre[0], width_pre[0]], f'train/train_pre_{epoch}')

            del pred_seg, pos, cos, sin, width
            del img, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, idx
            del loss

    return metric_logger

@torch.no_grad()
def validate(args, network, val_data, epoch):
    network.eval()

    
    logger.info("validating ...")
    MIoU_list = []
    loss_list = []
    total_giou_list = []
    total_cor_num = 0
    total = 0
    for img, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, idx in val_data:
        img = img.cuda(non_blocking=True)
        word_embeddings = word_embeddings.cuda(non_blocking=True)
        word_attention_mask = word_attention_mask.cuda(non_blocking=True)
        pos_img = pos_img.cuda(non_blocking=True)
        cos_img = cos_img.cuda(non_blocking=True)
        sin_img = sin_img.cuda(non_blocking=True)
        width_img = width_img.cuda(non_blocking=True)
        refmask = refmask.cuda(non_blocking=True)

        pred_seg, pos, cos, sin, width = network(img, word_embeddings, word_attention_mask)

        loss = compute_loss(pred_seg, refmask, [pos, cos, sin, width], [pos_img, cos_img, sin_img, width_img])
        loss_list.append(loss.item())

        _MIoU_list, I_list, U_list = caculate_seg_iou(pred_seg, refmask)
        MIoU_list += _MIoU_list
        flag, maxiou_list = count_grasp_correct([pos, cos, sin, width], refgrasps)
        total_cor_num += flag
        total += len(refgrasps)

        total_giou_list += maxiou_list

        if args.local_rank == 0 and args.visualize:
            for i, maxiou in range(maxiou_list):
                if maxiou >= 0.65:
                    pos_gt, angle_gt, width_gt = post_process_output(pos_img, cos_img, sin_img, width_img)

                    draw_img_grasp(img[i], refgrasps[i], refmask[i], [pos_gt[i], angle_gt[i], width_gt[i]], f'{epoch}_valid_{idx[i]}_gt')

                    pos_pre, angle_pre, width_pre = post_process_output(pos, cos, sin, width)

                    pred_mask = np.array(pred_seg.detach().cpu().numpy() > 0.35)

                    pre_grasp = detect_grasps(pos_pre[i], angle_pre[i], width_pre[i])[0]

                    draw_img_grasp(img[i], pre_grasp, pred_mask[i], [pos_pre[i], angle_pre[i], width_pre[i]], f'{epoch}_valid_{idx[i]}_pre_{maxiou:.3f}')
    
    miou = np.mean(np.array(MIoU_list))
    mloss = np.mean(np.array(loss_list))
    giou = total_cor_num / total
    gmiou = np.mean(np.array(total_giou_list))
    print("lenght of miou:", len(total_giou_list))
    if args.local_rank == 0:
        wandb.log({'v_miou':miou, 'v_giou':giou, 'v_gmiou':gmiou, 'v_loss:':mloss,'epoch': epoch})
        logger.info("MIoU: {:.4f}, GraspAcc: {:.4f}, gmiou: {:.4f}, correct grasp num: {}, total grasp num: {}, Mloss: {:.4f}".format(miou, giou, gmiou, total_cor_num, total, mloss))
    return miou, giou



def run():
    
    utils_.setup_seed(1024)
    args = parse_args()

    if args.local_rank == 0:
        wandb.init(config=args,
                project="refgrasp",
                dir="wandbDir",
                reinit=True)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    utils_.init_distributed_mode(args)


    train_dataset = RefOCIDGrasp('./data', "train", tokenizer_name=args.tokenizer , max_tokens=args.txt_length, output_size=args.img_size, use_bbox=args.use_box, 
                                 use_mask=args.use_mask, include_rgb=args.use_rgb, include_depth=args.use_depth, 
                                 random_rotate=args.random_rotate, random_crop=args.random_crop, random_bright=args.random_bright)
    
    val_dataset = RefOCIDGrasp('./data', "val", tokenizer_name=args.tokenizer, max_tokens=args.txt_length, output_size=args.img_size,use_bbox=args.use_box, 
                               use_mask=args.use_mask, include_rgb=args.use_rgb, include_depth=args.use_depth)
    
    logger.info(f"local rank {args.local_rank} / global rank {utils_.get_rank()} successfully built train dataset.")
    num_tasks = utils_.get_world_size()
    global_rank = utils_.get_rank()


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size//num_tasks, num_workers=args.workers, collate_fn=train_dataset.collate, sampler=train_sampler)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size//num_tasks, num_workers=args.workers, collate_fn=train_dataset.collate, sampler=val_sampler)



    logger.info('Loading Network...')
    net = CTNet(args)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = net.module
    if args.local_rank == 0:
        logger.info(single_model)
        logger.info(f"trainable parameters: {sum(p.numel() for p in single_model.parameters() if p.requires_grad)/1024/1024:.2f}")
    # resume training
    if args.resume:
        logger.info(f"loading {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])

        
    clip_params = []
    other_params = []

    for name, para in single_model.named_parameters():
        if para.requires_grad:
            if "encoder" in name and 'fusion' not in name:
                clip_params += [para]
            else:
                other_params += [para]

    params = [
    {"params": clip_params, "lr": args.clip_lr},
    {"params": other_params, "lr": args.lr, "weight_decay":args.weight_decay},
    ]
    optimizer = optim.AdamW(params, lr=args.lr)


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)


    if args.resume:
        print(f"resum from {args.resume}")
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
        print('resume completely!!!')
       
    else:
        resume_epoch = -999

    logger.info('Done')
    

    best_iou = 0.0
    save_flag = False
    
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        train_data_loader.sampler.set_epoch(epoch)
        logger.info('Begining Epoch {:02d}'.format(epoch))
        train_results = train_one_epoch(args, epoch, net, train_data_loader, optimizer, lr_scheduler)

        miou, giou = validate(args, net, val_data_loader, epoch)
        if giou >= best_iou:
            best_iou = giou
            save_flag = True

        dict_to_save = {'state_dict': single_model.state_dict(),
                        'optimizer': optimizer.state_dict(), 
                        'args': args,
                        'epoch': epoch,
                        'best_grasp_iou': best_iou,
                        'lr_scheduler': lr_scheduler.state_dict()}
        
        if args.local_rank == 0:
            lastname = os.path.join(args.output_dir, f"{args.model_name}_last_model_gpus.pth")
            utils_.save_on_master(dict_to_save, lastname)


        if args.local_rank == 0:
            if save_flag:
                save_flag = False
                bestname = os.path.join(args.output_dir, f"{args.model_name}_best_model_gpus.pth")
                shutil.copyfile(lastname, bestname)
    if args.local_rank == 0:
        wandb.finish()

if __name__ == '__main__':
    run()

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 1024 train_dual_multigpu.py --batch_size 64 --output_dir checkpoints --use_mask --random_rotate --random_crop --random_bright 2>&1 | tee ./terminal_log/referGrasp/multigpu_train_output