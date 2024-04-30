from utils.loss import grasp_loss, seg_loss
from loguru import logger
import torch
from torch.utils.data import DataLoader
from args import parse_args
from datasets.referRGBDataset import RefOCIDGrasp
from models.clip_corse2fine import CTNet
import torch.optim as optim
import time
import os
import shutil
from utils.evaluation import calculate_iou_match, caculate_seg_iou, count_grasp_correct, print_result
from utils.post_process import post_process_output
from tqdm import tqdm
import numpy as np
from utils.draw import draw_img_grasp
import cv2 
from utils.evaluation import detect_grasps
from utils.grasp import asGraspRectangle


@torch.no_grad()
def validate(args, network, device, val_data):
    network.eval()

    logger.info("validating ...")
    SegIoU_list = []
    SegI_list = []
    SegU_list = []


    loss_list = []
    total = 0

    angle_flag_dict = {5:0, 10:0, 15:0, 20:0, 25:0, 30:0}
    iou_flag_dict = {0.2:0, 0.25:0, 0.3:0, 0.35:0, 0.40:0, 0.45:0, 0.5:0}
    seg_P_list = {0.5:0, 0.6:0, 0.7:0, 0.8:0, 0.9:0}
    seg_IoU_list = {'OIoU':0, 'MIoU':0}


    for img, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, idx in tqdm(val_data):
        img = img.to(device)
        word_embeddings = word_embeddings.to(device)
        word_attention_mask = word_attention_mask.to(device)
        pos_img = pos_img.to(device)
        cos_img = cos_img.to(device)
        sin_img = sin_img.to(device)
        width_img = width_img.to(device)
        refmask = refmask.to(device)

        pred_seg, pos, cos, sin, width = network(img, word_embeddings, word_attention_mask)

        loss = seg_loss(pred_seg, refmask)
        loss_list.append(loss.item())

        segIoU, SegI, SegU = caculate_seg_iou(pred_seg, refmask)
        SegIoU_list += segIoU
        SegI_list += SegI
        SegU_list += SegU



        for key, value in angle_flag_dict.items():
            flag, maxiou_list = count_grasp_correct([pos, cos, sin, width], refgrasps, angle_threshold=key, iou_threshold=0.25)
            angle_flag_dict[key] += flag
        

        for key, value in iou_flag_dict.items():
            flag, maxiou_list = count_grasp_correct([pos, cos, sin, width], refgrasps, angle_threshold=30, iou_threshold=key)
            iou_flag_dict[key] += flag

        total += len(refgrasps)

    for key, value in angle_flag_dict.items():
        angle_flag_dict[key] = value/total

    for key, value in iou_flag_dict.items():
        iou_flag_dict[key] = value/total

    for key, value in seg_P_list.items():
        seg_P_list[key] = (np.array(SegIoU_list) >= key).mean()
    
    seg_IoU_list['MIoU'] = np.array(SegIoU_list).mean()
    seg_IoU_list['OIoU'] = np.array(SegI_list).sum() / (np.array(SegU_list).sum() + 1e-6)
    
    
    print_result(angle_flag_dict=angle_flag_dict, iou_flag_dict=iou_flag_dict, seg_iou_dict=seg_IoU_list, seg_p_dict=seg_P_list)
    
    

def run():
    args = parse_args()
    if args.log_name == '':
        log_name = args.split
    else:
        log_name = args.log_name
    logger.add(f'terminal_log/{args.model_name}/{log_name}.log')
    device = torch.device(f"cuda:{args.device}")
    val_dataset = RefOCIDGrasp('./data', args.split, tokenizer_name=args.tokenizer, include_rgb=args.use_rgb, include_depth=args.use_depth, use_bbox=args.use_box, use_mask=args.use_mask, output_size=args.img_size, max_tokens=args.txt_length)

    val_data = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=val_dataset.collate)

    logger.info('Loading Network...')
    net = CTNet(args)
    logger.info(net)
    checkpoint = torch.load(args.resume, map_location='cpu')
    logger.info('Done')
    net.load_state_dict(checkpoint['state_dict'])


    net.to(device)
    validate(args, net, device, val_data)
    


if __name__ == '__main__':
    run()
    
# python test_dual_evaluation.py --use_mask  --device 1 --split val --resume xx 