from skimage.feature import peak_local_max
import numpy as np
from PIL import Image, ImageDraw
from skimage.draw import polygon
from .grasp import asGrasp, asGraspRectangle

def detect_grasps(q_img, ang_img, width_img, no_grasps=1, width=None):
    """
    Detect grasps in a GG-CNN output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img:  length image network output
    :param no_grasps: Max number of grasps to return
    :param width: the width of finger
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.1, num_peaks=no_grasps)
    # print(local_max)
    grasps = []
    grasps_center_angle_l = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        grasp_angle = np.rad2deg(ang_img[grasp_point])
        if width_img is not None:
            length = width_img[grasp_point]
        if width is None:
            if length > 30:
                width = length * 0.33
            else:
                width = length * 0.5

        grasps_center_angle_l.append([grasp_point[1], grasp_point[0], grasp_angle, length])

        x1 = grasp_point[1] - length/2
        y1 = grasp_point[0] - width/2
        x2 = grasp_point[1] + length/2
        y2 = grasp_point[0] + width/2
        g = [x1, y1, grasp_angle, x2, y2]
        grasps.append(g)

    return grasps, grasps_center_angle_l


def calculate_iou_match(GT_grasps, pred_grasps_img, no_grasps=1, angle_threshold=np.pi/6, iou_threshold=0.25):
    '''
    GT_grasps: [[x1, y1, theta, x2, y2], ...]
    '''
    assert len(pred_grasps_img) == 3
    grasp_q, grasp_angle, grasp_width = pred_grasps_img
    gs, g_c_a_l = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    Miou = 0
    for g_c in gs:  
        Miou = max_iou(g_c, GT_grasps, angle_threshold)
        if Miou > iou_threshold:
            return True, Miou
    else:
        return False, Miou
    
def max_iou(g_c, GT_g, angle_threshold=np.pi/6):
    """
    g_c: [x1, y1, theta, x2, y2] [c_x, c_y, angle, length]
    GT_g: [x1, y1, theta, x2, y2]
    Return maximum IoU between self and a list of GraspRectangles
    :return: Maximum IoU with any of the GraspRectangles
    """
    max_iou = 0
    x1_p, y1_p, theta_p, x2_p, y2_p = g_c
    angle = theta_p
    length = abs(x2_p - x1_p)

    if length > 150:
        return 0

    for g_ in GT_g:
        height = abs(g_[4] - g_[1])  

        y1 = y1_p - height/2
        y2 = y2_p + height/2
        g = [x1_p, y1, angle, x2_p, y2] 
        iou = get_iou(g, g_, angle_threshold)

        max_iou = max(max_iou, iou)

    return max_iou

def get_iou(pre_g, GT_g, angle_threshold=np.pi/6):
    '''
    pre_g: [x1, y1, theta, x2, y2]
    GT_g:  [x1, y1, theta, x2, y2]
    '''
    pre_angle = pre_g[2]
    GT_angle = GT_g[2]

    pre_gr = asGraspRectangle(pre_g)
    GT_gr = asGraspRectangle(GT_g)


    if abs((np.deg2rad(pre_angle - GT_angle) + np.pi/2) % np.pi - np.pi/2) > angle_threshold:
        return 0

    rr1, cc1 = polygon(pre_gr[:, 1], pre_gr[:, 0])
    
    rr2, cc2 = polygon(GT_gr[:, 1], GT_gr[:, 0])

    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except:
        print("error in get_iou!")
        return 0

    canvas = np.zeros((r_max, c_max))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)
    return intersection/union

import torch
from torchvision.transforms import functional as F
import copy

def showResult(img_norm, q_img, ang_img, width_img, GT = None, name='result_1'):
    grasp, g_c = detect_grasps(q_img, ang_img, width_img)

    mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
    
    MEAN = [-mean/std for mean, std in zip(mean, std)]
    STD = [1/s for s in std]
    img_norm = F.normalize(img_norm, MEAN, STD)
    image = F.to_pil_image(img_norm)
    fig = copy.deepcopy(image)
    # print("draw grasp")

    draw = ImageDraw.Draw(fig)

    
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = asGraspRectangle(grasp[0])
    draw.line((x1,y1,x2,y2), fill='blue', width=2)
    draw.line((x2,y2,x3,y3), fill='yellow', width=2)
    draw.line((x3,y3,x4,y4), fill='blue', width=2)
    draw.line((x4,y4,x1,y1), fill='yellow', width=2)

    if GT is not None:
        for gt_grasp in GT:
            [[x1_, y1_], [x2_, y2_], [x3_, y3_], [x4_, y4_]] = asGraspRectangle(gt_grasp.numpy())

            draw.line((x1_,y1_,x2_,y2_), fill='purple', width=2)
            draw.line((x2_,y2_,x3_,y3_), fill='green', width=2)
            draw.line((x3_,y3_,x4_,y4_), fill='purple', width=2)
            draw.line((x4_,y4_,x1_,y1_), fill='green', width=2)

    if name is not None:
        fig.save(name+'.png')
    
def caculate_seg_iou(preds, gts):
    preds = torch.sigmoid(preds)
    Miou_list = []
    I_list = []
    U_list = []
    for i, pred in enumerate(preds):
        gt = gts[i]
        pred = pred.detach().cpu().numpy()
        gt = gt.cpu().numpy()
        pred = np.array(pred > 0.35)
        gt = np.array(gt)
        inter = np.sum(np.logical_and(pred, gt))
        union = np.sum(np.logical_or(pred, gt))
        iou = inter / (union + 1e-6)
        Miou_list.append(iou)
        I_list.append(inter)
        U_list.append(union)

    
    return Miou_list, I_list, U_list


from utils.post_process import post_process_output

def count_grasp_correct(pred_grasp_img, refgrasps, angle_threshold=30, iou_threshold=0.25):
    angle_threshold = np.deg2rad(angle_threshold)
    pos_img_pred, cos_img_pred, sin_img_pred, width_img_pred = pred_grasp_img

    pos_im_gs, ang_img_gs, width_img_gs = post_process_output(pos_img_pred, cos_img_pred, sin_img_pred, width_img_pred)

    num = 0
    max_iou_list = []

    for i in range(len(refgrasps)):
        cim, maxiou = calculate_iou_match(refgrasps[i].numpy(), (pos_im_gs[i], ang_img_gs[i], width_img_gs[i]), angle_threshold=angle_threshold, iou_threshold=iou_threshold)
        num += cim
        max_iou_list.append(maxiou)

    return num, max_iou_list



if __name__=="__main__":

    grasps_gt = np.load('grasps.npy')

    rgb_img = np.load('rgb_img.npy')
    q_img = np.load('pos_img.npy')
    ang_img = np.load('ang_img.npy')
    width_img = np.load('width_img.npy')

    q_img = np.array(q_img)
    ang_img = np.array(ang_img)
    width_img = np.array(width_img)
    rgb_img = Image.fromarray(rgb_img)

    draw = ImageDraw.Draw(rgb_img)
    g, g_c = detect_grasps(q_img, ang_img, width_img)
    for g_ in g:
        p = asGraspRectangle(g_)
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = p

        draw.line((x1,y1,x2,y2), fill='blue', width=2)
        draw.line((x2,y2,x3,y3), fill='yellow', width=2)
        draw.line((x3,y3,x4,y4), fill='blue', width=2)
        draw.line((x4,y4,x1,y1), fill='yellow', width=2)
    rgb_img.save('show.png')

    # grasps_gt = detect_grasps(q_img, ang_img, width_img, 10)
    print(calculate_iou_match(grasps_gt, q_img, ang_img, width_img))

from loguru import logger
def print_result(angle_flag_dict, iou_flag_dict, seg_iou_dict, seg_p_dict):
    r = str('')
    for key, value in angle_flag_dict.items():
        r += f'angle {key:.2f}:{value:.4f}  '
    logger.info(f'Grasp IoU: 0.25 => {r}')

    r = str('')
    for key, value in iou_flag_dict.items():
        r += f'IoU {key:.2f}:{value:.4f}  '
    logger.info(f'Grasp angle: 30 => {r}')

    r = str('')
    for key, value in seg_iou_dict.items():
        r += f'{key}:{value:.4f}  '
    logger.info(f'Segmentation  IoU=> {r}')

    r = str('')
    for key, value in seg_p_dict.items():
        r += f'P@{key:.2f}:{value:.4f}  '
    logger.info(f'Segmentation precision => {r}')