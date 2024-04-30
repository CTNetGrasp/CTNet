import matplotlib.pyplot as plt
import torch
from .evaluation import detect_grasps, asGraspRectangle
import torch
from torchvision.transforms import functional as F
import copy
from PIL import ImageDraw
import numpy as np
import cv2
cv2.setNumThreads(0)
from PIL import Image

def apply_mask(image, mask, color=[0, 1, 0], alpha=0.4):
    """
        image: [H, W, 3]
        mask: [H, W]
    """
    if isinstance(image, Image.Image) or isinstance(image, torch.Tensor):
        image = np.array(image)
    if isinstance(mask, Image.Image) or isinstance(mask, torch.Tensor):
        mask = np.array(mask)

    if 1. in mask:
        mask = mask.astype(np.float32) * 255.
    mask = mask.astype(np.uint8)
    result = copy.deepcopy(image)
    for c in range(3):
        result[:, :, c] = np.where(mask == 255,
                                  result[:, :, c] *
                                  (1 - alpha) + alpha*color[c]* 255,
                                  result[:, :, c])
    

    ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.drawContours(result, contours, 0, (255, 255, 255), 1)

    return Image.fromarray(result)

def inNorm(img_norm):
    mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
    
    MEAN = [-mean/std for mean, std in zip(mean, std)]
    STD = [1/s for s in std]
    img_norm = F.normalize(img_norm, MEAN, STD)
    image = F.to_pil_image(img_norm)
    fig = copy.deepcopy(image)
    return fig

def showResult(img_norm, q_img, ang_img, width_img, GT = None, name='result_1'):
    grasp, g_c = detect_grasps(q_img, ang_img, width_img)


    # print("draw grasp")
    fig = inNorm(img_norm)
    draw = ImageDraw.Draw(fig)

    
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = asGraspRectangle(grasp[0])
    draw.line((x1,y1,x2,y2), fill='red', width=2)
    draw.line((x2,y2,x3,y3), fill='blue', width=2)
    draw.line((x3,y3,x4,y4), fill='red', width=2)
    draw.line((x4,y4,x1,y1), fill='blue', width=2)

    if GT is not None:
        for gt_grasp in GT:
            [[x1_, y1_], [x2_, y2_], [x3_, y3_], [x4_, y4_]] = asGraspRectangle(gt_grasp.numpy())

            draw.line((x1_,y1_,x2_,y2_), fill='purple', width=2)
            draw.line((x2_,y2_,x3_,y3_), fill='green', width=2)
            draw.line((x3_,y3_,x4_,y4_), fill='purple', width=2)
            draw.line((x4_,y4_,x1_,y1_), fill='green', width=2)

    if name is not None:
        fig.save(name+'.png')


def draw_img_grasp(norm_img, refgrasp=None, seg_mask=None, name=None):
    '''
    ori_img: [3, H, W]
    refgrasp: [x1, y1, theta, x2, y2]
    grasp_imgs: [pos, cos, sin, width]  3*[1, H, W]
    '''
    

    fig = inNorm(norm_img)
    # fig.save('ori_img.png')

    fig1 = copy.deepcopy(fig)
    draw1 = ImageDraw.Draw(fig1)
    if refgrasp is not None:
        if not isinstance(refgrasp, list):
            refgrasp = refgrasp.cpu().numpy()
        for grasp in refgrasp:
            [[x1_, y1_], [x2_, y2_], [x3_, y3_], [x4_, y4_]] = asGraspRectangle(grasp)
            draw1.line((x1_,y1_,x2_,y2_), fill='red', width=2)
            draw1.line((x2_,y2_,x3_,y3_), fill='blue', width=2)
            draw1.line((x3_,y3_,x4_,y4_), fill='red', width=2)
            draw1.line((x4_,y4_,x1_,y1_), fill='blue', width=2)
            if name is not None:
                fig1.save(f'{name}_grasp.jpg')
    
    
    fig3 = copy.deepcopy(fig)
    if seg_mask is not None:
        seg_mask = seg_mask.squeeze(0)
        if not isinstance(seg_mask, np.ndarray):
            seg_mask = seg_mask.cpu().numpy()
        fig3 = apply_mask(fig3, seg_mask)
        if name is not None:
            fig3.save(f'{name}_mask.jpg')


import seaborn as sns
def draw_hotmap(map, vmin=0, vmax=1, name=None):
    plt.figure(dpi=300)
    sns.heatmap(map, yticklabels=False, xticklabels=False, cmap='jet', vmin=vmin, vmax=vmax, cbar_kws={ "pad":0.01, "ticks":[vmin, vmax]})
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'{name}.jpg')
    plt.clf()
    plt.close("all") 


