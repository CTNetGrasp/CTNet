import matplotlib.pyplot as plt
import torch
from .args import parse_args
from .models.clip_corse2fine import CGSnet
import time
import os
import shutil
from utils.post_process import post_process_output
from tqdm import tqdm
import numpy as np
from utils.draw import draw_img_grasp, draw_hotmap, inNorm, apply_mask
import cv2 
from utils.evaluation import detect_grasps
import torch.nn.functional as F
from PIL import Image

from models.tokenizer import tokenize
import copy
from PIL import ImageDraw
from utils.evaluation import asGraspRectangle
from threading import Thread
import random

def generate_random_str(randomlength=5):
  random_str =''
  base_str ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
  length =len(base_str) -1
  for i in range(randomlength):
    random_str +=base_str[random.randint(0, length)]
  return random_str

random_str_name = None

@torch.no_grad()
def real_inference(net, img=None, text=None, device='cuda:0'):
    if img is None:
        img = cv2.imread("img.jpg")
    if text is None:
        text = 'The red box on the rear.'
        

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))

    img_show = img

    pad = (640 - 480)//2
    boader = 100
    img = cv2.copyMakeBorder(img, pad+boader,pad+boader, boader, boader,cv2.BORDER_CONSTANT, value=(200, 200, 200))
    img = cv2.resize(img, (320, 320))

    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img.astype(np.float32)) 

    mean = torch.tensor([0.48145466, 0.4578275,
                                0.40821073]).reshape(3, 1, 1)

    std = torch.tensor([0.26862954, 0.26130258,
                                0.27577711]).reshape(3, 1, 1)



    if not isinstance(img, torch.FloatTensor):
        img = img.float()

    img = img.div_(255.).sub_(mean).div_(std)
    ##############################################

    input_ids = tokenize(text,15, True).squeeze(0)

    input_ids = input_ids[:15]

    attention_mask = [0] * 15
    padded_input_ids = [0] * 15

    padded_input_ids[:len(input_ids)] = input_ids
    attention_mask[:len(input_ids)] = [1]*len(input_ids) 
    word_embeddings = torch.tensor(padded_input_ids) 
    word_attention_mask = torch.tensor(attention_mask) 
    kg_word_embeddings = torch.zeros(1, 3, 512)
    ###############################################


    img, word_embeddings, word_attention_mask = img.unsqueeze(0), word_embeddings.unsqueeze(0), word_attention_mask.unsqueeze(0)
    pred_seg, pos, cos, sin, width = net(img.to(device), word_embeddings.to(device), kg_word_embeddings.to(device))

    pos, angle, width = post_process_output(pos, cos, sin, width)
    pred_mask = np.array(pred_seg.detach().cpu().numpy() > 0.35)
    origin_size = 640 + 2*boader 
    pos_show = cv2.resize(pos[0], [origin_size, origin_size])
    angle_show = cv2.resize(angle[0],  [origin_size, origin_size])
    width_show = cv2.resize(width[0],  [origin_size, origin_size]) / 320 * origin_size
    seg_show = cv2.resize(pred_mask[0][0].astype(np.uint8), [origin_size, origin_size])
    seg_show = seg_show[boader+pad:-pad-boader, boader:-boader]

    refgrasp, grasps_center_angle_l = detect_grasps(pos_show, angle_show, width_show)
    
    if len(refgrasp) == 0:
        print("without the referred object!")
        return (0, 0), 0, 0
    
    center_x, center_y, theta, width = np.array(grasps_center_angle_l[0]) - np.array([boader, pad+boader, 0, 0])
    center = (center_x, center_y)
    refgrasp_show = [np.array(refgrasp[0]) - np.array([boader, pad+boader, 0, boader, pad+boader]).tolist()]


    ##############################################
    

    random_str_name = generate_random_str(5)
    name = text.replace(' ', "_")
    vis_name = f'{name}_{int(time.time())}'
    draw_hotmap(pos_show, name=f'detection_vis/{vis_name}_quality')
    fig = Image.fromarray(img_show)
    fig1 = copy.deepcopy(fig)
    draw1 = ImageDraw.Draw(fig1)
    if refgrasp_show is not None:
        if not isinstance(refgrasp_show, list):
            refgrasp_show = refgrasp_show.cpu().numpy()
        for grasp in refgrasp_show:
            [[x1_, y1_], [x2_, y2_], [x3_, y3_], [x4_, y4_]] = asGraspRectangle(grasp)
            draw1.line((x1_,y1_,x2_,y2_), fill='red', width=2)
            draw1.line((x2_,y2_,x3_,y3_), fill='blue', width=2)
            draw1.line((x3_,y3_,x4_,y4_), fill='red', width=2)
            draw1.line((x4_,y4_,x1_,y1_), fill='blue', width=2)

            fig1.save(f'detection_vis/{vis_name}_grasp.jpg')

    
    fig3 = copy.deepcopy(fig)
    if seg_show is not None:
        if not isinstance(seg_show, np.ndarray):
            seg_show = seg_show.cpu().numpy()
        fig3 = apply_mask(fig3, seg_show[np.newaxis, :])

        fig3.save(f'detection_vis/{vis_name}_mask.jpg')
    ###########################################
    return center, -theta, width

class inference():
    def __init__(self) -> None:
        args = parse_args()
        self.device = torch.device("cuda:0")
        checkpoint_path = 'CGnet_best_model_gpus.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.net = CGSnet(args).to(self.device).eval()
        self.net.load_state_dict(checkpoint['state_dict'])
    
    def infer(self, img_color, text):
        img_color = cv2.convertScaleAbs(img_color, alpha=0.8, beta=20)
        center, theta, width=real_inference(self.net, img_color, text, self.device)
        return center, theta, width

if __name__=="__main__":
    args = parse_args()
    device = torch.device("cuda:0")

    net = CGSnet(args).to(device).eval()
    checkpoint_path = 'CGnet_best_model_gpus.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])


    img_color = cv2.imread("input_image_D435.png")
    img_color = cv2.convertScaleAbs(img_color, alpha=0.8, beta=20)
    while True:
        text = input('please input the description:')


        center, theta, width=inference(net, img_color, text, device) 



