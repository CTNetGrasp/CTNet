'''
该数据处理函数主要负责图片数据增强和归一化,并将抓取位置、角度和抓取爪宽度用Gaussian map表示。

'''
from utils.grasp import asGraspRectangle, asGrasp
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Tuple
from pathlib import Path
import torch
import torchvision.transforms.functional as F
import os
from bisect import bisect_left
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
from transformers import RobertaTokenizer, BertTokenizer

import random
from skimage.draw import polygon, disk

import matplotlib.pyplot as plt
import seaborn as sns
import copy
import json
from loguru import logger

import time
import cv2
cv2.setNumThreads(0)

def apply_mask(image, mask, color=[0, 1, 0], alpha=0.4):
    """Apply the given mask to the image.
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
    

class RefGraspBase(torch.utils.data.Dataset): 
    def __init__(self, output_size=300, include_depth=False, include_rgb=True, use_mask=True, use_bbox=True, random_rotate=False, random_crop=False, random_bright=False, meanstd=True):
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_crop = random_crop
        self.random_bright = random_bright
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.use_mask = use_mask
        self.use_bbox = use_bbox
        self.meanstd = meanstd

        self.expression_files = []

        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')        

    def get_rgb(self, idx):
        raise NotImplementedError()

    def get_depth(self, idx):
        raise NotImplementedError()
    
    def get_refmask(self, idx):
        raise NotImplementedError()
    
    def get_refbbox(self, idx):
        raise NotImplementedError()
    
    def get_refgrasp(self, idx):
        raise NotImplementedError()
    
    def get_token(self, idx, maskimg=0):
        raise NotImplementedError()

    def normalize(self, rgb_img, depth, refgrasp, refbbox, refmask):
        raise NotImplementedError()
    
    def resize(self, rgb_img, depth_img, refgrasp, refbbox, refmask):
        if rgb_img is not None:
            W_ori, H_ori = rgb_img.size
        else:
            W_ori, H_ori = depth_img.size
        if rgb_img is not None:
            rgb_img = rgb_img.resize((self.output_size, self.output_size))
        if depth_img is not None:
            depth_img = depth_img.resize((self.output_size, self.output_size))
        if refmask is not None:
            refmask = refmask.resize((self.output_size, self.output_size))

        scale_H = self.output_size / (H_ori*1.0 )
        scale_W = self.output_size / (W_ori*1.0 )
        # print(scale_H, scale_W)
        if refbbox is not None:
            refbbox = np.array(refbbox) * np.array([scale_W, scale_H, scale_W, scale_H])

        refgrasp = np.array(refgrasp) * np.array([scale_W, scale_H, 1, scale_W, scale_H])

        return rgb_img, depth_img, refgrasp, refbbox, refmask
    
    def enhance(self, rgb_img, depth_img, refmask, refbbox, refgrasp):
        
        def _crop(rgb_img, depth_img, refmask, refbbox, refgrasp):
            if rgb_img is not None:
                W, H = rgb_img.size
            else:
                W, H = depth_img.size

            if refbbox is not None:
                min_x, min_y, max_x, max_y = refbbox
            else:
                rect = []
                for gr in refgrasp:
                    rect.append(asGraspRectangle(gr))
                rect = np.array(rect)
                rect_T = np.swapaxes(rect, 1, 2)
                min_x, min_y, max_x, max_y = rect_T[:,0,:].min(), rect_T[:,1,:].min(), rect_T[:,0,:].max(), rect_T[:,1,:].max()
            
            dx1, dy1, dx2, dy2 = random.sample(range(0, 50), k=4)

            x1_crop = dx1 if dx1 < min_x else min_x-1
            y1_crop = dy1 if dy1 < min_y else min_y-1
            x2_crop = W - 1 - dx2 if W - 1 - dx2 > max_x else max_x+1
            y2_crop = H - 1 - dy2 if H - 1 - dy2 > max_y else max_y+1
            # print((x1_crop, y1_crop, x2_crop, y2_crop))
            if rgb_img is not None:
                rgb_img = rgb_img.crop((x1_crop, y1_crop, x2_crop, y2_crop))
            if depth_img is not None:
                depth_img = depth_img.crop((x1_crop, y1_crop, x2_crop, y2_crop))
            if refmask is not None:
                refmask = refmask.crop((x1_crop, y1_crop, x2_crop, y2_crop))
            if refbbox is not None:
                refbbox = np.array(refbbox) - np.array([x1_crop, y1_crop, x1_crop, y1_crop])

            refgrasp = np.array(refgrasp) - np.array([x1_crop, y1_crop, 0 ,x1_crop, y1_crop])

            return rgb_img, depth_img, refmask, refbbox, refgrasp 

        def _bright_contrast_color(rgb_img):
            if rgb_img is None:
                return None
            img = rgb_img
            bright = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.9, 1.5)
            color = random.uniform(0.8, 1.5)

            random_seed = random.random()
            if random_seed > 0.5:
                brightEnhancer = ImageEnhance.Brightness(img)
                img = brightEnhancer.enhance(bright)

            random_seed = random.random()
            if random_seed > 0.5:    
                contrastEnhancer = ImageEnhance.Contrast(img)
                img = contrastEnhancer.enhance(contrast)
            random_seed = random.random()

            if random_seed > 0.5:
                colorEnhancer = ImageEnhance.Color(img)
                img = colorEnhancer.enhance(color)
            return img
        
        def _random_rotate(rgb_img, depth_img, refmask, refbbox, refgrasp):
            rgb_img_old, depth_img_old, refmask_old, refbbox_old, refgrasp_old = rgb_img, depth_img, refmask, refbbox, refgrasp
            max_angle = 15
            
            rdm_a = random.uniform(-max_angle, max_angle)
            if rgb_img is not None:
                W, H = rgb_img.size
                rgb_img = rgb_img.rotate(-rdm_a)
                
            if depth_img is not None:
                W, H = depth_img.size
                depth_img = depth_img.rotate(-rdm_a)

            if refmask is not None:
                refmask = refmask.rotate(-rdm_a)


            refgrasp = _rotGrasp(refgrasp, rdm_a, H, W)

            if refbbox is not None:
                refbbox = _mask2bbox(refmask)

            if refgrasp is False:
                return rgb_img_old, depth_img_old, refmask_old, refbbox_old, refgrasp_old

            return rgb_img, depth_img, refmask, refbbox, refgrasp
    
        def _mask2bbox(mask):
            mask_np = np.array(mask)
            H, W = mask_np.shape
            x_min, x_max, y_min, y_max = W-1, 0, H-1, 0
            
            for i in range(W):
                if 1 in mask_np[:, i]:
                    if x_min > i:
                        # print(i)
                        x_min = i
                    if x_max < i:
                        x_max = i
            for i in range(H):
                if 1 in mask_np[i, :]:
                    if y_min > i:
                        y_min = i
                    if y_max < i:
                        y_max = i
            # print("BBox:", x_min, y_min, x_max, y_max)
            return np.array([x_min, y_min, x_max, y_max])

        def _rotGrasp(refgrasp, rdm_a, img_height, img_width):
            H, W = img_height, img_width
            c_x_o, c_y_o = W /2, H/2  # 图片中心
            # print(rdm_a)
            b = np.deg2rad(rdm_a) 
            rects = [asGraspRectangle(grasp_box) for grasp_box in refgrasp]
            rect_news = []

            for r in rects: 
                rect_new = []
                for p in r:
                    c_x_new = (p[0] - c_x_o) * np.cos(b) - (p[1] - c_y_o) * np.sin(b) + c_x_o
                    c_y_new = (p[0] - c_x_o) * np.sin(b) + (p[1] - c_y_o) * np.cos(b) + c_y_o

                    rect_new.append([round(c_x_new), round(c_y_new)])

                rect_news.append(rect_new)

            # print(rect_news)
            if (np.array(rect_news) < 0).mean() > 0 \
                or (np.array(rect_news)[:,:,0] >= img_width).mean() > 0 \
                or (np.array(rect_news)[:,:,1] >= img_height).mean() > 0:

                return False

            refgrasp_new = asGrasp(np.array(rect_news))

            return np.array(refgrasp_new)
        
        if self.random_crop:
            rgb_img, depth_img, refmask, refbbox, refgrasp = _crop(rgb_img, depth_img, refmask, refbbox, refgrasp) 
        # print("crop:", refgrasp)
        if self.random_bright:
            rgb_img = _bright_contrast_color(rgb_img)
        if self.random_rotate:
            rgb_img, depth_img, refmask, refbbox, refgrasp = _random_rotate(rgb_img, depth_img, refmask, refbbox, refgrasp)
        # print("rotate:", refgrasp)
        refgrasp = np.array(refgrasp)

        return rgb_img, depth_img, refmask, refbbox, refgrasp  
    

    def get_refgrasp_img(self, refgrasps, shape=(480, 640)):
        pos_out = np.zeros(shape)
        ang_out = np.zeros(shape)
        width_out = np.zeros(shape) 
        # print(refgrasps)
        for i, gr in enumerate(refgrasps):
            gr_rec = asGraspRectangle(gr)  # [p1, p2, p3, p4]
            x1, y1, theta, x2, y2 = gr
            rr, cc = polygon(gr_rec[:,1], gr_rec[:,0], shape) 
            ang_out[rr, cc] = np.deg2rad(theta)
            length = abs(x2 - x1)
            width_out[rr, cc] = length
            pos_out[rr, cc] = 1

        pos_out = self.GaussianPos(refgrasps, shape) # gaussian representation
        return pos_out, ang_out, width_out 

    
    def GaussianPos(self, refgrasps, shape):
        def gaussian(x, y, x0, y0, r):
            g = np.exp(-((x-x0)**2 + (y-y0)**2)/(2*r**2))
            return g
        pos_out = np.zeros(shape)
        for i, gr in enumerate(refgrasps):
            x1, y1, theta, x2, y2 = gr
            cx, cy = round((x1 + x2) /2), round((y1 + y2) / 2 )
            radius = round(min(abs(x2-x1), abs(y2 - y1)) / 2)

            rr, cc = disk((cy, cx), radius, shape=shape)
            
            # g = gaussian(cx-10, cy-10, cx, cy, radius)
            # print(g)
            for i in range(len(rr)):
                g = gaussian(cc[i], rr[i], cx, cy, radius)
                if pos_out[rr[i], cc[i]] != 0:
                    pos_out[rr[i], cc[i]] = g if g > pos_out[rr[i], cc[i]] else pos_out[rr[i], cc[i]]
                else:
                    pos_out[rr[i], cc[i]] = gaussian(cc[i], rr[i], cx, cy, radius)

        return pos_out


    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
        
    def __getitem__(self, idx):
        # idx = 16
        # t0 = time.time()
        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx)
        else:
            depth_img = None
        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx)
        else:
            rgb_img = None
        if self.use_mask:
            refmask = self.get_refmask(idx)
            # print('use mask')
        else:
            refmask = None
        if self.use_bbox:
            refbbox = self.get_refbbox(idx)
            # print('use bbox')
        else:
            refbbox = None
        
        # Load the grasps
        refgrasp = self.get_refgrasp(idx)

        rgb_img_o, depth_img_o, refmask_o, refbbox_o, refgrasp_o = rgb_img, depth_img, refmask, refbbox, refgrasp

        rgb_img, depth_img, refmask, refbbox, refgrasp = self.enhance(rgb_img, depth_img, refmask, refbbox, refgrasp)


        if refgrasp.shape[0] == 0:
            rgb_img, depth_img, refmask, refbbox, refgrasp = rgb_img_o, depth_img_o, refmask_o, refbbox_o, refgrasp_o

        rgb_img, depth_img, refgrasp, refbbox, refmask = self.normalize(rgb_img, depth_img, refgrasp, refbbox, refmask)

        rgb_img, depth_img, refgrasp, refbbox, refmask = self.resize(rgb_img, depth_img, refgrasp, refbbox, refmask)

        word_embeddings, word_attention_mask, sentence_raw = self.get_token(idx)

        # self.drawGrasp(fig=rgb_img, graspboxes=refgrasp, name=f'grasp_vis_enhance/{idx}_{sentence_raw}')
        if self.include_rgb:
            rgb_img = np.array(rgb_img)
            rgb_img = rgb_img.transpose(2, 0, 1)

        # np.save('rgb_img.npy', rgb_img)
        # np.save('grasps.npy', np.array(refgrasp))
        if  self.include_depth:
            depth_img = np.array(depth_img)

        if refmask is not None:
            refmask = np.array(refmask)

        

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)

        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)
            if not isinstance(x, torch.FloatTensor):
                x = x.float()
            if self.meanstd:
                x = x.div_(255.).sub_(self.mean).div_(self.std)

        # print("total:",t4-t0,"get data:", t1-t0, "enhance:", t2-t1, "token:", t3-t2, 'totensor:', t4-t3)
        # generate maps of grasp

        # pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        
        pos_img, ang_img, width_img = self.get_refgrasp_img(refgrasp, shape=(rgb_img.shape[1], rgb_img.shape[2]))

        # np.save('pos_img.npy', pos_img)
        # np.save('ang_img.npy', ang_img)
        # np.save('width_img.npy', width_img)
        
        # plt.figure(dpi=300)
        # cmap = sns.heatmap(ang_img, yticklabels=False, xticklabels=False, cmap='jet', vmin=-np.pi/2, vmax=np.pi/2, cbar_kws={"ticks": [-np.pi/2, 0, np.pi/2]})
        # # cbar = cmap.collections[0].colorbar
        # # cbar.ax.ticklabels( ['-pi/2', 0, 'pi/2'])
        # plt.tight_layout()
        # plt.savefig('heatmapang.jpg')

        pos = self.numpy_to_torch(pos_img)   # 0-1

        cos = self.numpy_to_torch(np.cos(2*ang_img)) # 0, 1

        sin = self.numpy_to_torch(np.sin(2*ang_img)) # -1, 1

        if self.use_mask:
            refmask = self.numpy_to_torch(refmask)

        if self.use_bbox:
            refbbox = torch.from_numpy(refbbox)

        refgrasp = torch.from_numpy(refgrasp)
        

        width_img = np.clip(width_img, 0.0, 150.0)/150.0  # 0-1
        width = self.numpy_to_torch(width_img)
        return x, word_embeddings, word_attention_mask, pos, cos, sin, width, refgrasp, refbbox, refmask, idx
    
    def __len__(self):
        return len(self.expression_files)  # 缩减训练集数量
    





class RefOCIDGrasp(RefGraspBase):
    def __init__(self, data, split, max_tokens=20, tokenizer_name="bert", output_size=300, include_depth=False, include_rgb=True, use_mask=True, use_bbox=True, random_rotate=False, random_crop=False, random_bright=False, meanstd=True):
        super().__init__(output_size, include_depth, include_rgb, use_mask, use_bbox, random_rotate, random_crop, random_bright=random_bright, meanstd=meanstd)
        logger.info(f"loading {split} set......")
        self.data = data
        self.split = split
        self.max_tokens = max_tokens
        if split is not "train":
            self.random_bright=False
            self.random_crop = False
            self.random_rotate = False
        logger.info(f'Enhancement! random_rotate: {self.random_rotate}, self.random_crop:{self.random_crop}, self.random_bright:{self.random_bright}')

        with open(os.path.join(data, 'ref-OCID-Grasp', f"{split}_expressions.json"), "r") as f:
            self.examples = json.load(f)

        self.tokenizer_name = tokenizer_name

        if self.tokenizer_name == 'robert':
            self.tokenizer = RobertaTokenizer.from_pretrained('/data/users/xjl/modelZoo/roberta-base')
        elif self.tokenizer_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('/data/users/xjl/modelZoo/bert-base-uncased')
        elif self.tokenizer_name == 'clip':
            from models.tokenizer import tokenize
            self.tokenizer = tokenize
        else:
            print("tokenizer error!")
            exit()

        # 获取referring信息
        self.rgb_paths, self.sentences, self.bboxes, self.clutter_splits, self.ref_graspRectangles, self.class_names = [], [], [], [], [], []
        # Dataset is a bit big... let's cache!
        os.makedirs(os.path.join(self.data, "refer-cache"), exist_ok=True)
        cached_pt = os.path.join(self.data, "refer-cache", f"{self.split}-grasp-dataset.pt") 
        
        if os.path.exists(cached_pt):
            # self.rgb_paths, self.language, self.bboxes, self.ref_graspRectangles, self.clutter_splits, self.instanceIDScene = self.process_dataset()
            logger.info("loadding dataset from cache ...") 
            self.rgb_paths, self.sentences, self.bboxes, self.ref_graspRectangles, self.clutter_splits, self.instanceIDScene, self.class_names = torch.load(str(cached_pt))
            # print(len(self.rgb_paths), len(self.sentences), len(self.bboxes), len(self.ref_graspRectangles), len(self.clutter_splits), len(self.instanceIDScene))
            assert (
                {"train": 178647, "val": 12606, "test": 18804}.get(self.split, -1)
                == len(self.rgb_paths)
                == len(self.sentences)
                == len(self.bboxes)
                == len(self.ref_graspRectangles) # [[p0, p1, p2, p3], ...]   [[[x1, y1],[x2, y2],[x3,y3],[x4, y4]], ...]
                == len(self.clutter_splits)
                == len(self.examples)
                == len(self.instanceIDScene)
            ), "Error on load from cache!"
            logger.info("Done!") 
        else:
            self.rgb_paths, self.sentences, self.bboxes, self.ref_graspRectangles , self.clutter_splits, self.instanceIDScene, self.class_names = self._process_dataset()
            torch.save([self.rgb_paths, self.sentences, self.bboxes, self.ref_graspRectangles , self.clutter_splits, self.instanceIDScene, self.class_names], str(cached_pt))
        self.expression_files = list(zip(self.rgb_paths, self.sentences, self.bboxes, self.ref_graspRectangles, self.class_names))

    def get_rgb(self, idx):
        image_path = self.rgb_paths[idx]
        img = cv2.imread(image_path)
        # cv2.imwrite('norma.png', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # img = Image.open(image_path).convert("RGB")
        return img
    
    def get_depth(self, idx):
        depth_path = self.rgb_paths[idx].replace('rgb', 'depth')
        # depth = Image.open(depth_path) # I:int32位
        # depth = Image.fromarray(np.array(depth).astype('uint16')) # 转为16位
        depth = cv2.imread(depth_path, cv2.CV_16UC1)
        depth = Image.fromarray(depth)
        
        return depth
    
    def get_refmask(self, idx):

        mask_label_path = self.rgb_paths[idx].replace('rgb', 'label')

        np_mask = np.array(Image.open(mask_label_path).convert("L"))
        ref_mask = np.zeros(np_mask.shape)
        ref_mask[np_mask == self.instanceIDScene[idx].numpy()] = 1
        ref_mask = Image.fromarray(ref_mask.astype(np.uint8), mode="P")

        return ref_mask
    
    
    
    def get_refbbox(self, idx):
        ref_bbox = np.array(self.bboxes[idx])
        return ref_bbox

    def get_refgrasp(self, idx):
        ref_graspRectangle = self.ref_graspRectangles[idx]
        ##### 转成抓取格式 p0 p1 p2 p3 转为x1 y1 theta x2 y2 cls#########
        ref_graspbbox = asGrasp(ref_graspRectangle)

        return ref_graspbbox


    def get_token(self, idx, maskimg=0):
        if isinstance(idx, str):
            sentence_raw = idx
        else:
            sentence_raw = self.sentences[idx]
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens
        if self.tokenizer_name is "bert":
            input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
        elif self.tokenizer_name is 'clip':
            input_ids = self.tokenizer(sentence_raw, self.max_tokens, True).squeeze(0)
        input_ids = input_ids[:self.max_tokens]
        padded_input_ids[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)] = [1]*len(input_ids)  ### denote: 0 means this word is masked
        word_embeddings = torch.tensor(padded_input_ids) # [max_lenght]
        word_attention_mask = torch.tensor(attention_mask) # [max_lenght]
        if maskimg:
            pass

        return word_embeddings, word_attention_mask, sentence_raw  
    
    def get_class_name(self, idx):
        return self.class_names[idx]

    def drawGrasp(self, idx=0, fig=None, graspboxes=None, name='vis_refgrasp'):
        if fig is None or graspboxes is None:
            fig = self.get_rgb(idx)
            graspboxes = self.get_refgrasp(idx)
        else:
            graspboxes = list(graspboxes)
            if isinstance(fig, np.ndarray):             
                fig = Image.fromarray(fig.astype(np.uint8))
        fig = copy.deepcopy(fig)
        print("draw grasp")
        draw = ImageDraw.Draw(fig)
        if not isinstance(graspboxes, list):
            graspboxes = [graspboxes]
        for grasp in graspboxes:
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = asGraspRectangle(grasp)
            draw.line((x1,y1,x2,y2), fill='red', width=2)
            draw.line((x2,y2,x3,y3), fill='blue', width=2)
            draw.line((x3,y3,x4,y4), fill='red', width=2)
            draw.line((x4,y4,x1,y1), fill='blue', width=2)
        if name is not None:
            fig.save(name+'.png')
        return fig
        

    def drawBbox(self, idx=0, fig=None, bbox=None, name='vis_refBbox'):
        if fig is None or bbox is None:
            fig = self.get_rgb(idx)
            bbox = self.get_refbbox(idx)
        else:
            if isinstance(fig, np.ndarray):             
                fig = Image.fromarray(fig.astype(np.uint8))
        print("draw bbox")
        fig = copy.deepcopy(fig)
        draw = ImageDraw.Draw(fig)
        bx1, by1, bx2, by2 = bbox
        draw.rectangle([bx1, by1, bx2, by2], outline=(255, 0, 0))
        if name is not None:
            fig.save(name+'.png')
        return fig
    
    def drawMask(self, idx=0, fig=None, mask=None, name=None):
        print("draw mask")
        if fig is None or mask is None:
            fig = self.get_rgb(idx)
            mask = self.get_refmask(idx)
        else:
            if isinstance(fig, np.ndarray):             
                fig = Image.fromarray(fig.astype(np.uint8))

        # mask = np.expand_dims(np.array(mask), 2).repeat(3,axis=2)
        # mask =  Image.fromarray((mask * [0, 255, 0]).astype(np.uint8)).convert("RGB")
        # img_vis = Image.blend(fig, mask, alpha=0.3)
        img_vis = apply_mask(fig, mask)

        if name is not None:
            img_vis.save(name + '.png')
        return img_vis

    def _process_dataset(self):
        def get_split(take_id: int) -> int:
            assert 0 <= take_id <= 20, "Bounds violation - `take_id` must be in [0, 20]!"
            return bisect_left([9, 16, 20], take_id)

        rgb_paths, language, bboxes, clutter_splits, instanceIDScenes, ref_graspRectangles, class_names = [], [], [], [], [], [], []
        for _idx, key in tqdm(enumerate(self.examples), total=len(self.examples), leave=False):
            example = self.examples[key]

            # Parse Example => rgb_path :: str, lang :: str, bbox :: torch.Tensor, clutter split :: int
            rgb_path, lang = os.path.join(self.data , "OCID_grasp" , example["scene_path"]), example["sentence"]
            rgb_name = example["scene_path"].split('/')[-1]
            instanceIDScene = example["scene_instance_id"]
            clutter_split = get_split(example["take_id"])
            class_name = example['class']
            anno_path = os.path.join(self.data, "OCID_grasp" ,example["sequence_path"], 'Annotations', rgb_name[:-4] + '.txt')
            allgraspRectangle_array = self._readGraspfromTxt(anno_path)
            
            ref_graspRectangle = self._filter_graspRectangle_from_maskref(allgraspRectangle_array, rgb_path, instanceIDScene)

            bbox = torch.tensor(json.loads(example["bbox"]), dtype=torch.int)
            assert bbox[0] < bbox[2] <= 640 and bbox[1] < bbox[3] <= 480, "Invalid Bounding Box Size!"

            # Add to Trackers --> note that we pad/preprocess the RGB images *in __getitem__()*
            rgb_paths.append(rgb_path)
            language.append(lang)
            bboxes.append(bbox)
            clutter_splits.append(clutter_split)
            instanceIDScenes.append(instanceIDScene)
            ref_graspRectangles.append(ref_graspRectangle) # # [[p0, p1, p2, p3], ...]   [[[x1, y1],[x2, y2],[x3,y3],[x4, y4]], ...]
            class_names.append(class_name)
            # if _idx > 10:
            #     break
            
        # Return (as Tensors if applicable)
        print("Number:", len(rgb_paths), len(language), len(bboxes), len(ref_graspRectangles))
        j = 0
        for grasp in ref_graspRectangles:
            if len(grasp) == 0:
                j+=1
        print("zero :", j)
        return rgb_paths, language, torch.stack(bboxes), ref_graspRectangles, torch.tensor(clutter_splits, dtype=torch.int), torch.tensor(instanceIDScenes, dtype=torch.int), class_names
        
    def _readGraspfromTxt(self, path):  
        with open(path, "r") as f:
            points_list = []
            grasp_boxes_list = []
            for count, line in enumerate(f):
                line = line.rstrip()
                [x, y] = line.split(' ')

                x = float(x)
                y = float(y)

                pt = (x, y)
                points_list.append(pt)

                if len(points_list) == 4:
                    grasp_boxes_list.append(points_list)
                    points_list = []

        grasp_box_arry = np.asarray(grasp_boxes_list)
        return grasp_box_arry  # [[p0, p1, p2, p3], ...]   [[[x1, y1],[x2, y2],[x3,y3],[x4, y4]], ...]
    
    def _filter_graspRectangle_from_maskref(self, all_graspsRectangle, rgb_path, instanceIDScene):
        ref_graspsRectangle = []

        refmasks = rgb_path.replace('rgb', 'label')
        np_mask = np.array(Image.open(refmasks).convert("L"))
        ref_mask = np.zeros(np_mask.shape)
        ref_mask[np_mask == instanceIDScene] = 1
        for grasp in all_graspsRectangle:  #[rec1, rec2, ...]
            
            [x1, y1], [x2, y2], [x3, y3], [x4, y4] = grasp
            cx, cy = (x1 + x2 + x3 + x4) / 4.0, (y1 + y2 + y3 + y4) / 4.0
            if ref_mask[round(cy), round(cx)]:
                ref_graspsRectangle.append(np.array(grasp))
        if len(ref_graspsRectangle) == 0:
            print("Goals that don't exist:", rgb_path, instanceIDScene)
        return ref_graspsRectangle

    
    
    def _crop2square(self, rgb_img, depth, refgrasp, refbbox, refmask):
        if refbbox is not None:
            min_x, min_y, max_x, max_y = refbbox
        else:
            rect = []
            for gr in refgrasp:
                rect.append(asGraspRectangle(gr))
            rect = np.array(rect)
            rect_T = np.swapaxes(rect, 1, 2)
            min_x, min_y, max_x, max_y = rect_T[:,0,:].min(), rect_T[:,1,:].min(), rect_T[:,0,:].max(), rect_T[:,1,:].max()
        if rgb_img is not None:
            W, H = rgb_img.size
        else:
            W, H = depth.size
        min_size = min(W, H)

        if (W - min_size) / 2 > min_x or (H - min_size) /2 > min_y: # 点在左边
            if rgb_img is not None:
                rgb_img = rgb_img.crop((0, 0, min_size, min_size))
            if depth is not None:
                depth = depth.crop((0, 0, min_size, min_size))
            if refmask is not None:
                refmask = refmask.crop((0, 0, min_size, min_size))

        elif W - (W - min_size) /2 < max_x or H - (H - min_size) /2 < max_y: # 点在右边
            if rgb_img is not None:
                rgb_img = rgb_img.crop((W-min_size, H-min_size, W-1, H-1))
            if depth is not None:
                depth = depth.crop((W-min_size, H-min_size, W-1, H-1))
            refgrasp = np.array(refgrasp) - np.array([W-min_size, H-min_size, 0, W-min_size, H-min_size])
            
            if refbbox is not None:
                refbbox = np.array(refbbox) - np.array([W-min_size, H-min_size, W-min_size, H-min_size])
            
            if refmask is not None:
                refmask = refmask.crop((W-min_size, H-min_size, W-1, H-1))
        else: # 点在中间
            if rgb_img is not None:
                rgb_img = rgb_img.crop(((W - min_size) /2, (H-min_size)/2, W - (W - min_size) /2, H - (H - min_size) /2))
            if depth is not None:
                depth = depth.crop(((W - min_size) /2, (H-min_size)/2, W - (W - min_size) /2, H - (H - min_size) /2))
            # print(refgrasp)
            refgrasp = np.array(refgrasp) - np.array([(W - min_size) /2, (H-min_size)/2, 0, (W - min_size) /2, (H-min_size)/2])
            
            if refbbox is not None:
                refbbox = np.array(refbbox) - np.array([(W - min_size) /2, (H-min_size)/2, (W - min_size) /2, (H-min_size)/2])
            
            if refmask is not None:
                refmask = refmask.crop(((W - min_size) /2, (H-min_size)/2, W - (W - min_size) /2, H - (H - min_size) /2))
        
        return rgb_img, depth, refgrasp, refbbox, refmask

    def normalize(self, rgb_img, depth, refgrasp, refbbox, refmask):
        
        rgb_img, depth, refgrasp, refbbox, refmask = self._crop2square(rgb_img, depth, refgrasp, refbbox, refmask)
        
        return rgb_img, depth, refgrasp, refbbox, refmask

    def collate(self, data):
        x, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, idx = zip(*data)

        x = torch.stack(x, dim=0)

        pos_img = torch.stack(pos_img, dim=0)
        cos_img = torch.stack(cos_img, dim=0)
        sin_img = torch.stack(sin_img, dim=0)
        width_img = torch.stack(width_img, dim=0)

        refgrasps = refgrasps
        
        if None not in refbbox:
            refbbox = torch.stack(refbbox, dim=0)
        else:
            refbbox = None
        
        if None not in refmask:
            refmask = torch.stack(refmask, dim=0)
        else:
            refmask = None
        
        word_embeddings = torch.stack(word_embeddings, dim=0)
        word_attention_mask = torch.stack(word_attention_mask, dim=0)

        return x, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, idx



if __name__=="__main__":
    import json
    data='./data'
    train_dataset = RefOCIDGrasp('./data', "train", tokenizer_name='clip', include_rgb=True, include_depth=False, use_bbox=True, use_mask=True, output_size=480, max_tokens=15, random_rotate=True, random_bright=True, random_crop=True)
    # train_dataset = RefOCIDGrasp(data, "train", use_bbox=True, use_mask=True, output_size=480, include_depth=False, meanstd=False)
    # train_dataset.drawGrasp(0)
    # train_dataset.drawBbox(0)
    # train_dataset.drawMask(0)
    # print(train_dataset.get_token(0)[-1])
    # print(train_dataset.expression_files[0])
    # print(len(train_dataset))
    import time
    # dataLoader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=train_dataset.collate)
    # for data in dataLoader:
    #     # print(data[0].shape, data[1].shape, data[2].shape, data[3].shape, data[4].shape, data[6].shape)
    #     pass
    #     # time.sleep(1)
    #     # break

    
    for data in train_dataset:
        img, word_embeddings, word_attention_mask, pos_img, cos_img, sin_img, width_img, refgrasps, refbbox, refmask, idx = data
        train_dataset.drawGrasp(fig=data[0].numpy().transpose(1, 2, 0), graspboxes=refgrasps.numpy(), name='normal_grasp')
        train_dataset.drawBbox(fig=data[0].numpy().transpose(1, 2, 0), bbox=data[3], name='normal_bbox')
        train_dataset.drawMask(fig=data[0].numpy().transpose(1,2,0), mask=data[4][0], name='normal_mask')
        rects = train_dataset.ref_graspRectangles[idx]
        img = data[0].numpy().transpose(1, 2, 0)
        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
        for i, r in enumerate(rects[0]):
            cv2.circle(img, (int(r[0]), int(r[1])), 2, color[i], 4)
        cv2.imwrite('ciclr.png', img)
        pos, cos, sin, width = data[3:7]
        plt.figure(dpi=300)
        sns.heatmap(pos[0], yticklabels=False, xticklabels=False, cmap='jet', vmin=0, vmax=1)
        
        plt.tight_layout()
        plt.savefig('heatmappos.jpg')

    #     plt.figure(dpi=300)
    #     sns.heatmap(cos[0], yticklabels=False, xticklabels=False, cmap='jet', vmin=-1, vmax=1)
    #     plt.tight_layout()
    #     plt.savefig('heatmapcos.jpg')

    #     plt.figure(dpi=300)
    #     sns.heatmap(sin[0], yticklabels=False, xticklabels=False, cmap='jet', vmin=-1, vmax=1)
    #     plt.tight_layout()
    #     plt.savefig('heatmapsin.jpg')

    #     plt.figure(dpi=300)
    #     sns.heatmap(width[0], yticklabels=False, xticklabels=False, cmap='jet', vmin=0, vmax=1)
    #     plt.tight_layout()
    #     plt.savefig('heatmapwidth.jpg')
        time.sleep(5)
    
    
