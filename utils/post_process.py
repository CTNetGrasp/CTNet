import torch
from skimage.filters import gaussian
import numpy as np

def post_process_seg(preds):
    preds = torch.sigmoid(preds)
    preds_out = []
    for i, pred in enumerate(preds):
        pred = pred.detach().cpu().numpy()
        pred = np.array(pred > 0.35)
        preds_out.append(pred)
    return preds_out


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.detach().cpu().numpy().squeeze(1)
    cos_img = cos_img.detach().cpu().numpy().squeeze(1)
    sin_img = sin_img.detach().cpu().numpy().squeeze(1)
    width_img = width_img.detach().cpu().numpy().squeeze(1)

    q_img_list = []
    ang_img_list = []
    width_img_list = []
    
    for i in range(q_img.shape[0]):
        q_i = q_img[i]
        ang_i= (np.arctan2(sin_img[i], cos_img[i]) / 2.0)
        width_i = width_img[i] * 150.0

        q_img_list.append(gaussian(q_i, 1.0, preserve_range=True))
        ang_img_list.append(gaussian(ang_i, 0.5, preserve_range=True))
        width_img_list.append(gaussian(width_i, 0.5, preserve_range=True))

        # q_img_list.append(q_i)
        # ang_img_list.append(ang_i)
        # width_img_list.append(width_i)

    return np.array(q_img_list), np.array(ang_img_list), np.array(width_img_list)