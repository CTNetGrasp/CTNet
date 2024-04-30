import torch
import torch.nn.functional as F

def grasp_loss(pred, truth):

    assert isinstance(pred, tuple) or len(pred) == 4
    pos_pred, cos_pred, sin_pred, width_pred = pred
    pos_gt, cos_gt, sin_gt, width_gt = truth

    p_loss = F.smooth_l1_loss(pos_pred, pos_gt)
    cos_loss = F.smooth_l1_loss(cos_pred, cos_gt)
    sin_loss = F.smooth_l1_loss(sin_pred, sin_gt)
    
    width_loss = F.smooth_l1_loss(width_pred, width_gt)


    loss = p_loss + 3*(cos_loss + sin_loss) + 2*width_loss
    
    return loss

def seg_loss(pred, gt):
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(pred, gt.shape[-2:],
                                     mode='nearest').detach()
    loss = F.binary_cross_entropy_with_logits(pred, gt)
    return loss

def compute_loss(seg_pred, seg_gt, grasp_pred, grasp_gt):
    weight = 2.0
    if seg_pred is None:
        s_loss = 0
        weight = 1
    else:
        s_loss = seg_loss(seg_pred, seg_gt)

    g_loss = grasp_loss(grasp_pred, grasp_gt)

    return s_loss + weight*g_loss