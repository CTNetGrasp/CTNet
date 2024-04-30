import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip import build_model
# from .single_decoder import singleDecoder
from .dual_decoder3 import dualDecoder

class CTNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(args.clip_pretrain, map_location="cpu").eval()
        self.encoder = build_model(clip_model.state_dict(), txt_length=args.txt_length, fus=False).float()
        self.decoder = dualDecoder(args)


    def forward(self, images, words, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        if mask is None:
            mask = torch.zeros_like(words).masked_fill_(words == 0, 1).bool()
        image_features, word_features, logits_per_image, sent = self.encoder(images, words)

        seg_mask, pos, cos, sin, width = self.decoder(image_features[::-1], word_features)
        
        return seg_mask, pos, cos, sin, width
    
