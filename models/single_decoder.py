import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import WAOP

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        # self.bn1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        # self.bn2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_1x1conv = False
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.use_1x1conv = True

        self.LeakyReLU1 = nn.ReLU(inplace=True)

        self.LeakyReLU2 = nn.ReLU(inplace=True)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = self.LeakyReLU1(x)
        x = self.bn2(self.conv2(x))
        if self.use_1x1conv:
            x_in = self.conv1x1(x_in)
        return self.LeakyReLU2(x + x_in)
    
class FIF(nn.Module):
    def __init__(self, feats_dims=[1024, 1024, 512, 256], word_dim=512, hidden_dim=128):
        super().__init__()
        hidden_size = hidden_dim
        c4_size = feats_dims[0]
        c3_size = feats_dims[1]
        c2_size = feats_dims[2]
        c1_size = feats_dims[3]
        # self.res1 = ResidualBlock(c4_size+c3_size, hidden_size)
        self.CNL1 = nn.Sequential(
            nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1),
            # nn.InstanceNorm2d(hidden_size, affine=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True)
        )

        self.WAOP1 = WAOP(img_dim=hidden_size, word_dim=word_dim)
        self.CNL2 = nn.Sequential(
            nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1),
            # nn.InstanceNorm2d(hidden_size, affine=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True)
        )

        # self.res2 = ResidualBlock(hidden_size + c2_size, hidden_size)
        

        self.WAOP2 = WAOP(img_dim=hidden_size, word_dim=word_dim)


        # self.res3 = ResidualBlock(hidden_size + c1_size, hidden_size)
        self.CNL3 = nn.Sequential(
            nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1),
            # nn.InstanceNorm2d(hidden_size, affine=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.WAOP3 = WAOP(img_dim=hidden_size, word_dim=word_dim)

        self.CNL4 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size//2, 3, padding=1),
            # nn.InstanceNorm2d(hidden_size, affine=True),
            nn.BatchNorm2d(hidden_size//2),
            nn.ReLU(inplace=True)
        )


        self.CNL5 = nn.Sequential(
            nn.Conv2d(hidden_size//2, hidden_size//2, 3, padding=1),
            nn.InstanceNorm2d(hidden_size//2, affine=True),
            nn.ReLU(inplace=True)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, img_feats, word):
        '''
            img_feats: [B, C1, H/8, W/8], [B, C2, H/16, W/16], [B, C3, H/32, W/32]
            word_feats: [B, C, N]
            sent_feats: [B, C]

        '''
        x_c4, x_c3, x_c2, x_c1 = img_feats
        
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)

        x = torch.cat([x_c4, x_c3], dim=1)

        # x = self.res1(x)
        x = self.CNL1(x)
        x = self.WAOP1(x, word)

        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)

        # x = self.res2(x)
        x = self.CNL2(x)
        x = self.WAOP2(x, word)

        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)

        x = self.CNL3(x)
        # x = self.res3(x)
        x = self.WAOP3(x, word)

        x = F.interpolate(input=x, size=(2*x.size(-2), 2*x.size(-1)), mode='bilinear', align_corners=True)
        # x = self.CNL1(x)
        x = self.CNL4(x)

        x = F.interpolate(input=x, size=(2*x.size(-2), 2*x.size(-1)), mode='bilinear', align_corners=True)
        # x = self.CNL2(x)
        x = self.CNL5(x)
    
        return x
    
        


class singleDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.seg = FIF(args.feats_dims, args.word_dim, hidden_dim=args.grasp_dim)
        self.seg_head =  nn.Conv2d(args.grasp_dim//2, 1, kernel_size=1)
        self.pos_output =  nn.Conv2d(args.grasp_dim//2, 1, kernel_size=1)
        self.cos_output =  nn.Conv2d(args.grasp_dim//2, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(args.grasp_dim//2, 1, kernel_size=1)
        self.width_output = nn.Conv2d(args.grasp_dim//2, 1, kernel_size=1)

        self.img_size = args.img_size
    
    def forward(self, features, word):
        feat = self.seg(features, word)

        seg_mask = self.seg_head(feat)
        pos = self.pos_output(feat)
        cos = self.cos_output(feat)
        sin = self.sin_output(feat)
        width = self.width_output(feat)
        
        pos = torch.sigmoid(pos)
        width = torch.sigmoid(width)
        return seg_mask, pos, cos, sin, width


