import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SAMMI

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.use_1x1conv = False
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.use_1x1conv = True

        self.LeakyReLU1 = nn.LeakyReLU(inplace=True)

        self.LeakyReLU2 = nn.LeakyReLU(inplace=True)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = self.LeakyReLU1(x)
        x = self.bn2(self.conv2(x))
        if self.use_1x1conv:
            x_in = self.conv1x1(x_in)
        return self.LeakyReLU2(x + x_in)
    
class segmentation(nn.Module):
    def __init__(self, feats_dims=[2048, 1024, 512, 256], word_dim=512, hidden_dim=128):
        super().__init__()
        hidden_size = hidden_dim
        c4_size = feats_dims[0]
        c3_size = feats_dims[1]
        c2_size = feats_dims[2]
        c1_size = feats_dims[3]
        self.res1 = ResidualBlock(c4_size+c3_size, hidden_size)

        self.SAMMI1 = SAMMI(img_dim=hidden_size, word_dim=word_dim)


        self.res2 = ResidualBlock(hidden_size + c2_size, hidden_size)

        self.SAMMI2 = SAMMI(img_dim=hidden_size, word_dim=word_dim)


        self.res3 = ResidualBlock(hidden_size + c1_size, hidden_size)

        self.SAMMI3 = SAMMI(img_dim=hidden_size, word_dim=word_dim)


        self.seg_conv1 = nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.seg_bn1 = nn.InstanceNorm2d(hidden_dim//2, affine=True)
        self.seg_LeakyReLU1 = nn.LeakyReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.1)

        self.seg_head =  nn.Conv2d(hidden_dim//2, 1, kernel_size=1)

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

        features = []
        
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)

        x = torch.cat([x_c4, x_c3], dim=1)

        x = self.res1(x)
        x = self.SAMMI1(x, word)
        features.append(x)
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)

        x = self.res2(x)
        x = self.SAMMI2(x, word)
        features.append(x)
        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)

        x = self.res3(x)
        x = self.SAMMI3(x, word)
        features.append(x)

        x = self.seg_conv1(x)
        x = self.seg_bn1(x)
        x = self.seg_LeakyReLU1(x)
        x = self.dropout1(x)
        mask_pre = self.seg_head(x)

        mask_pre = F.interpolate(input=mask_pre, size=(x.size(-2)*4, x.size(-1)*4), mode='bilinear', align_corners=True)
        return features, mask_pre
    
class grasp(nn.Module):
    def __init__(self, enc_dims, seg_dim, word_dim, hidden_dim, img_size):
        super().__init__()
        self.img_size = img_size

        self.conv1x1 = nn.Conv2d(enc_dims+3*seg_dim, 2*hidden_dim, kernel_size=1)

        self.conv3x3 = nn.Conv2d(2*hidden_dim, hidden_dim, 3, padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(hidden_dim, affine=True)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.SAMMI = SAMMI(img_dim=hidden_dim, word_dim=word_dim)

        self.dropout1 = nn.Dropout(p=0.1)
  

        self.conv3x3_2 = nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1, bias=False)
        self.bn_2 = nn.InstanceNorm2d(hidden_dim//2, affine=True)
        self.LeakyReLU_2 = nn.LeakyReLU(inplace=True)

        self.dropout2 = nn.Dropout(p=0.1)


        self.pos_output =  nn.Conv2d(hidden_dim//2, 1, kernel_size=1)
        self.cos_output =  nn.Conv2d(hidden_dim//2, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(hidden_dim//2, 1, kernel_size=1)
        self.width_output = nn.Conv2d(hidden_dim//2, 1, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, enc_fs, seg_fs, word):

        ef = F.interpolate(input=enc_fs, size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)

        sf1 = F.interpolate(input=seg_fs[0], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)
        sf2 = F.interpolate(input=seg_fs[1], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)
        sf3 = F.interpolate(input=seg_fs[2], size=(self.img_size//4, self.img_size//4), mode='bilinear', align_corners=True)
        seg_f = self.conv1x1(torch.cat([ef, sf1, sf2, sf3], dim=1))


        fs = self.conv3x3(seg_f)
        fs = self.bn(fs)
        fs = self.LeakyReLU(fs)
        fs = self.SAMMI(fs, word)
        fs = F.interpolate(input=fs, size=(self.img_size//2, self.img_size//2), mode='bilinear', align_corners=True) # 160 160
        fs = self.dropout1(fs)

        fs = self.conv3x3_2(fs)
        fs = self.bn_2(fs)
        fs = self.LeakyReLU_2(fs)
        fs = F.interpolate(input=fs, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True) # 320 320
        fs = self.dropout2(fs)

        pos = self.pos_output(fs)
        cos = self.cos_output(fs)
        sin = self.sin_output(fs)
        width = self.width_output(fs)
        
        pos = torch.sigmoid(pos)
        width = torch.sigmoid(width)

        return pos, cos, sin, width



class dualDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.seg = segmentation(args.feats_dims, args.word_dim, hidden_dim=args.seg_dim)
        self.grasp_head = grasp(args.feats_dims[0], args.seg_dim, args.word_dim, args.grasp_dim, args.img_size)
        self.img_size = args.img_size
    
    def forward(self, features, word):
        seg_feats,  seg_mask= self.seg(features, word)
        pos, cos, sin, width = self.grasp_head(features[0], seg_feats, word)

        return seg_mask, pos, cos, sin, width


