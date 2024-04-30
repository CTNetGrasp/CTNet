import torch
import torch.nn as nn


class WAOP(nn.Module):
    def __init__(self, img_dim, word_dim, hidden_dim=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = img_dim

        self.hidden_dim = hidden_dim
        self.word_dim = word_dim

        self.Linear_img_q1 = nn.Linear(img_dim, hidden_dim)
        self.linear_word_k1 = nn.Linear(word_dim, hidden_dim)
        self.linear_word_v1 = nn.Linear(word_dim, hidden_dim)

        self.Linear_img_q2 = nn.Linear(img_dim, hidden_dim)
        self.linear_word_k2 = nn.Linear(word_dim, hidden_dim)
        self.linear_word_v2 = nn.Linear(word_dim, hidden_dim)


        self.poolingW = nn.AdaptiveAvgPool2d((None, 1))
        self.poolingH = nn.AdaptiveAvgPool2d((1, None))

        self.Linear_img_q3 = nn.Linear(img_dim, hidden_dim)
        self.linear_word_k3 = nn.Linear(word_dim, hidden_dim)
        self.linear_word_v3 = nn.Linear(word_dim, hidden_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img, word, word_mask=None):
        '''
        img_vec: [B, C1, H, W]
        word: [B, N, C2]
        word_mask: [B, N]  # 0 represent mask
        '''
        img_residual = img

        B, C1, H, W = img.shape



        img_q_h = self.poolingW(img).squeeze(-1).permute(0, 2, 1) 
        img_q_h = self.Linear_img_q1(img_q_h) # [B, H, C2]

        word_k_h = self.linear_word_k1(word)
        word_v_h = self.linear_word_v1(word)


        img_attn_h = torch.matmul(img_q_h, word_k_h.transpose(1,2)) # [B, H, C2] * [B ,c2, N] *  = [B, H, N]

        img_attn_h = (self.hidden_dim ** -.5) * img_attn_h

        if word_mask is not None:
            word_mask = word_mask.unsqueeze(1)
            word_mask = word_mask.expand(-1, H, -1) # [B, H, N]
            img_attn_h = img_attn_h.masked_fill(word_mask == 0, -1e9)
        
        img_attn_h = torch.softmax(img_attn_h, dim=-1) # [B, N, H]

        img_H = torch.matmul(img_attn_h, word_v_h)   # [B, H, N] * [B, N, C2] = [B, H, c2]
        img_H = img_H.permute(0, 2, 1)

        img_H = img_residual + img_H.unsqueeze(-1).expand(-1, -1, -1, W)  # [B, C2, H, W]



    
        img_q_w = self.poolingH(img_H).squeeze(-2).permute(0, 2, 1)
        img_q_w = self.Linear_img_q2(img_q_w)  # [B, W, C2]

        word_k_w = self.linear_word_k2(word)
        word_v_w = self.linear_word_v2(word)


        img_attn_w = torch.matmul(img_q_w, word_k_w.transpose(1,2)) # [B, H, C2] * [B ,c2, N] *  = [B, H, N]

        img_attn_w = (self.hidden_dim ** -.5) * img_attn_w

        if word_mask is not None:
            img_attn_w = img_attn_w.masked_fill(word_mask == 0, -1e9)
        
        img_attn_w = torch.softmax(img_attn_w, dim=-1) # [B, N, H]

        img_W = torch.matmul(img_attn_w, word_v_w)   # [B, H, N] * [B, N, C2] = [B, H, c2]
        img_W = img_W.permute(0, 2, 1)
        img_W = img_residual + img_W.unsqueeze(-2).expand(-1, -1, H, -1)  # [B, C2, H, W]


        img_q_hw = img_W.view(B, -1, H* W).permute(0, 2, 1)

        img_q_hw = self.Linear_img_q3(img_q_hw) # [B, HW, C2]

        word_k_hw = self.linear_word_k3(word)
        word_v_hw = self.linear_word_v3(word)


        img_attn_hw = torch.matmul(img_q_hw, word_k_hw.permute(0, 2, 1)) # [B, HW, C2] * [B ,c2, N] *  = [B, HW, N]

        img_attn_hw = (self.hidden_dim ** -.5) * img_attn_hw

        if word_mask is not None:
            img_attn_hw = img_attn_hw.masked_fill(word_mask == 0, -1e9)
        
        img_attn_hw = torch.softmax(img_attn_hw, dim=-1) # [B, HW, N]

        img_HW = torch.matmul(img_attn_hw, word_v_hw)   # [B, HW, N] * [B, N, C2] = [B, HW, c2]

        img_HW = img_residual + img_HW.permute(0, 2, 1).view(B, -1, H, W)  # [B, C2, H, W]


        return img_HW
    

class SAMMI(nn.Module): # scale-aware multi-modal interaction
    def __init__(self, img_dim, word_dim, hidden_dim=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = img_dim

        self.hidden_dim = hidden_dim
        self.word_dim = word_dim

        self.Linear_img_q1 = nn.Linear(img_dim, hidden_dim)
        self.linear_word_k1 = nn.Linear(word_dim, hidden_dim)
        self.linear_word_v1 = nn.Linear(word_dim, hidden_dim)

        self.IN1 = nn.InstanceNorm2d(hidden_dim, affine=True)

        self.Linear_img_q2 = nn.Linear(img_dim, hidden_dim)
        self.linear_word_k2 = nn.Linear(word_dim, hidden_dim)
        self.linear_word_v2 = nn.Linear(word_dim, hidden_dim)
        self.IN2 = nn.InstanceNorm2d(hidden_dim, affine=True)


        self.poolingW = nn.AdaptiveAvgPool2d((None, 1))
        self.poolingH = nn.AdaptiveAvgPool2d((1, None))

        self.Linear_img_q3 = nn.Linear(img_dim, hidden_dim)
        self.linear_word_k3 = nn.Linear(word_dim, hidden_dim)
        self.linear_word_v3 = nn.Linear(word_dim, hidden_dim)
        self.IN3 = nn.InstanceNorm2d(hidden_dim, affine=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img, word, word_mask=None):
        '''
        img_vec: [B, C1, H, W]
        word: [B, N, C2]
        word_mask: [B, N]  # 0 represent mask
        '''
        img_residual = img

        B, C1, H, W = img.shape



        img_q_h = self.poolingW(img).squeeze(-1).permute(0, 2, 1) 
        img_q_h = self.Linear_img_q1(img_q_h) # [B, H, C2]

        word_k_h = self.linear_word_k1(word)
        word_v_h = self.linear_word_v1(word)


        img_attn_h = torch.matmul(img_q_h, word_k_h.transpose(1,2)) # [B, H, C2] * [B ,c2, N] *  = [B, H, N]

        img_attn_h = (self.hidden_dim ** -.5) * img_attn_h

        if word_mask is not None:
            word_mask = word_mask.unsqueeze(1)
            word_mask = word_mask.expand(-1, H, -1) # [B, H, N]
            img_attn_h = img_attn_h.masked_fill(word_mask == 0, -1e9)
        
        img_attn_h = torch.softmax(img_attn_h, dim=-1) # [B, N, H]

        img_H = torch.matmul(img_attn_h, word_v_h)   # [B, H, N] * [B, N, C2] = [B, H, c2]
        img_H = img_H.permute(0, 2, 1)
        img_H_exp = img_H.unsqueeze(-1).expand(-1, -1, -1, W)
        img_H_residual = img_H_exp


        img_H = img_residual + img_H_exp  # [B, C2, H, W]
        img_H = self.IN1(img_H)

        img_q_w = self.poolingH(img_H).squeeze(-2).permute(0, 2, 1)
        img_q_w = self.Linear_img_q2(img_q_w)  # [B, W, C2]

        word_k_w = self.linear_word_k2(word)
        word_v_w = self.linear_word_v2(word)


        img_attn_w = torch.matmul(img_q_w, word_k_w.transpose(1,2)) # [B, H, C2] * [B ,c2, N] *  = [B, H, N]

        img_attn_w = (self.hidden_dim ** -.5) * img_attn_w

        if word_mask is not None:
            img_attn_w = img_attn_w.masked_fill(word_mask == 0, -1e9)
        
        img_attn_w = torch.softmax(img_attn_w, dim=-1) # [B, N, H]

        img_W = torch.matmul(img_attn_w, word_v_w)   # [B, H, N] * [B, N, C2] = [B, H, c2]
        img_W = img_W.permute(0, 2, 1)
        img_W_exp = img_W.unsqueeze(-2).expand(-1, -1, H, -1)
        img_W_residule = img_W_exp

        img_W = img_residual + img_H_residual + img_W_exp  # [B, C2, H, W]
        img_W = self.IN2(img_W)


        img_q_hw = img_W.view(B, -1, H* W).permute(0, 2, 1)

        img_q_hw = self.Linear_img_q3(img_q_hw) # [B, HW, C2]

        word_k_hw = self.linear_word_k3(word)
        word_v_hw = self.linear_word_v3(word)


        img_attn_hw = torch.matmul(img_q_hw, word_k_hw.permute(0, 2, 1)) # [B, HW, C2] * [B ,c2, N] *  = [B, HW, N]

        img_attn_hw = (self.hidden_dim ** -.5) * img_attn_hw

        if word_mask is not None:
            img_attn_hw = img_attn_hw.masked_fill(word_mask == 0, -1e9)
        
        img_attn_hw = torch.softmax(img_attn_hw, dim=-1) # [B, HW, N]

        img_HW = torch.matmul(img_attn_hw, word_v_hw)   # [B, HW, N] * [B, N, C2] = [B, HW, c2]
        img_HW_exp = img_HW.permute(0, 2, 1).view(B, -1, H, W)

        img_HW = img_residual +  img_W_residule + img_H_residual + img_HW_exp # [B, C2, H, W]
        img_HW = self.IN3(img_HW)


        return img_HW