import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DynamicFilter(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, groups=None):
        super(DynamicFilter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.groups = groups if groups else in_channels

        self.conv = nn.Conv2d(in_channels, self.groups * kernel_size**2, kernel_size=1, bias=False)
        self.conv_gate = nn.Conv2d(self.groups * kernel_size**2, self.groups * kernel_size**2, kernel_size=1, bias=False)
        self.act_gate = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(self.groups * kernel_size**2)
        self.act = nn.Softmax(dim=-2)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        n, c, h, w = x.shape
        groups = min(self.groups, c)

        dynamic_filter = self.ap(x)
        dynamic_filter = self.conv(dynamic_filter)
        dynamic_filter = dynamic_filter * self.act_gate(self.conv_gate(dynamic_filter))
        dynamic_filter = self.bn(dynamic_filter)
        dynamic_filter = self.act(dynamic_filter)

        x_unfold = F.unfold(self.pad(x), kernel_size=self.kernel_size)
        x_unfold = x_unfold.view(n, groups, c//groups, self.kernel_size**2, h*w)
        dynamic_filter = dynamic_filter.view(n, groups, 1, self.kernel_size**2, 1)
        output = torch.sum(x_unfold * dynamic_filter, dim=3).view(n, c, h, w)
        return output

class WSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WSPBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CrossAttention(nn.Module):
    def __init__(self, q_channels, kv_channels, embed_dim):
        super().__init__()
        self.q1_proj = nn.Conv2d(q_channels, embed_dim, 1)
        self.q2_proj = nn.Conv2d(q_channels, embed_dim, 1)
        
        self.k_proj = nn.Conv2d(kv_channels, embed_dim, 1)
        self.v_proj = nn.Conv2d(kv_channels, embed_dim, 1)
        
        self.out_proj = nn.Conv2d(embed_dim*2, embed_dim, 1)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, q1, q2, kv):
        if kv.size()[-2:] != q1.size()[-2:]:
            kv = F.interpolate(kv, size=q1.shape[-2:], mode='bilinear', align_corners=False)
        
        Q1 = self.q1_proj(q1).permute(0, 2, 3, 1).flatten(1, 2)  
        Q2 = self.q2_proj(q2).permute(0, 2, 3, 1).flatten(1, 2)
        K = self.k_proj(kv).permute(0, 2, 3, 1).flatten(1, 2)
        V = self.v_proj(kv).permute(0, 2, 3, 1).flatten(1, 2)

        attn1 = F.softmax(torch.bmm(Q1, K.transpose(1,2)) / (self.embed_dim**0.5), dim=-1)
        out1 = torch.bmm(attn1, V)
        attn2 = F.softmax(torch.bmm(Q2, K.transpose(1,2)) / (self.embed_dim**0.5), dim=-1)
        out2 = torch.bmm(attn2, V)
        
        combined = torch.cat([out1, out2], dim=-1).permute(0, 2, 1)
        combined = combined.view(combined.size(0), self.embed_dim*2, q1.size(2), q1.size(3))
        output = self.out_proj(combined)
        return self.layer_norm(output.permute(0,2,3,1)).permute(0,3,1,2)

class DynamicFilterWithImageInput(nn.Module):
    def __init__(self, in_channels, img_channels=3, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.img_feature = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)))
        
        self.conv_filter = nn.Conv2d(64, in_channels * kernel_size**2, 1)
        self.bn = nn.BatchNorm2d(in_channels * kernel_size**2)
        self.act = nn.Softmax(dim=1)
        self.pad = nn.ReflectionPad2d(kernel_size//2)

    def forward(self, x_feat, raw_img):
        if raw_img.size(1) == 1: 
            raw_img = raw_img.repeat(1, 3, 1, 1) 
        elif raw_img.size(1) != 3: 
            raise ValueError(f"Expected input to have 1 or 3 channels, but got {raw_img.size(1)} channels")
        batch_size = x_feat.size(0)
        
        img_feat = self.img_feature(raw_img)
        dynamic_filter = self.conv_filter(img_feat)
        dynamic_filter = self.bn(dynamic_filter).view(batch_size, self.in_channels, self.kernel_size**2)
        dynamic_filter = self.act(dynamic_filter)

        x_padded = self.pad(x_feat)
        x_unfold = F.unfold(x_padded, self.kernel_size).view(batch_size, self.in_channels, self.kernel_size**2, -1)
        output = torch.einsum('nckl,nck->ncl', x_unfold, dynamic_filter).view_as(x_feat)
        return output

class IQANet(nn.Module):
    def __init__(self):
        super(IQANet, self).__init__()
        self.resnet101 = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-2])
        self.resnet18 = self._get_resnet18()
        self.resnet18_single = self._adapt_resnet18_for_single_channel()
        
        self.wsp_t = WSPBlock(512, 512)
        self.wsp_b = WSPBlock(512, 512)
        self.wsp_r = WSPBlock(512, 512)
        self.wsp_i = WSPBlock(512, 512)
        
        self.dynamic_filter_rgb = DynamicFilter(3)
        self.dynamic_filter_R = DynamicFilterWithImageInput(2048, img_channels=3)
        self.dynamic_filter_T = DynamicFilterWithImageInput(2048, img_channels=3)
        
        self.ca_tb = CrossAttention(q_channels=512, kv_channels=6144, embed_dim=256)
        self.ca_ri = CrossAttention(q_channels=512, kv_channels=6144, embed_dim=256)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(2048, 1)

    def _get_resnet18(self):
        model = models.resnet18(pretrained=True)
        return nn.Sequential(*list(model.children())[:-2])

    def _adapt_resnet18_for_single_channel(self):
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        return nn.Sequential(*list(model.children())[:-2])

    def forward(self, x, T, B, R, I):
        x_feat = self.resnet101(x) 

        dynamic_B = self.dynamic_filter_R(x_feat, T)
        dynamic_I = self.dynamic_filter_T(x_feat, R)
        new_x_feat = torch.cat([x_feat,dynamic_I, dynamic_B], dim=1) 

        i_feat = self.wsp_i(self.resnet18_single(I))
        b_feat = self.wsp_b(self.resnet18(B))
        r_feat = self.wsp_r(self.resnet18(R))
        t_feat = self.wsp_t(self.resnet18(T))
        
        new_x_feat = torch.cat([x_feat,dynamic_I, dynamic_B], dim=1) 
        ca_tb = self.ca_tb(i_feat, t_feat, new_x_feat)
        ca_ri = self.ca_ri(r_feat, b_feat, new_x_feat)

        fused = self.fusion_conv(ca_tb + ca_ri)
        global_feat = F.adaptive_avg_pool2d(fused, (1,1)).flatten(1)
        return self.fc(global_feat)
