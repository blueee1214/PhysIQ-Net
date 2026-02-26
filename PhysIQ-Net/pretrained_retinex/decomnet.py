import torch
import torch.nn as nn
import torchvision.models as models
import os
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import BPoly
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def load(self, load_path, optim=None):
        if load_path.endswith(".pth"):
            self.load_state_dict(torch.load(load_path, map_location='cuda', weights_only=False))
            print(f"Loading net from {load_path} ...")
            self.start_ep, self.total_steps = 0, 0
            return None
        elif load_path.endswith(".tar"):
            ckpt = torch.load(load_path, map_location='cuda', weights_only=False)
            print(f"Checkpoint keys: {ckpt.keys()}")  # 打印文件内容
            self.load_state_dict(ckpt)  # 直接加载权重
            print(f"Loading from {load_path} ...")
            return optim
        else:
            print(f"Fail to load from {load_path}.")
            
    def to_cuda(self):
        self.cuda()

    def get_parameters(self):
        return self.parameters()

    def get_state_dict(self):
        return self.state_dict()

    def to_eval(self):
        self.eval()  # switch to eval mode

    def to_train(self):
        self.train()  # switch to train mode
# 确保 DecomNet 已正确定义

class DecomNet(BaseNet):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L

