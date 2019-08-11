from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


class conv_sknet(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16):
        super(conv_sknet, self).__init__()

        def register(name, tensor):
            self.register_parameter(name, Parameter(tensor))

        register('A', torch.rand(max(out_ch // reduction, 32), out_ch))
        register('B', torch.rand(max(out_ch // reduction, 32), out_ch))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, dilation=2, padding=2),
            nn.BatchNorm2d(out_ch, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
                nn.Linear(out_ch, max(out_ch // reduction, 32)),
                nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_3 = self.conv_3(x)
        x_5 = self.conv_5(x)
        x_fuse = x_3 + x_5
        b, c, _, _ = x_fuse.size()  # b=4, c=128
        x_fuse_s = self.avg_pool(x_fuse).view(b, c)
        x_fuse_z = self.fc(x_fuse_s)

        s1 = torch.Tensor(np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) / (np.exp(np.array(torch.mm(x_fuse_z, self.A).cpu().detach().numpy())) + np.exp(np.array(torch.mm(x_fuse_z, self.B).cpu().detach().numpy()))))
        s1 = s1.view(b, c, 1, 1).cuda()
        s2 = 1 - s1
        V_a = x_3 * s1
        V_b = x_5 * s2
        V = V_a + V_b

        return V


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()
        self.conv = nn.Sequential(
            conv_sknet(in_ch, out_ch),
            conv_sknet(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_sknet(in_ch, out_ch),
            conv_sknet(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up1, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv_sknet(2 * in_ch, in_ch)
        self.conv2 = conv_sknet(in_ch, out_ch)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  #  order matters?
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Up2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up2, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv_sknet(1024, 2 * out_ch)
        self.conv2 = conv_sknet(2 * out_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Up3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up3, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv_sknet(512, 2 * out_ch)
        self.conv2 = conv_sknet(2 * out_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x2 = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x3, x2], dim=1)  # order matters?
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Up4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up4, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv_sknet(256, out_ch)
        self.conv2 = conv_sknet(out_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)  # 1*1 conv kernal

    def forward(self, x):
        x = self.conv(x)
        return x

