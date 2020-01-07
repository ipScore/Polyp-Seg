import torch
import torch.nn as nn


class Double_conv(nn.Module):
    # (conv => BN => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # default: nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nn.BatchNorm2d(out_ch, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            Double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up1, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Double_conv(2 * in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  #  order matters?
        x = self.conv(x)
        return x


class Up_cons(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up_cons, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Double_conv(2 * in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Up2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up2, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Double_conv(1024, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x3, x2, x1], dim=1)  #  order matters?
        x = self.conv(x)
        return x


class Up3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up3, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Double_conv(512, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x2 = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x3, x2], dim=1)  # order matters?
        x = self.conv(x)
        return x


class Up4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up4, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Double_conv(256, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x3 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)  # 1*1 conv kernal

    def forward(self, x):
        x = self.conv(x)
        return x
