from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = Inconv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1_a3up = Up1(512, 256)
        self.up1_b3up = Up1(512, 256)
        self.up2_a3up = Up2(256, 128)
        self.up2_b3up = Up2(256, 128)
        self.up3_a3up = Up3(128, 64)
        self.up3_b3up = Up3(128, 64)
        self.up4_a3up = Up4(64, 64)
        self.up4_b3up = Up4(64, 64)
        self.outc_a3up = Outconv(64, n_classes)
        self.outc_b3up = Outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1_a3up(x5, x4)
        x6_b = self.up1_b3up(x5, x4)

        x7 = self.up2_a3up(x6, x3, x4)
        x7_b = self.up2_b3up(x6_b, x3, x4)

        x8 = self.up3_a3up(x7, x3, x2)
        x8_b = self.up3_b3up(x7_b, x3, x2)

        x9 = self.up4_a3up(x8, x1, x2)
        x9_b = self.up4_b3up(x8_b, x1, x2)

        x10 = self.outc_a3up(x9)
        x10_b = self.outc_b3up(x9_b)

        return x10, x10_b


if __name__ == '__main__':
    input = torch.rand(4, 3, 512, 512)
    unet = UNet(n_channels=3, n_classes=1)
    output_area, output_bd = unet(input)

    for name, param in unet.named_parameters():
        if param.requires_grad:
            print(name)