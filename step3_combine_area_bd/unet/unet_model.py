from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = Inconv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1_3up = Up1(512, 256)
        self.up1_b = Up1(512, 256)
        self.up2_3up = Up2(256, 128)
        self.up2_b = Up2(256, 128)
        self.up3_3up = Up3(128, 64)
        self.up3_b = Up3(128, 64)
        self.up4_3up = Up4(64, 64)
        self.up4_b = Up4(64, 64)
        self.outc_3up = Outconv(64, n_classes)
        self.outc_b = Outconv(64, n_classes)

        self.inc_cons = Inconv(1, 64)
        self.down1_cons = Down(64, 128)
        self.down2_cons = Down(128, 128)
        self.up1_cons = Up_cons(128, 64)
        self.up2_cons = Up_cons(64, 64)
        self.outc_cons = Outconv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1_3up(x5, x4)
        x6_b = self.up1_b(x5, x4)
        x7 = self.up2_3up(x6, x3, x4)
        x7_b = self.up2_b(x6_b, x3, x4)
        x8 = self.up3_3up(x7, x3, x2)
        x8_b = self.up3_b(x7_b, x3, x2)
        x9 = self.up4_3up(x8, x1, x2)
        x9_b = self.up4_b(x8_b, x1, x2)
        x10 = self.outc_3up(x9)
        x10_b = self.outc_b(x9_b)

        x11 = self.inc_cons(x10)
        x12 = self.down1_cons(x11)
        x13 = self.down2_cons(x12)
        x14 = self.up1_cons(x13, x12)
        x15 = self.up2_cons(x14, x11)
        x16 = self.outc_cons(x15)

        return x10, x10_b, x16


if __name__ == '__main__':
    input = torch.rand(4, 3, 512, 512)
    unet = UNet(n_channels=3, n_classes=1)
    output_area, output_bd, output_bd_constraint = unet(input)

    # print(output_area.size())  # same as print(output_area.shape)
    # print(output_bd.size())  # same as print(output_bd.shape)
    # print(output_bd_constraint.size())  # same as print(output_bd.shape)
    #
    for name, param in unet.named_parameters():
        if param.requires_grad:
            print(name)
