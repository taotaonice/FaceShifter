import torch
from torch import nn


def conv4x4(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c,kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True),
    )


def deconv4x4(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.1, inplace=True),
    )


class HearNet(nn.Module):
    def __init__(self):
        super(HearNet, self).__init__()
        self.down1 = conv4x4(6, 64)
        self.down2 = conv4x4(64, 128)
        self.down3 = conv4x4(128, 256)
        self.down4 = conv4x4(256, 512)
        self.down5 = conv4x4(512, 512)

        self.up1 = deconv4x4(512, 512)
        self.up2 = deconv4x4(512*2, 256)
        self.up3 = deconv4x4(256*2, 128)
        self.up4 = deconv4x4(128*2, 64)
        self.up5 = nn.Conv2d(64*2, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        c4 = self.down4(c3)
        c5 = self.down5(c4)

        m1 = self.up1(c5)
        m1 = torch.cat((c4, m1), dim=1)
        m2 = self.up2(m1)
        m2 = torch.cat((c3, m2), dim=1)
        m3 = self.up3(m2)
        m3 = torch.cat((c2, m3), dim=1)
        m4 = self.up4(m3)
        m4 = torch.cat((c1, m4), dim=1)

        out = nn.functional.interpolate(m4, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.up5(out)
        return torch.tanh(out)
