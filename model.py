import torch
import torch.nn as nn
from torchvision.models import resnet
from coord_conv import CoordConv, CoordConvTranspose, AddCoordinates

class Lateral(nn.Module):
    def __init__(self, in_channel, out_channel, with_r=True):
        super().__init__()
        r = 3 if with_r else 2
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid(),
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel + r, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.layer(x)
        h, w = x.shape[2:]
        mask = self.attn(x).repeat(1, 1, h, w)
        x = x * mask
        return x

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False, down=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1 or down:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False, down=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias, down)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, down)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(CoordConv(in_planes, in_planes//4, 1, 1, 0, bias=bias, with_r=True),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(CoordConvTranspose(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias, with_r=True),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(CoordConv(in_planes//4, out_planes, 1, 1, 0, bias=bias, with_r=True),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)
        return x


class DNet(nn.Module):
    def __init__(self, n_classes=1):
        super(DNet, self).__init__()
        base = resnet.resnet18(pretrained=True)

        self.in_block = nn.Sequential(
            nn.Conv2d(1, 64, (7, 7), stride=(2, 2), padding=3),
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4


        self.decoder1 = Decoder(256, 256, 3, 1, 1, 0)
        self.decoder2 = Decoder(256, 256, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 256, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        #self.bp1 = CoordConv(64, 256, 1, 1, 0, bias=False, with_r=True)
        #self.bp2 = CoordConv(64, 256, 1, 1, 0, bias=False, with_r=True)
        #self.bp3 = CoordConv(128, 256, 1, 1, 0, bias=False, with_r=True)
        #self.bp4 = CoordConv(256, 256, 1, 1, 0, bias=False, with_r=True)

        self.bp1 = Lateral(64, 256, with_r=True)
        self.bp2 = Lateral(64, 256, with_r=True)
        self.bp3 = Lateral(128, 256, with_r=True)
        self.bp4 = Lateral(256, 256, with_r=True)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)

        # Classifier
        #self.classifier1 = nn.Conv2d(256, n_classes, 3, 1, 1)
        #self.classifier2 = nn.Conv2d(256, n_classes, 3, 1, 1)
        #self.classifier3 = nn.Conv2d(256, n_classes, 3, 1, 1)
        #self.classifier4 = nn.Conv2d(256, n_classes, 3, 1, 1)

        self.classifier1 = CoordConvTranspose(256, n_classes, 2, 2, 0, with_r=True)
        self.classifier2 = CoordConvTranspose(256, n_classes, 2, 2, 0, with_r=True)
        self.classifier3 = CoordConvTranspose(256, n_classes, 2, 2, 0, with_r=True)
        self.classifier4 = CoordConvTranspose(256, n_classes, 2, 2, 0, with_r=True)

        self.tp_conv1 = nn.Sequential(CoordConvTranspose(256, 32, 3, 2, 1, 1, with_r=True),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(CoordConv(32, 32, 3, 1, 1, with_r=True),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        #self.tp_conv2 = nn.Conv2d(32, n_classes, 3, 1, 1)
        self.tp_conv2 = CoordConvTranspose(32, n_classes, 2, 2, 0, with_r=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        # Initial block
        # print(x.shape)
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        e4 = self.decoder4(e4)
        d4 = self.relu(self.bn4(self.bp4(e3)) + e4[:, :, :e3.shape[2], :e3.shape[3]])
        d3 = self.relu(self.bn3(self.bp3(e2)) + self.decoder3(d4))
        d2 = self.relu(self.bn2(self.bp2(e1)) + self.decoder2(d3))
        d1 = self.relu(self.bn1(self.bp1(x)) + self.decoder1(d2))

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.sigm(self.tp_conv2(y))

        y1 = self.sigm(self.classifier1(d1))
        y2 = self.sigm(self.classifier2(d2))
        y3 = self.sigm(self.classifier3(d3))
        y4 = self.sigm(self.classifier4(d4))
        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)
        # print(y4.shape)
        # print(y.shape)
        return y4, y3, y2, y1, y

