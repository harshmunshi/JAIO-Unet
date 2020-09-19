import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class DownConvBlock(nn.Module):
    """
    NOTE: This is for the down block including the double conv
    """
    def __init__(self, in_planes, out_planes, stride, padding, kernel_size):
        self.conv1 = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2D(out_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        # no nonlinearity after this point
    
    def forward(self,inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return out

class Encoder(nn.Module):
    """
    Encoder of the U-Net arch, used to capture the context
    """
    def __init__(self, in_planes, out_planes):
        super(Encoder, self).__init__()
        # define the network here
        # The first down conv block
        self.dconv1 = DownConvBlock(in_planes, out_planes, stride=1, padding=1, kernel_size=3) #d=64
        self.dconv2 = DownConvBlock(out_planes, out_planes*2, stride=1, padding=1, kernel_size=3) #d=128
        self.dconv3 = DownConvBlock(out_planes*2, out_planes*4, stride=1, padding=1, kernel_size=3) #d=256
        self.dconv4 = DownConvBlock(out_planes*4, out_planes*8, stride=1, padding=1, kernel_size=3) #d=512
        self.dconv5 = DownConvBlock(out_planes*8, out_planes*16, stride=1, padding=1, kernel_size=3) #d=1024
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, inputs):
        out_dconv1 = self.dconv1(inputs)
        out_dconv1_mp = self.maxpool(out_dconv1)

        out_dconv2 = self.dconv2(out_dconv1_mp)
        out_dconv2_mp = self.maxpool(out_dconv2)

        out_dconv3 = self.dconv3(out_dconv2_mp)
        out_dconv3_mp = self.maxpool(out_dconv3)
        
        out_dconv4 = self.dconv4(out_dconv3_mp)
        out_dconv4_mp = self.maxpool(out_dconv4)

        out_dconv5 = self.dconv5(out_dconv4_mp)
        return [out_dconv1, out_dconv2, out_dconv3, out_dconv4, out_dconv5]


class Decoder(nn.Module):
    def __init__(self, in_planes, out_planes, encoder_in, encoder_out, n_classes):
        # in case of the decoder the input will be the last layer
        self.encoder = Encoder(encoder_in, encoder_out)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.uconv1 = DownConvBlock(out_planes*16, out_planes*8, stride=1, padding=1, kernel_size=3) #1024->512
        self.uconv2 = DownConvBlock(out_planes*8, out_planes*4, stride=1, padding=1, kernel_size=3) #512 -> 256
        self.uconv3 = DownConvBlock(out_planes*4, out_planes*2, stride=1, padding=1, kernel_size=3) #256 -> 128
        self.uconv4 = DownConvBlock(out_planes*2, out_planes, stride=1, padding=1, kernel_size=3) # 128 -> 64

        # The output map is a 1x1 convolutional layer 
        self.output_map = nn.Conv2D(out_planes, n_classes, kernel_size=1)
    
    def forward(self, x):
        # c1 is out_dconv1, c2 is out_dconv2.... etc
        c1, c2, c3, c4, c5 = self.encoder(x)

        # upsample c5 and add to c4
        in_1 = torch.cat([c4, self.up(c5)], dim=1)
        x = self.uconv1(in_1)

        in_2 = torch.cat([c3, self.up(x)], dim=1)
        x = self.uconv2(in_2)

        in_3 = torch.cat([c2, self.up(x)], dim=1)
        x = self.uconv3(in_3)

        int_4 = torch.cat([c1, self.up(x)], dim=1)
        x = self.uconv4(in_4)

        x = self.output_map(x)
        return x