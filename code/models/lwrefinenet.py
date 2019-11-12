# Code taken and adapted from:
# https://github.com/DrSleep/light-weight-refinenet/blob/9d539788892a91e2a797b1f06b6de68063188e20/models/mobilenet.py
# https://github.com/DrSleep/light-weight-refinenet/blob/9d539788892a91e2a797b1f06b6de68063188e20/utils/layer_factory.py
"""RefineNet-LightWeight
RefineNet-LigthWeight PyTorch for non-commercial purposes
Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn


class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block from https://arxiv.org/abs/1801.04381"""
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super(InvertedResidualBlock, self).__init__()
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1)
        self.output = nn.Sequential(convbnrelu(in_planes, intermed_planes, 1),
                                    convbnrelu(intermed_planes, intermed_planes, 3, stride=stride, groups=intermed_planes),
                                    convbnrelu(intermed_planes, out_planes, 1, act=False))

    def forward(self, x):
        residual = x
        out = self.output(x)
        if self.residual:
            return out + residual
        else:
            return out


class MBv2(nn.Module):
    """Net Definition"""
    mobilenet_config = [[1, 16, 1, 1], # expansion rate, output channels, number of repeats, stride
                        [6, 24, 2, 2],
                        [6, 32, 3, 2],
                        [6, 64, 4, 2],
                        [6, 96, 3, 1],
                        [6, 160, 3, 2],
                        [6, 320, 1, 1],
                       ]
    in_planes = 32 # number of input channels
    num_layers = len(mobilenet_config)

    def __init__(self, num_channels, num_classes):
        super(MBv2, self).__init__()

        self.layer1 = convbnrelu(num_channels, self.in_planes, kernel_size=3, stride=2)
        c_layer = 2
        for t,c,n,s in (self.mobilenet_config):
            layers = []
            for idx in range(n):
                layers.append(InvertedResidualBlock(self.in_planes, c, expansion_factor=t, stride=s if idx == 0 else 1))
                self.in_planes = c
            setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers))
            c_layer += 1

        # Light-Weight RefineNet #
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
        self.crp4 = self._make_crp(256, 256, 4)
        self.crp3 = self._make_crp(256, 256, 4)
        self.crp2 = self._make_crp(256, 256, 4)
        self.crp1 = self._make_crp(256, 256, 4)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.segm = conv3x3(256, num_classes, bias=True)
        self.relu = nn.ReLU6(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)  # x / 2
        l3 = self.layer3(x)  # 24, x / 4
        l4 = self.layer4(l3)  # 32, x / 8
        l5 = self.layer5(l4)  # 64, x / 16
        l6 = self.layer6(l5)  # 96, x / 16
        l7 = self.layer7(l6)  # 160, x / 32
        l8 = self.layer8(l7)  # 320, x / 32
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size=l6.size()[2:], mode='bilinear', align_corners=True)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size=l4.size()[2:], mode='bilinear', align_corners=True)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        out_segm = self.segm(l3)

        return out_segm

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)


def mbv2(num_classes, **kwargs):
    """Constructs the network.
    Args:
        num_classes (int): the number of classes for the segmentation head to output.
    """
    model = MBv2(num_classes, **kwargs)
    return model


def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_planes),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_planes))


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x
