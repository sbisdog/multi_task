import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.detection.models.backbone import ResNetBackbone
from public.detection.models.head import CenterNetHetRegWhHead

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = [
    'resnet18_centernet',
    'resnet34_centernet',
    'resnet50_centernet',
    'resnet101_centernet',
    'resnet152_centernet',
]

model_urls = {
    'resnet18_centernet':
    '{}/detection_models/resnet18dcn_centernet_coco_multi_scale_resize512_mAP0.266.pth'
    .format(pretrained_models_path),
    'resnet34_centernet':
    'empty',
    'resnet50_centernet':
    'empty',
    'resnet101_centernet':
    'empty',
    'resnet152_centernet':
    'empty',
}



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SEBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1):
        super(SEBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()

#         self.stride = stride

    def forward(self, x):
        residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

        out = self.ca(x) * x
        out = self.sa(out) * out

        out += residual
        out = self.relu(out)

        return out


class UP_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP_layer, self ).__init__() 
        self.layer = nn.Sequential(
        #                 DeformableConv2d(in_channels,
        #                                  out_channels,
        #                                  kernel_size=(3, 3),
        #                                  stride=1,
        #                                  padding=1,
        #                                  groups=1,
        #                                  bias=False)
                        nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=out_channels,
                                           out_channels=out_channels,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1,
                                           output_padding=0,
                                           bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
        )
        for m in self.layer.modules():
            if isinstance(m, nn.ConvTranspose2d):
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (
                            1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
        
    def forward(self, x):
        out = self.layer(x)
        return out

class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, selayer=False):
        super(ShortCut, self).__init__()
        if selayer:
            self.conv1 = nn.Sequential(
                        SEBlock(in_channels),
                        nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False)
                )
        else:
            self.conv1 = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False)
        for m in self.conv1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        return self.conv1(x)


class TTFHead(nn.Module):
    def __init__(self, C5_inplanes, num_classes, out_channels, selayer):
        super(TTFHead, self).__init__()
        self.up5t4 = UP_layer(C5_inplanes, out_channels[0])
        self.up4t3 = UP_layer(out_channels[0], out_channels[1])
        self.up3t2 = UP_layer(out_channels[1], out_channels[2])
        
        self.shortcut4 = ShortCut(int(C5_inplanes/2),
                                    out_channels[0], selayer)
        self.shortcut3 = ShortCut(int(C5_inplanes/4),
                                    out_channels[1], selayer)
        self.shortcut2 = ShortCut(int(C5_inplanes/8),
                                    out_channels[2], selayer)
        
        
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64,
                      out_channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[-1],
                      num_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(64,
                      out_channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[-1],
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(64,
                      out_channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels[-1],
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        )
        
        self.heatmap_head[-1].bias.data.fill_(-2.19)

        for m in self.offset_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        [C2, C3, C4, C5] = x
        C4_up = self.up5t4(C5) + self.shortcut4(C4)
        C3_up = self.up4t3(C4_up) + self.shortcut3(C3)
        C2_up = self.up3t2(C3_up) + self.shortcut2(C2)
        
        del C2, C3, C4, C5, C3_up, C4_up
        heatmap_output = self.heatmap_head(C2_up)
        offset_output = self.offset_head(C2_up)
        wh_output = self.wh_head(C2_up)

        return heatmap_output, offset_output, wh_output
    
    


# assert input annotations are[x_min,y_min,x_max,y_max]
class CenterNet(nn.Module):
    def __init__(self, resnet_type, num_classes=80, multi_head=False, selayer=True, use_ttf=False, cls_mlp=False):
        super(CenterNet, self).__init__()
        self.backbone = ResNetBackbone(resnet_type=resnet_type)
        expand_ratio = {
            "resnet18": 1,
            "resnet34": 1,
            "resnet50": 4,
            "resnet101": 4,
            "resnet152": 4
        }
        C5_inplanes = int(512 * expand_ratio[resnet_type])
        self.multi_head = multi_head
        self.use_ttf = use_ttf

        self.centernet_head = CenterNetHetRegWhHead(
            C5_inplanes,
            num_classes[0],
            num_layers=3,
            out_channels=[256, 128, 64])
        
        if multi_head:
            if selayer:
                if use_ttf:
                    self.centernet_head_2 = nn.Sequential(
                                        SEBlock(C5_inplanes),
                                        TTFHead(
                                            C5_inplanes,
                                            num_classes[1],
                                            out_channels=[256, 128, 64],
                                            selayer = selayer
                                        ))
                else:
                    self.centernet_head_2 = nn.Sequential(
                                            SEBlock(C5_inplanes),
                                            CenterNetHetRegWhHead(
                                                C5_inplanes,
                                                num_classes[1],
                                                num_layers=3,
                                                out_channels=[256, 128, 64])
                                            )
                
                    
            else:
                if use_ttf:
                    self.centernet_head_2 = TTFHead(
                                                C5_inplanes,
                                                num_classes[1],
                                                out_channels=[256, 128, 64],
                                                selayer = selayer
                                            )
                else:
                    self.centernet_head_2 = CenterNetHetRegWhHead(
                        C5_inplanes,
                        num_classes[1],
                        num_layers=3,
                        out_channels=[256, 128, 64])
        self.cls_mlp = cls_mlp
        if cls_mlp:
            self.cls = nn.Conv2d(num_classes[0], 1, 1)
            self.offset = nn.Conv2d(2, 2, 1)
            self.wh = nn.Conv2d(2, 2, 1)

    def forward(self, inputs):
        [C2, C3, C4, C5] = self.backbone(inputs)

        del inputs
        
        if self.multi_head:
            if self.use_ttf:
                heatmap_output, offset_output, wh_output = self.centernet_head_2([C2, C3, C4, C5])
            else:
                heatmap_output, offset_output, wh_output = self.centernet_head_2(C5)
            del C2, C3, C4
        else:
            heatmap_output, offset_output, wh_output = self.centernet_head(C5)

        del C5
        
        if self.cls_mlp:
            heatmap_output = self.cls(heatmap_output)
            offset_output = self.offset(offset_output)
            wh_output = self.wh(wh_output)

        # if input size:[B,3,640,640]
        # heatmap_output shape:[3, 80, 160, 160]
        # offset_output shape:[3, 2, 160, 160]
        # wh_output shape:[3, 2, 160, 160]

        return heatmap_output, offset_output, wh_output


def _centernet(arch, pretrained, **kwargs):
    model = CenterNet(arch, **kwargs)

    if pretrained:
        pretrained_models = torch.load(model_urls[arch + "_centernet"],
                                       map_location=torch.device('cpu'))

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def resnet18_centernet(pretrained=False, **kwargs):
    return _centernet('resnet18', pretrained, **kwargs)


def resnet34_centernet(pretrained=False, **kwargs):
    return _centernet('resnet34', pretrained, **kwargs)


def resnet50_centernet(pretrained=False, **kwargs):
    return _centernet('resnet50', pretrained, **kwargs)


def resnet101_centernet(pretrained=False, **kwargs):
    return _centernet('resnet101', pretrained, **kwargs)


def resnet152_centernet(pretrained=False, **kwargs):
    return _centernet('resnet152', pretrained, **kwargs)


if __name__ == '__main__':
    net = CenterNet(resnet_type="resnet50", num_classes=[80,1], multi_head=True, selayer=False, use_ttf=True)
    image_h, image_w = 512, 512
    heatmap_output, offset_output, wh_output = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])

    print("1111", heatmap_output.shape, offset_output.shape, wh_output.shape)
