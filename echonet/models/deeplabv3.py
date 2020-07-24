#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:01:14 2020

@author: zdadadaz
"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from collections import OrderedDict

__all__ = ["DeepLabV3"]


class DeepLabV3(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        xx = self.backbone(x)
        xx1 = self.classifier(xx)
        y = F.interpolate(xx1, size=input_shape, mode='bilinear', align_corners=False)
        
        return y

class DeepLabV3_multi(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLabV3_multi, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        xx = self.backbone(x)

        # multi-task
        ef1 = self.avgpool(xx)
        ef1 = torch.flatten(ef1, 1)
        ef2 = self.fc(ef1)
        
        xx1 = self.classifier(xx)
        y = F.interpolate(xx1, size=input_shape, mode='bilinear', align_corners=False)
        
        return y, ef2

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def DeepLabV3_main():
    model = torchvision.models.__dict__["resnet50"](
        pretrained=False,
        replace_stride_with_dilation=[False, True, True])
    backbone = torch.nn.Sequential(*(list(model.children())[:-2]))

    classifier = DeepLabHead(2048, 1)
    model = DeepLabV3(backbone, classifier)
    return model

def DeepLabV3_multi_main():
    backbone = torchvision.models.__dict__["resnet50"](
        pretrained=False,
        replace_stride_with_dilation=[False, True, True])
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))

    classifier = DeepLabHead(2048, 1)
    model= DeepLabV3_multi(backbone, classifier)
    return model


class ASPPConv_3D(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv_3D, self).__init__(*modules)


class ASPPPooling_3D(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling_3D, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        print(x.size(),size)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)


class ASPP_3D(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP_3D, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv_3D(in_channels, out_channels, rate1))
        modules.append(ASPPConv_3D(in_channels, out_channels, rate2))
        modules.append(ASPPConv_3D(in_channels, out_channels, rate3))
        modules.append(ASPPPooling_3D(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHead_3D(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead_3D, self).__init__(
            ASPP_3D(in_channels, [12, 24, 36]),
            nn.Conv3d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, num_classes, 1)
        )


class DeepLabV3_3D(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLabV3_3D, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        xx = self.backbone(x)
        xx1 = self.classifier(xx)
        y = F.interpolate(xx1, size=input_shape, mode='bilinear', align_corners=False)
        
        return y


def DeepLabV3_3D_main():
    model = torchvision.models.video.__dict__['r2plus1d_18'](pretrained=False)
    backbone = torch.nn.Sequential(*(list(model.children())[:-2]))
    classifier = DeepLabHead_3D(512, 1)
    model = DeepLabV3_3D(backbone, classifier)
    return model

# +
# X = torch.rand(3*2,3,32,112,112)
# # flow = torch.rand(3*2,2,32,112,112)
# model = DeepLabV3_3D_main()
# print(model(X).size())
