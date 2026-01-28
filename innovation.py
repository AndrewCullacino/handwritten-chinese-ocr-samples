"""
Apache v2 license
Copyright (C) <2018-2021> Intel Corporation
SPDX-License-Identifier: Apache-2.0
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """
    SENet Paper: https://arxiv.org/pdf/1709.01507.pdf
    Code: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return self.sigmoid(attn)


class ResidualSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ResidualSpatialAttention, self).__init__()
        self.spatial = SpatialAttention(kernel_size=kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attn = self.spatial(x)
        return x * (1.0 + self.gamma * attn)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel, block, num_blocks):
        super(ResNet, self).__init__()
        inout_channels = [int(out_channel / 8), # 64
                          int(out_channel / 4), #128
                          int(out_channel / 2), #256
                          int(out_channel / 1), #512
                          out_channel]          #512
        self.inplanes = int(out_channel / 8) #64

        self.conv0_1 = nn.Conv2d(in_channel, inout_channels[0], 3, 1, 1)
        self.bn0_1 = nn.BatchNorm2d(inout_channels[0])
        self.conv0_2 = nn.Conv2d(inout_channels[0], self.inplanes, 3, 1, 1)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)

        self.block1 = self._make_block(block, inout_channels[1], num_blocks[0])
        self.conv1 = nn.Conv2d(inout_channels[1], inout_channels[1], 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(inout_channels[1])

        self.block2 = self._make_block(block, inout_channels[2], num_blocks[1])
        self.conv2 = nn.Conv2d(inout_channels[2], inout_channels[2], 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(inout_channels[2])

        self.block3 = self._make_block(block, inout_channels[3], num_blocks[2])
        self.conv3 = nn.Conv2d(inout_channels[3], inout_channels[3], 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(inout_channels[3])

        self.block4 = self._make_block(block, inout_channels[4], num_blocks[3])
        self.conv4 = nn.Conv2d(inout_channels[4], inout_channels[4], 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(inout_channels[4])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.9)

    def _make_block(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        blocks = []
        blocks.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            blocks.append(block(self.inplanes, planes))
        return nn.Sequential(*blocks)

    def forward(self, x):
        features = []  # 用于存储多尺度特征

        # stage 0
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # stage 1
        x = self.block1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        features.append(x)  # 128

        # stage 2
        x = self.block2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout2(x)
        features.append(x)  # 256

        # stage 3
        x = self.block3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout3(x)
        features.append(x)  # 512

        # stage 4
        x = self.block4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout4(x)
        features.append(x)  # 512

        return x, features


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, channels=[128, 256, 512, 512], out_channels=512):
        super(MultiScaleFeatureFusion, self).__init__()
        
        # 特征转换层
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for c in channels
        ])
        
        # 特征融合后的处理
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # 将所有特征图调整到相同大小并转换通道数
        target_size = features[-1].shape[2:]  # 使用最后一层的尺寸
        transformed = []
        
        for feat, transform in zip(features, self.transforms):
            x = transform(feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            transformed.append(x)
        
        # 特征融合（加权求和）
        fused = sum(transformed) / len(transformed)
        return self.fusion(fused)


class hctr_model(nn.Module):
    def __init__(self, use_multiscale=False, use_spatial_attn=False):
        super(hctr_model, self).__init__()
        self.img_height = 128
        self.PAD = 'NormalizePAD'
        self.optimizer = 'Adam'
        self.pred = 'Classification'
        self.num_classes = None
        self.use_multiscale = use_multiscale
        self.use_spatial_attn = use_spatial_attn

        self.cnn = ResNet(1, 512, BasicBlock, [2, 4, 5, 1])
        
        if self.use_multiscale:
            self.feature_fusion = MultiScaleFeatureFusion()
        if self.use_spatial_attn:
            self.spatial_attn = ResidualSpatialAttention()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = None

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
        self.fc = nn.Linear(512, self.num_classes)
        return self

    def forward(self, input):
        if self.num_classes is None or self.fc is None:
            raise ValueError("请先调用set_num_classes设置类别数")
        
        if len(input.shape) != 4:
            raise ValueError(f"输入维度应该是[B,C,H,W]，但得到{input.shape}")
        
        # 获取CNN特征
        x, features = self.cnn(input)
        
        # 使用多尺度特征融合
        if self.use_multiscale:
            x = self.feature_fusion(features)
        if self.use_spatial_attn:
            x = self.spatial_attn(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        if x.shape[1] != self.num_classes:
            raise ValueError(f"输出维度错误：期望{self.num_classes}类，但得到{x.shape[1]}类")
        
        return x
