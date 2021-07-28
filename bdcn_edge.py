import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model_i3d import build_fc
from pyramidpooling import SpatialPyramidPooling


class BDCNNeck(nn.Module):
    def __init__(self, hparams):
        super(BDCNNeck, self).__init__()
        self.hparams = hparams
        self.bdcn_outputs = [int(i) for i in self.hparams.bdcn_outputs.split(',')]
        self.num_bdcns = len(self.bdcn_outputs)
        assert self.num_bdcns == 1

        self.pool_size = self.hparams.bdcn_pool_size
        self.fc_in_dim = int((self.hparams.video_size / self.pool_size) ** 2 * 1 * self.hparams.video_frames)

        if self.hparams.pooling_mode == 'max':
            # pool = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
            pool = nn.AdaptiveMaxPool2d(self.pool_size)
        elif self.hparams.pooling_mode == 'avg':
            # pool = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.pool_size)
            pool = nn.AdaptiveAvgPool2d(self.pool_size)
        else:
            NotImplementedError()

        self.read_out_layers = nn.Sequential(
            nn.Sigmoid(),
            pool,
            nn.Flatten(),
        )

        self.fc = build_fc(hparams, self.fc_in_dim, hparams['output_size'])

    def forward(self, x):
        # x: (None, C, D, H, W)
        x = x[self.bdcn_outputs[0]]
        # print(x.shape)
        x = self.read_out_layers(x)
        out = self.fc(x)
        out_aux = None
        return out, out_aux
