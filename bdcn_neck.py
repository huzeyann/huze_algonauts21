import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model_i3d import build_fc
from pyramidpooling3d import SpatialPyramidPooling3D


class BDCNNeck(nn.Module):
    def __init__(self, hparams):
        super(BDCNNeck, self).__init__()
        self.hparams = hparams

        self.pool_size = self.hparams.bdcn_pool_size
        self.fc_in_dim = int(self.pool_size ** 2 * 1 * self.hparams.video_frames)
        self.lstm_in_dim = int(self.pool_size ** 2 * 1)

        if self.hparams.pooling_mode == 'max':
            # pool = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
            pool = nn.AdaptiveMaxPool2d(self.pool_size)
        elif self.hparams.pooling_mode == 'avg':
            # pool = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.pool_size)
            pool = nn.AdaptiveAvgPool2d(self.pool_size)
        else:
            NotImplementedError()

        self.read_out_layers = nn.Sequential(
            # nn.Sigmoid(),
            pool,
        )

        self.lstm = nn.LSTM(input_size=self.lstm_in_dim, hidden_size=self.hparams.layer_hidden,
                            num_layers=self.hparams.lstm_layers, batch_first=True)

        self.fc = build_fc(hparams, self.hparams.layer_hidden * self.hparams.video_frames, hparams['output_size'])

    def forward(self, x):
        # x: (None, D, H, W)
        x = self.read_out_layers(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.lstm(x)[0]
        # print(x.shape)
        out = self.fc(x.reshape(x.shape[0], -1))
        out_aux = None
        return out, out_aux
