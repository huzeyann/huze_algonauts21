import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model_i3d import build_fc, ConvResponseModel
from pyramidpooling3d import SpatialPyramidPooling3D, SpatialPyramidPooling2D


class VggishNeck(nn.Module):
    def __init__(self, hparams):
        super(VggishNeck, self).__init__()
        self.hparams = hparams
        assert self.hparams.separate_rois == False
        self.rois = self.hparams.rois

        self.in_dim = 128 * 3

        if self.hparams['track'] == 'full_track':
            if self.hparams['no_convtrans']:
                self.head = build_fc(hparams, self.in_dim,
                                     hparams['output_size'])
            else:
                self.head = ConvResponseModel(self.in_dim,
                                              hparams['num_subs'], hparams)
        else:
            self.head = build_fc(hparams, self.in_dim,
                                 hparams['output_size'])

    def forward(self, x):
        # x: (None, D, H, W)
        x = x['audio']
        out = self.head(x)
        out_aux = None
        out = {self.rois: out}
        return out, out_aux
