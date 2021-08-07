import os
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model_i3d import build_fc, ConvResponseModel, FcFusion, ConvFusion
from pyramidpooling3d import SpatialPyramidPooling3D, SpatialPyramidPooling2D


class BitNeck(nn.Module):
    def __init__(self, hparams):
        super(BitNeck, self).__init__()
        self.hparams = hparams
        assert self.hparams.old_mix
        assert self.hparams.pathway == 'none'

        video_size = hparams['video_size'] if hparams['crop_size'] == 0 else hparams['crop_size']
        self.x1_twh = (int(hparams['video_frames'] / 2), int(video_size / 4), int(video_size / 4))
        self.x2_twh = tuple(map(lambda x: int(x / 2), self.x1_twh))
        self.x3_twh = tuple(map(lambda x: int(x / 2), self.x2_twh))
        self.x4_twh = tuple(map(lambda x: int(x / 2), self.x3_twh))
        self.x1_c, self.x2_c, self.x3_c, self.x4_c = 256, 512, 1024, 2048
        self.x5_dim = 21843  # imagenet 21k
        self.twh_dict = {'x1': self.x1_twh, 'x2': self.x2_twh, 'x3': self.x3_twh, 'x4': self.x4_twh}
        self.c_dict = {'x1': self.x1_c, 'x2': self.x2_c, 'x3': self.x3_c, 'x4': self.x4_c}
        self.planes = hparams['conv_size']
        self.pyramid_layers = hparams['pyramid_layers'].split(',')  # x1,x2,x3,x4
        self.pyramid_layers.sort()
        assert len(self.pyramid_layers) == 1
        self.layer = self.pyramid_layers[0]

        if hparams['separate_rois']:
            self.rois = hparams['rois'].split(',')
            self.output_sizes = hparams['roi_lens']
        else:
            self.rois = [hparams['rois']]
            self.output_sizes = [hparams['output_size']]

        self.first_convs = nn.ModuleDict()
        self.poolings = nn.ModuleDict()
        self.fc_input_dims = {}
        self.ch_response = nn.ModuleDict()
        self.final_fusions = nn.ModuleDict()

        for roi, output_size in zip(self.rois, self.output_sizes):
            if self.layer == 'x5':  # i3d_flow
                self.neck = build_fc(hparams, self.x5_dim, output_size, part='first')
                nn.LSTM(input_size=self.x5_dim, hidden_size=self.hparams.layer_hidden,
                        num_layers=self.hparams.lstm_layers, batch_first=True)
                continue
            else:
                self.first_conv = nn.Conv3d(self.c_dict[self.layer], self.planes, kernel_size=1, stride=1)
                if hparams['spp']:
                    levels = np.array([hparams['spp_size'], hparams['spp_size']])
                    self.pooling = SpatialPyramidPooling2D(
                        levels=levels,
                        mode=hparams['pooling_mode']
                    )
                    in_dim = np.sum(levels[0] * levels[1]) * self.planes
                else:
                    if hparams['pooling_mode'] == 'avg':
                        self.pooling = nn.AdaptiveAvgPool2d(hparams['pooling_size'])
                        in_dim = hparams['pooling_size'] * hparams['pooling_size'] * self.planes
                    elif hparams['pooling_mode'] == 'max':
                        self.pooling = nn.AdaptiveMaxPool2d(hparams['pooling_size'])
                        in_dim = hparams['pooling_size'] * hparams['pooling_size'] * self.planes
                    else:
                        NotImplementedError()
                self.neck = build_fc(hparams, in_dim, output_size, part='first')



        in_size = self.hparams.first_layer_hidden * self.num_chs if hparams['final_fusion'] == 'concat' \
            else self.hparams.first_layer_hidden
        for roi, output_size in zip(self.rois, self.output_sizes):
            self.final_fusions.update({f'{roi}': nn.Sequential(
                FcFusion(fusion_type=hparams['final_fusion']),
                build_fc(self.hparams, in_size, output_size),
            )})

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
            nn.Sigmoid(),
            pool,
        )

        self.lstm = nn.LSTM(input_size=self.lstm_in_dim, hidden_size=self.hparams.layer_hidden,
                            num_layers=self.hparams.lstm_layers, batch_first=True)

        self.fc = build_fc(hparams, self.hparams.layer_hidden * self.hparams.video_frames, hparams['output_size'])

    def forward(self, x):
        # x: (None, D, H, W)
        x = x[self.bdcn_outputs[0]]
        x = self.read_out_layers(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.lstm(x)[0]
        # print(x.shape)
        out = self.fc(x.reshape(x.shape[0], -1))
        out_aux = None
        return out, out_aux
