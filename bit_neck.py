import os
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model_i3d import build_fc, ConvResponseModel, FcFusion, ConvFusion
from pyramidpooling3d import SpatialPyramidPooling3D, SpatialPyramidPooling2D


class LSTMReadout(nn.Module):
    def __init__(self):
        super(LSTMReadout, self).__init__()

    def forward(self, x):
        return x[0]


class LSTMReadoutLast(nn.Module):
    def __init__(self):
        super(LSTMReadoutLast, self).__init__()

    def forward(self, x):
        return x[:, -1, :]


class BitNeck(nn.Module):
    def __init__(self, hparams):
        super(BitNeck, self).__init__()
        self.hparams = hparams
        assert self.hparams.old_mix
        assert self.hparams.pathways == 'none'

        video_size = hparams['video_size'] if hparams['crop_size'] == 0 else hparams['crop_size']
        assert video_size == 224
        self.num_frames = hparams['video_frames']
        self.x1_twh = (int(hparams['video_frames'] / 2), int(video_size / 4), int(video_size / 4))
        self.x2_twh = tuple(map(lambda x: int(x / 2), self.x1_twh))
        self.x3_twh = tuple(map(lambda x: int(x / 2), self.x2_twh))
        self.x4_twh = tuple(map(lambda x: int(x / 2), self.x3_twh))
        self.x1_c, self.x2_c, self.x3_c, self.x4_c = 256, 512, 1024, 2048
        self.x5_dim = 1000
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
        assert len(self.rois) == 1
        assert len(self.output_sizes) == 1
        self.roi = self.rois[0]
        output_size = self.output_sizes[0]

        if self.layer == 'x5':
            if self.hparams.track == 'mini_track':
                self.head = nn.Sequential(
                    nn.LSTM(input_size=self.x5_dim, hidden_size=self.hparams.layer_hidden,
                            num_layers=self.hparams.lstm_layers, batch_first=True),
                    LSTMReadout(),
                    LSTMReadoutLast(),
                    nn.Flatten(),
                    build_fc(hparams, self.hparams.layer_hidden, output_size, part='full')
                )
            elif self.hparams.track == 'full_track':
                if self.hparams.no_convtrans:
                    self.head = nn.Sequential(
                        nn.LSTM(input_size=self.x5_dim, hidden_size=self.hparams.layer_hidden,
                                num_layers=self.hparams.lstm_layers, batch_first=True),
                        LSTMReadout(),
                        LSTMReadoutLast(),
                        nn.Flatten(),
                        build_fc(hparams, self.hparams.layer_hidden, output_size, part='full')
                    )
                else:
                    self.head = nn.Sequential(
                        nn.LSTM(input_size=self.x5_dim, hidden_size=self.hparams.layer_hidden,
                                num_layers=self.hparams.lstm_layers, batch_first=True),
                        LSTMReadout(),
                        LSTMReadoutLast(),
                        nn.Flatten(),
                        # build_fc(hparams, self.hparams.layer_hidden, output_size, part='full')
                        ConvResponseModel(self.hparams.layer_hidden, output_size, hparams)
                    )
        else:
            self.first_conv = nn.Conv2d(self.c_dict[self.layer], self.planes, kernel_size=1, stride=1)
            if hparams['spp']:
                levels = np.array(hparams['spp_size'])
                self.pooling = SpatialPyramidPooling2D(
                    levels=levels,
                    mode=hparams['pooling_mode']
                )
                in_dim = np.sum(levels * levels) * self.planes
            else:
                if hparams['pooling_mode'] == 'avg':
                    self.pooling = nn.AdaptiveAvgPool2d(hparams['pooling_size'])
                    in_dim = hparams['pooling_size'] * hparams['pooling_size'] * self.planes
                elif hparams['pooling_mode'] == 'max':
                    self.pooling = nn.AdaptiveMaxPool2d(hparams['pooling_size'])
                    in_dim = hparams['pooling_size'] * hparams['pooling_size'] * self.planes
                else:
                    NotImplementedError()
            self.neck = nn.Sequential(
                self.first_conv,
                self.pooling,
            )

            if self.hparams.track == 'mini_track':
                self.head = nn.Sequential(
                    nn.LSTM(input_size=in_dim, hidden_size=self.hparams.layer_hidden,
                            num_layers=self.hparams.lstm_layers, batch_first=True),
                    LSTMReadout(),
                    LSTMReadoutLast(),
                    nn.Flatten(),
                    build_fc(hparams, self.hparams.layer_hidden, output_size, part='full')
                )
            elif self.hparams.track == 'full_track':
                if self.hparams.no_convtrans:
                    self.head = nn.Sequential(
                        nn.LSTM(input_size=in_dim, hidden_size=self.hparams.layer_hidden,
                                num_layers=self.hparams.lstm_layers, batch_first=True),
                        LSTMReadout(),
                        LSTMReadoutLast(),
                        nn.Flatten(),
                        build_fc(hparams, self.hparams.layer_hidden, output_size, part='full')
                    )
                else:
                    self.head = nn.Sequential(
                        nn.LSTM(input_size=in_dim, hidden_size=self.hparams.layer_hidden,
                                num_layers=self.hparams.lstm_layers, batch_first=True),
                        LSTMReadout(),
                        LSTMReadoutLast(),
                        nn.Flatten(),
                        ConvResponseModel(self.hparams.layer_hidden, output_size, hparams)
                    )
    def forward(self, x):
        x = x[self.layer]
        s = x.shape
        if self.layer == 'x5':
            x = x.reshape(int(s[0] / self.num_frames), self.num_frames, -1)
            out = self.head(x)
        else:
            # x = x.reshape(s[0] * s[1], *s[2:])
            # print(x.shape)
            x = self.neck(x)
            x = x.reshape(int(s[0] / self.num_frames), self.num_frames, -1)
            out = self.head(x)
        out_aux = None
        out = {self.roi: out}
        return out, out_aux
