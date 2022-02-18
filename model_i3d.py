import os
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

from pyramidpooling3d import SpatialPyramidPooling3D


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1),
        out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = torch.cat([out.data, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1
    Conv3d = staticmethod(conv3x3x3)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.Conv3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    Conv3d = nn.Conv3d

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = self.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = self.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    Conv3d = nn.Conv3d

    def __init__(self, block, layers, shortcut_type='B', num_classes=305):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        self.conv1 = self.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.init_weights()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    self.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, self.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class MiniFC(nn.Module):

    def __init__(self, hparams):
        super(MiniFC, self).__init__()
        self.no_pooling = True if hparams['pooling_mode'] == 'no' else False
        self.global_pooling = hparams['global_pooling']
        if hparams['backbone_type'] == 'x3':
            conv_indim = 1024
        elif hparams['backbone_type'] == 'x4':
            conv_indim = 2048
        elif hparams['backbone_type'] == 'x2':
            conv_indim = 512
        else:
            raise Exception("?")
        self.conv = nn.Sequential(nn.Conv3d(conv_indim, hparams['conv_size'], kernel_size=1, stride=1), )

        if hparams['global_pooling']:
            self.global_avgpool = nn.AdaptiveAvgPool3d(1)
            input_dim = hparams['conv_size']
        else:
            if self.no_pooling:
                if hparams['backbone_type'] == 'x3':
                    input_dim = hparams['conv_size'] * int(hparams['video_frames'] / 8) * \
                                int(hparams['video_size'] / 16) * int(hparams['video_size'] / 16)
                elif hparams['backbone_type'] == 'x4':
                    input_dim = hparams['conv_size'] * int(hparams['video_frames'] / 16) * \
                                int(hparams['video_size'] / 32) * int(hparams['video_size'] / 32)
                elif hparams['backbone_type'] == 'x2':
                    input_dim = hparams['conv_size'] * int(hparams['video_frames'] / 4) * \
                                int(hparams['video_size'] / 8) * int(hparams['video_size'] / 8)
            else:
                if hparams['backbone_type'] == 'x3':
                    levels = np.array([[1, 2, 2], [1, 2, 4], [1, 2, 4]])
                elif hparams['backbone_type'] == 'x4':
                    levels = np.array([[1, 1, 1], [1, 2, 3], [1, 2, 3]])
                elif hparams['backbone_type'] == 'x2':
                    levels = np.array([[1, 2, 4], [1, 2, 4], [1, 2, 4]])
                self.pyramidpool = SpatialPyramidPooling3D(levels, hparams['pooling_mode'])
                input_dim = hparams['conv_size'] * np.sum(levels[0] * levels[1] * levels[2])

        self.fc = build_fc(hparams, input_dim, hparams['output_size'])

    def forward(self, x):
        x = self.conv(x)
        if not self.global_pooling:
            if not self.no_pooling:
                x = self.pyramidpool(x)
                # x = torch.cat([v for k, v in x.items()], 1)
        else:
            x = self.global_avgpool(x)

        x = torch.cat([
            x.reshape(x.shape[0], -1),
        ], 1)

        out = self.fc(x)

        return out


def build_fc(p, input_dim, output_dim, part='full'):
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leakyrelu': nn.LeakyReLU(),
        'elu': nn.ELU(),
    }

    if part == 'full':
        layer_hidden = p.get(f"layer_hidden")
    elif part == 'first':
        layer_hidden = p.get(f"first_layer_hidden")
        layer_hidden = layer_hidden if layer_hidden > 0 else p.get(f"layer_hidden")
    elif part == 'last':
        layer_hidden = p.get(f"layer_hidden")
    else:
        NotImplementedError()

    module_list = []
    for i in range(p.get(f"num_layers")):
        if i == 0:
            in_size, out_size = input_dim, layer_hidden
        else:
            in_size, out_size = layer_hidden, layer_hidden
        module_list.append(nn.Linear(in_size, out_size))
        if p.get('fc_batch_norm'):
            module_list.append(nn.BatchNorm1d(out_size))
        module_list.append(activations[p.get('activation')])
        module_list.append(nn.Dropout(p.get("dropout_rate")))
        if i == 0:
            layer_one_size = len(module_list)

    if p.get(f"num_layers") == 0:
        out_size = input_dim

    # last layer
    module_list.append(nn.Linear(out_size, output_dim))

    if part == 'full':
        module_list = module_list
    elif part == 'first':
        module_list = module_list[:layer_one_size]
    elif part == 'last':
        module_list = module_list[layer_one_size:]
    else:
        NotImplementedError()

    return nn.Sequential(*module_list)

class FcFusion(nn.Module):
    def __init__(self, fusion_type='concat'):
        super(FcFusion, self).__init__()
        assert fusion_type in ['add', 'avg', 'concat', ]
        self.fusion_type = fusion_type

    def init_weights(self):
        pass

    def forward(self, input):
        assert (isinstance(input, tuple)) or (isinstance(input, dict)) or (isinstance(input, list))
        if isinstance(input, dict):
            input = tuple(input.values())

        if self.fusion_type == 'add':
            out = torch.sum(torch.stack(input, -1), -1, keepdim=False)

        elif self.fusion_type == 'avg':
            out = torch.mean(torch.stack(input, -1), -1, keepdim=False)

        elif self.fusion_type == 'concat':
            out = torch.cat(input, -1)

        else:
            raise ValueError

        return out


class ConvResponseModel(nn.Module):

    def __init__(self, in_dim, out_dim, hparams):
        super(ConvResponseModel, self).__init__()
        self.layer_hidden = hparams['layer_hidden']
        self.fc = nn.Sequential(
            nn.Linear(in_dim, self.layer_hidden),
            nn.ELU(),
            nn.Linear(self.layer_hidden, 1024 * 4 * 5 * 4),
            nn.ELU()
        )
        if hparams['convtrans_bn']:
            self.convt = nn.Sequential(
                nn.ConvTranspose3d(1024, 512, (3, 3, 3), (2, 2, 2)),
                nn.BatchNorm3d(512),
                nn.ELU(),
                nn.ConvTranspose3d(512, 256, (3, 3, 3), (2, 2, 2)),
                nn.BatchNorm3d(256),
                nn.ELU(),
                nn.ConvTranspose3d(256, 128, (3, 3, 3), (2, 2, 2)),
                nn.BatchNorm3d(128),
                nn.ELU(),
                nn.ConvTranspose3d(128, out_dim, (3, 3, 3), (2, 2, 2)),
            )
        else:
            self.convt = nn.Sequential(
                nn.ConvTranspose3d(1024, 512, (3, 3, 3), (2, 2, 2)),
                # nn.BatchNorm3d(512),
                nn.ELU(),
                nn.ConvTranspose3d(512, 256, (3, 3, 3), (2, 2, 2)),
                # nn.BatchNorm3d(256),
                nn.ELU(),
                nn.ConvTranspose3d(256, 128, (3, 3, 3), (2, 2, 2)),
                # nn.BatchNorm3d(128),
                nn.ELU(),
                nn.ConvTranspose3d(128, out_dim, (3, 3, 3), (2, 2, 2)),
            )

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], 1024, 4, 5, 4)
        x = self.convt(x)
        return x


class ConvFusion(nn.Module):
    def __init__(self, num_voxels, num_chs, fusion_type='concat', detach=False):
        super(ConvFusion, self).__init__()
        self.detach = detach
        assert fusion_type in ['concat', 'conv', 'conv_voxel']
        self.fusion_type = fusion_type

        if fusion_type == 'conv_voxel':
            self.weight = torch.nn.Parameter(data=torch.rand(num_voxels, num_chs), requires_grad=True)
        elif fusion_type == 'conv':
            self.weight = torch.nn.Parameter(data=torch.rand(num_chs), requires_grad=True)
        elif fusion_type == 'concat':
            self.fc = nn.Sequential(nn.ELU(),
                                    nn.Linear(num_voxels * num_chs, num_voxels))

    def init_weights(self):
        pass

    def forward(self, input):
        assert (isinstance(input, tuple)) or (isinstance(input, dict)) or (isinstance(input, list))
        if isinstance(input, dict):
            input = tuple(input.values())

        if self.detach:
            input = tuple(i.clone().detach() for i in input)

        if self.fusion_type == 'concat':
            out = torch.cat(input, -1)
            out = out.reshape(out.shape[0], -1)
            out = self.fc(out)

        elif self.fusion_type == 'conv' or self.fusion_type == 'conv_voxel':
            out = torch.stack(input, -1)
            out = (out * self.weight).mean(-1)

        else:
            raise ValueError

        return out


class I3d_neck(nn.Module):

    def __init__(self, hparams):
        super(I3d_neck, self).__init__()
        self.hparams = hparams

        def roundup(x):
            return int(np.ceil(x))

        if self.hparams.backbone_type == 'i3d_rgb':
            video_size = hparams['video_size'] if hparams['crop_size'] == 0 else hparams['crop_size']
            self.x1_twh = (roundup(hparams['video_frames'] / 2), roundup(video_size / 4), roundup(video_size / 4))
            self.x2_twh = tuple(map(lambda x: roundup(x / 2), self.x1_twh))
            self.x3_twh = tuple(map(lambda x: roundup(x / 2), self.x2_twh))
            self.x4_twh = tuple(map(lambda x: roundup(x / 2), self.x3_twh))
            self.x1_c, self.x2_c, self.x3_c, self.x4_c = 256, 512, 1024, 2048
        elif self.hparams.backbone_type == 'i3d_flow':
            # assert self.hparams.load_from_np
            # self.x1_twh = (32, 28, 28)
            # self.x2_twh = (16, 14, 14)
            # self.x3_twh = (8, 7, 7)
            # self.x4_twh = (8, 7, 7)
            video_frames = hparams['video_frames']
            video_size = hparams['video_size'] if hparams['crop_size'] == 0 else hparams['crop_size']
            # print(video_size)
            self.x1_twh = (roundup(video_frames / 2), roundup(video_size / 8), roundup(video_size / 8))
            self.x2_twh = (roundup(self.x1_twh[0] / 2), roundup(self.x1_twh[1] / 2), roundup(self.x1_twh[2] / 2))
            self.x3_twh = (roundup(self.x2_twh[0] / 2), roundup(self.x2_twh[1] / 2), roundup(self.x2_twh[2] / 2))
            # print(self.x3_twh)
            self.x4_twh = (roundup(self.x2_twh[0] / 2), roundup(self.x2_twh[1] / 2), roundup(self.x2_twh[2] / 2))
            self.x1_c, self.x2_c, self.x3_c, self.x4_c = 192, 480, 832, 1024
        else:
            NotImplementedError()
        self.twh_dict = {'x1': self.x1_twh, 'x2': self.x2_twh, 'x3': self.x3_twh, 'x4': self.x4_twh}
        self.c_dict = {'x1': self.x1_c, 'x2': self.x2_c, 'x3': self.x3_c, 'x4': self.x4_c}
        self.planes = hparams['conv_size']
        self.pyramid_layers = hparams['pyramid_layers'].split(',')  # x1,x2,x3,x4
        self.pyramid_layers.sort()
        self.pathways = hparams['pathways'].split(',')  # ['topdown', 'bottomup'] aka 'parallel', or "none"
        self.is_pyramid = False if self.pathways[0] == 'none' else True
        self.aux_heads = True if hparams['aux_loss_weight'] > 0 else False
        self.old_mix = hparams['old_mix']
        if hparams['separate_rois']:
            self.rois = hparams['rois'].split(',')
            self.output_sizes = hparams['roi_lens']
        else:
            self.rois = [hparams['rois']]
            self.output_sizes = [hparams['output_size']]
        self.pooling_modes = {  # pooling_mode in ['no', 'spp', 'avg']
            'x1': hparams['x1_pooling_mode'],
            'x2': hparams['x2_pooling_mode'],
            'x3': hparams['x3_pooling_mode'],
            'x4': hparams['x4_pooling_mode'],
        }
        self.spp_level_dict = {
            'x1': np.array([hparams['spp_size_t_x1'], hparams['spp_size_x1'], hparams['spp_size_x1']]),
            'x2': np.array([hparams['spp_size_t_x2'], hparams['spp_size_x2'], hparams['spp_size_x2']]),
            'x3': np.array([hparams['spp_size_t_x3'], hparams['spp_size_x3'], hparams['spp_size_x3']]),
            'x4': np.array([hparams['spp_size_t_x4'], hparams['spp_size_x4'], hparams['spp_size_x4']]),
        }
        if self.is_pyramid:
            assert len(self.pathways) >= 1 and self.pathways[0] != 'none'

        self.first_convs = nn.ModuleDict()
        self.smooths = nn.ModuleDict()
        self.poolings = nn.ModuleDict()
        self.fc_input_dims = {}
        self.ch_response = nn.ModuleDict()
        self.final_fusions = nn.ModuleDict()

        self.is_x_label = False
        if 'x_label' in self.pyramid_layers:
            self.pyramid_layers.remove('x_label')
            self.is_x_label = True

        self.num_chs = len(self.pyramid_layers) * len(self.pathways)
        for roi, output_size in zip(self.rois, self.output_sizes):
            for x_i in self.pyramid_layers:
                for pathway in self.pathways:
                    k = f'{roi}_{pathway}_{x_i}'

                    if x_i == 'x5':  # i3d_flow
                        self.first_convs.update({k: nn.Flatten()})
                        self.poolings.update({k: nn.Flatten()})
                        if self.old_mix:
                            self.ch_response.update({k: build_fc(hparams, 1024, output_size, part='first')})
                        else:
                            self.ch_response.update({k: build_fc(hparams, 1024, output_size)})
                        # print(self.ch_response)
                        continue

                    self.first_convs.update(
                        {k: nn.Conv3d(self.c_dict[x_i], self.planes, kernel_size=1, stride=1)})

                    if self.is_pyramid:
                        self.smooths.update(
                            {k: nn.Conv3d(self.planes, self.planes, kernel_size=3, stride=1, padding='same')})

                    if self.pooling_modes[x_i] == 'no':
                        self.poolings.update({k: nn.Flatten()})
                        self.fc_input_dims.update({k: self.planes * np.product(self.twh_dict[x_i])})
                    elif self.pooling_modes[x_i] == 'avg':
                        self.poolings.update({k: nn.Sequential(
                            nn.AdaptiveAvgPool3d(1),
                            nn.Flatten())})
                        self.fc_input_dims.update({k: self.planes})
                    elif self.pooling_modes[x_i] == 'max':
                        self.poolings.update({k: nn.Sequential(
                            nn.AdaptiveMaxPool3d(1),
                            nn.Flatten())})
                        self.fc_input_dims.update({k: self.planes})
                    elif self.pooling_modes[x_i] == 'spp':
                        self.poolings.update({k: nn.Sequential(
                            SpatialPyramidPooling3D(self.spp_level_dict[x_i],
                                                    hparams['pooling_mode']),
                            nn.Flatten())})
                        self.fc_input_dims.update({k: np.sum(
                            self.spp_level_dict[x_i][0] * self.spp_level_dict[x_i][1] * self.spp_level_dict[x_i][
                                2]) * self.planes})
                    elif self.pooling_modes[x_i] == 'adaptive_max':
                        size = hparams[f'pooling_size_{x_i}']
                        size_t = hparams[f'pooling_size_t_{x_i}']
                        self.poolings.update({k: nn.Sequential(
                            nn.AdaptiveMaxPool3d((size_t, size, size)),
                            nn.Flatten())})
                        self.fc_input_dims.update({k: self.planes * size_t * size * size})
                    elif self.pooling_modes[x_i] == 'adaptive_avg':
                        size = hparams[f'pooling_size_{x_i}']
                        size_t = hparams[f'pooling_size_t_{x_i}']
                        self.poolings.update({k: nn.Sequential(
                            nn.AdaptiveAvgPool3d((size_t, size, size)),
                            nn.Flatten())})
                        self.fc_input_dims.update({k: self.planes * size_t * size * size})
                    else:
                        NotImplementedError()

                    if self.old_mix:
                        self.ch_response.update({k: build_fc(
                            hparams, self.fc_input_dims[k], output_size, part='first')})
                    else:
                        if self.hparams['track'] == 'full_track':
                            if self.hparams['no_convtrans']:
                                self.ch_response.update({k: build_fc(
                                    hparams, self.fc_input_dims[k], output_size)})
                            else:
                                self.ch_response.update(
                                    {k: ConvResponseModel(self.fc_input_dims[k], hparams['num_subs'], hparams)})
                        else:
                            self.ch_response.update({k: build_fc(
                                hparams, self.fc_input_dims[k], output_size)})

        if self.old_mix:
            in_size = self.hparams.first_layer_hidden * self.num_chs if hparams['final_fusion'] == 'concat' \
                else self.hparams.first_layer_hidden
            for roi, output_size in zip(self.rois, self.output_sizes):
                if self.hparams['track'] == 'full_track':
                    if self.hparams['no_convtrans']:
                        self.final_fusions.update({f'{roi}': nn.Sequential(
                            FcFusion(fusion_type=hparams['final_fusion']),
                            build_fc(self.hparams, in_size, output_size),
                        )})
                    else:
                        self.final_fusions.update({f'{roi}': nn.Sequential(
                            FcFusion(fusion_type=hparams['final_fusion']),
                            ConvResponseModel(in_size, hparams['num_subs'], hparams)
                        )})
                else:
                    self.final_fusions.update({f'{roi}': nn.Sequential(
                        FcFusion(fusion_type=hparams['final_fusion']),
                        build_fc(self.hparams, in_size, output_size),
                    )})
        else:
            for roi, output_size in zip(self.rois, self.output_sizes):
                self.final_fusions.update({f'{roi}': ConvFusion(
                    num_voxels=output_size, num_chs=self.num_chs,
                    fusion_type=hparams['final_fusion'], detach=hparams['detach_aux'])})

    def forward(self, x):

        # if self.is_x_label:
        #     x_label = x['x_label']

        # vid
        out = {}
        for roi in self.rois:
            for x_i in self.pyramid_layers:
                for pathway in self.pathways:
                    k = f'{roi}_{pathway}_{x_i}'
                    out[k] = x[x_i].clone()
        x = out
        # x = {f'{pathway}_{x_i}': x[x_i] for x_i in self.pyramid_layers for pathway in self.pathways}
        x = {k: self.first_convs[k](v) for k, v in x.items()}
        self.x_npooled = x
        if self.is_pyramid:
            x = self.pyramid_pathway(x, self.pyramid_layers, self.pathways)
        x = {k: self.poolings[k](v) for k, v in x.items()}
        self.x_pooled = x
        # fmri
        x = {k: self.ch_response[k](v) for k, v in x.items()}
        # print(self.ch_response)
        # print(self.final_fusions)

        out = {}
        for roi in self.rois:
            roi_out_auxs = []
            for k, v in x.items():
                if roi not in k:
                    continue
                roi_out_auxs.append(v)
            roi_out = self.final_fusions[roi](roi_out_auxs)
            out[roi] = roi_out

        out_aux = None if self.old_mix or self.hparams.track == 'full_track' else x
        # for x_i in self.pyramid_layers:
        #     for pathway in self.pathways:
        #         roi_out_auxs = []
        #         for roi in self.rois:
        #             k = f'{roi}_{pathway}_{x_i}'
        #             roi_out_auxs.append(x[k])
        #         roi_out_auxs = torch.cat(roi_out_auxs, -1)
        #         out_aux[f'{pathway}_{x_i}'] = roi_out_auxs

        return out, out_aux

    def pyramid_pathway(self, x, layers, pathways):
        for roi in self.rois:
            for pathway in pathways:

                if pathway == 'bottomup':
                    layers_iter = layers
                elif pathway == 'topdown':
                    layers_iter = reversed(layers)
                elif pathway == 'none':
                    continue
                else:
                    NotImplementedError()

                for i, x_i in enumerate(layers_iter):
                    k = f'{roi}_{pathway}_{x_i}'
                    if i == 0:
                        pass
                    else:
                        x[k] = self.resample_and_add(prev, x[k])
                    x[k] = self.smooths[k](x[k])
                    prev = x[k]
        return x

    @staticmethod
    def resample_and_add(x, y):
        target_shape = y.shape[2:]
        out = F.interpolate(x, size=target_shape, mode='nearest')
        return out + y


def modify_resnets(model):
    # Modify attributs
    model.last_linear, model.fc = model.fc, None

    def features(self, input):
        x = self.conv1(input)
        # print("conv, ", x.view(-1)[:10])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # torch.Size([1, 64, 8, 56, 56])

        x = self.layer1(x)  # torch.Size([1, 256, 8, 56, 56])
        x = self.layer2(x)  # torch.Size([1, 512, 4, 28, 28])
        x = self.layer3(x)  # torch.Size([1, 1024, 2, 14, 14])
        x = self.layer4(x)  # torch.Size([1, 2048, 1, 7, 7])
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    setattr(model.__class__, 'features', features)
    setattr(model.__class__, 'logits', logits)
    setattr(model.__class__, 'forward', forward)
    return model


def modify_resnets_patrial_x_all(model):
    # del model.fc
    # del model.last_linear

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x_label = self.logits(x4)

        return {
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'x_label': x_label,
        }

    setattr(model.__class__, 'forward', forward)
    return model


ROOT_URL = 'http://moments.csail.mit.edu/moments_models'
weights = {
    'resnet50': 'moments_v2_RGB_resnet50_imagenetpretrained.pth.tar',
    'resnet3d50': 'moments_v2_RGB_imagenet_resnet3d50_segment16.pth.tar',
    'multi_resnet3d50': 'multi_moments_v2_RGB_imagenet_resnet3d50_segment16.pth.tar',
}


def load_checkpoint(weight_file):
    if not os.access(weight_file, os.W_OK):
        weight_url = os.path.join(ROOT_URL, weight_file)
        os.system('wget ' + weight_url)
    checkpoint = torch.load(weight_file, map_location=lambda storage, loc: storage)  # Load on cpu
    return {str.replace(str(k), 'module.', ''): v for k, v in checkpoint['state_dict'].items()}


def resnet50(num_classes=305, pretrained=True):
    model = models.__dict__['resnet50'](num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_checkpoint(weights['resnet50']))
    model = modify_resnets(model)
    return model


def resnet3d50(num_classes=305, pretrained=True, **kwargs):
    """Constructs a ResNet3D-50 model."""
    model = modify_resnets(ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs))
    if pretrained:
        model.load_state_dict(load_checkpoint(weights['resnet3d50']))
    return model


def multi_resnet3d50(num_classes=292, pretrained=True, cache_dir='~/.cache/', **kwargs):
    """Constructs a ResNet3D-50 model."""
    model = modify_resnets(ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs))
    if pretrained:
        model.load_state_dict(load_checkpoint(os.path.join(cache_dir, weights['multi_resnet3d50'])))
    return model


def load_model(arch):
    model = {'resnet3d50': resnet3d50,
             'multi_resnet3d50': multi_resnet3d50, 'resnet50': resnet50}.get(arch, 'resnet3d50')()
    model.eval()
    return model


def load_transform():
    """Load the image transformer."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])


def load_categories(filename):
    """Load categories."""
    with open(filename) as f:
        return [line.rstrip() for line in f.readlines()]
