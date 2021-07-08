import os
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

from pyramidpooling import SpatialPyramidPooling


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
        self.conv31 = nn.Sequential(nn.Conv3d(1024, hparams['conv_size'], kernel_size=1, stride=1), )

        if self.no_pooling:
            input_dim = hparams['conv_size'] * int(hparams['video_frames'] / 8) * \
                        int(hparams['video_size'] / 16) * int(hparams['video_size'] / 16)
        else:
            levels = np.array([[1, 2, 2], [1, 2, 4], [1, 2, 4]])
            self.pyramidpool = SpatialPyramidPooling(levels, hparams['pooling_mode'], hparams['softpool'])
            input_dim = hparams['conv_size'] * np.sum(levels[0] * levels[1] * levels[2])

        self.fc = build_fc(hparams, input_dim, hparams['output_size'])

    def forward(self, x):
        x3 = x
        x3 = self.conv31(x3)
        if not self.no_pooling:
            x3 = self.pyramidpool(x3)

        x = torch.cat([
            x3.reshape(x3.shape[0], -1),
        ], 1)

        out = self.fc(x)

        return out


def build_fc(p, input_dim, output_dim):
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leakyrelu': nn.LeakyReLU(),
        'elu': nn.ELU(),
    }

    module_list = []
    for i in range(p.get(f"num_layers")):
        if i == 0:
            in_size, out_size = input_dim, p.get(f"layer_hidden")
        else:
            in_size, out_size = p.get(f"layer_hidden"), p.get(f"layer_hidden")
        module_list.append(nn.Linear(in_size, out_size))
        if p.get('fc_batch_norm'):
            module_list.append(nn.BatchNorm1d(out_size))
        module_list.append(activations[p.get('activation')])
        module_list.append(nn.Dropout(p.get("dropout_rate")))

    if p.get(f"num_layers") == 0:
        out_size = input_dim

    # last layer
    module_list.append(nn.Linear(out_size, output_dim))

    return nn.Sequential(*module_list)


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


def modify_resnets_patrial_x3(model):
    del model.fc
    del model.last_linear
    del model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    setattr(model.__class__, 'forward', forward)
    return model


def modify_resnets_patrial_x4(model):
    del model.fc
    del model.last_linear

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    setattr(model.__class__, 'forward', forward)
    return model


def modify_resnets_patrial_x_all(model):
    del model.fc
    del model.last_linear

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

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
