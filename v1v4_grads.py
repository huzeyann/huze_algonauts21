#!/usr/bin/env python
# coding: utf-8

# In[56]:


from argparse import ArgumentParser

import yaml

import torch
import torch.nn.functional as F

import pathlib

# In[2]:


from utils import *
from model_i3d import multi_resnet3d50
from dataloading import AlgonautsDataset, AlgonautsDataModule

# In[3]:


from main import LitModel
from bdcn import load_bdcn
from model_i3d import *
from i3d_flow import *

import numpy as np
from scipy import fftpack

# In[6]:


resolution = 128
d_reso = 16


# In[149]:

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--roi', type=str, default='V1')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument("--reverse", default=False, action="store_true")
    parser.add_argument('--end', type=int, default=1102)
    parser.add_argument('--task_id', type=str, default='8a14a40f93e44692b3ca8b21dafdc798')
    parser.add_argument('--save_dir', type=str, default='/mnt/v1v4/grads/')
    parser.add_argument('--ckpt_dir', type=str, default='/mnt/v1v4/ckpts/')

    args = parser.parse_args()
    return args


args = parse_args()
# In[15]:


# parser = ArgumentParser()
# parser.add_argument('--roi', type=str, default='V1')
# parser.add_argument('--device', type=int, default=1)
# parser.add_argument('--start', type=int, default=0)
# parser.add_argument("--reverse", default=False, action="store_true")
# parser.add_argument('--end', type=int, default=1102)
# parser.add_argument('--task_id', type=str, default='8a14a40f93e44692b3ca8b21dafdc798')
# parser.add_argument('--save_dir', type=str, default='/mnt/v1v4/grads/')
# parser.add_argument('--ckpt_dir', type=str, default='/mnt/v1v4/ckpts/')
#
# args = parser.parse_args([])

# In[76]:


DEVICE = args.device
# roi = args.roi
idxs = np.arange(args.start, args.end)
task_id = args.task_id

hp_path = f'csv_logs/{task_id}/default/version_0/hparams.yaml'
with open(hp_path) as f:
    hparams = yaml.safe_load(f)
rois = hparams['rois'].split(',')
assert len(rois) == 1
roi = rois[0]

# In[8]:


import json

config_file = 'voxels.json'
with open(config_file, 'r') as f:
    voxel_config = json.load(f)

config_file = 'datasets/config.json'
with open(config_file, 'rb') as f:
    sub_config = json.load(f)

# In[9]:

for k, v in voxel_config.items():
    print(k, len(v))

# In[11]:


# In[77]:


dm = AlgonautsDataModule(datasets_dir='/home/huze/algonauts_datasets/',
                         additional_features='',
                         additional_features_dir='',
                         rois=roi, num_frames=hparams['video_frames'],
                         resolution=hparams['video_size'],
                         cached=True, track='mini_track', subs='all',
                         preprocessing_type=hparams['preprocessing_type'], voxel_idxs=None)

dm.setup('test')

# In[12]:


ckpt_dir = args.ckpt_dir
ckpt_path = os.path.join(ckpt_dir, task_id)
ckpt_paths = list(pathlib.Path(ckpt_path).glob('**/*.ckpt'))
assert len(ckpt_paths) == 1
ckpt_path = ckpt_paths[0]

# In[146]:


save_dir = args.save_dir
save_dir = os.path.join(save_dir, task_id, roi)
os.makedirs(save_dir, exist_ok=True)
print('saving to ...', save_dir)

# In[13]:


backbone = modify_resnets_patrial_x_all(multi_resnet3d50(cache_dir=hparams['i3d_rgb_dir']))
plmodel = LitModel.load_from_checkpoint(ckpt_path, backbone=backbone, hparams=hparams, voxel_idxs=None)

plmodel = plmodel.to(DEVICE)


# mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# mean, std = torch.tensor(std).to(DEVICE), torch.tensor(mean).to(DEVICE)

# In[139]:


# In[78]:


def get_2d_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')


def get_2d_idct(coefficients):
    """ Get 2D Inverse Cosine Transform of Image
    """
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')


def get_reconstructed_image(raw):
    img = raw.clip(0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    return img

data_loader = dm.predict_dataloader()

num_voxels = len(voxel_config[roi])

# grads = torch.zeros((len(data_loader.dataset), num_voxels,
#                   resolution, resolution), dtype=torch.float32)

if args.reverse:
    idxs = idxs[::-1]

for idx in idxs:
    save_path = os.path.join(save_dir, f'video-{idx:04d}-dict.pkl.npy')
    if os.path.exists(save_path):
        print('skipping ...', save_path)
        continue
    print('saving to ...', save_path)

    save_dict = {
        'd_coeffs': torch.zeros((num_voxels, hparams['video_frames'], d_reso, d_reso), dtype=torch.float32),
        'mods_video_before_rescale': torch.zeros((num_voxels), dtype=torch.float32),
        'mods_video_after_rescale': torch.zeros((num_voxels), dtype=torch.float32),
        'mods_image_before_rescale': torch.zeros((num_voxels, hparams['video_frames']), dtype=torch.float32),
        'mods_image_after_rescale': torch.zeros((num_voxels, hparams['video_frames']), dtype=torch.float32),
        'medians_video': torch.zeros((num_voxels), dtype=torch.float32),
        'medians_image': torch.zeros((num_voxels, hparams['video_frames']), dtype=torch.float32),
    }

    vid = data_loader.dataset.__getitem__(idx)
    vid = torch.tensor(vid['video']).to(DEVICE).unsqueeze(0)
    vid.requires_grad = True
    out = plmodel({'video': vid})
    out = out[0][roi][0]

    for i_voxel in tqdm(range(num_voxels)):
        #         with timeit('backward'):
        loss = out[i_voxel] * 1e5
        loss.backward(retain_graph=True)

        grad = vid.grad.detach().clone()
        vid.grad = None

        # downsample
        grad = F.interpolate(grad, size=(
            grad.shape[-3], resolution, resolution))

        grad = grad[0]
        grad = grad.mean(0)  # grey
        grad_2 = grad ** 2
        save_dict['mods_video_before_rescale'][i_voxel] = torch.sqrt(grad_2.sum()).cpu()
        save_dict['mods_image_before_rescale'][i_voxel] = torch.sqrt(grad_2.sum(-1).sum(-1)).cpu()

        # median
        median = grad.median()
        save_dict['medians_video'][i_voxel] = median.cpu()
        save_dict['medians_image'][i_voxel] = grad.reshape(hparams['video_frames'], -1).median(-1)[0].cpu()

        # reject outliers
        m = 10
        d = torch.abs(grad - median)
        mdev = torch.median(d)
        mdev = mdev if mdev else 1.
        s = d / mdev
        mmax = median + m * mdev
        mmin = median - m * mdev
        grad = torch.clamp(grad, min=mmin, max=mmax)

        # rescale
        grad = 255 * (grad - mmin) / (mmax - mmin)

        grad_2 = grad ** 2
        save_dict['mods_video_after_rescale'][i_voxel] = torch.sqrt(grad_2.sum()).cpu()
        save_dict['mods_image_after_rescale'][i_voxel] = torch.sqrt(grad_2.sum(-1).sum(-1)).cpu()

        for i_frame in range(hparams['video_frames']):
            save_dict['d_coeffs'][i_voxel][i_frame] = torch.tensor(
                get_2d_dct(grad[i_frame].cpu().numpy())[:d_reso, :d_reso])

    #         if i_voxel == 10:
    #             break

    #     with open(save_path, 'wb') as f:
    #     break
    np.save(save_path, save_dict)

# In[ ]:
