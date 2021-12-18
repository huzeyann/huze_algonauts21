#!/usr/bin/env python
# coding: utf-8

# In[1]:
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

# In[6]:


resolution = 128


# In[149]:


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--roi', type=str, default='STS')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument("--reverse", default=False, action="store_true")
    parser.add_argument('--end', type=int, default=1102)
    parser.add_argument('--save_dir', type=str, default='/data_60/huze/algonauts_grad_data/')
    parser.add_argument('--ckpt_dir', type=str, default='/data_smr/huze/projects/my_algonauts/checkpoints_mmit_ft/')

    args = parser.parse_args()
    return args

args = parse_args()
DEVICE = args.device
roi = args.roi
idxs = np.arange(args.start, args.end)

# In[8]:


import json

config_file = 'voxels.json'
with open(config_file, 'r') as f:
    voxel_config = json.load(f)

config_file = 'datasets/config.json'
with open(config_file, 'rb') as f:
    sub_config = json.load(f)

task_ids = {
    'rgb': ['3e036ee5f38241eabf926fd256ff8fb7', '1816ebcf8ea44ff0bb39c250f7e121ab', 'fcd1a62399e94b0ab7b2a63aa6462134', '9f1cb4529c5b4857b2fd43a20c78e10c', 'bdf95214335d4faf8adb566091a083d6', 'ebe86bc42eef4f0f99234f99e3ced141', '5d92210e1e76412d900b88c304bfe7d6', '727c694f4af6453aa616d4930a122a03', '7d99866ea5d04840b5a0bff850d30dfc', '8393ca02bdc74959ae7d9ae25e0a928e', 'be23db02f6f34f7aa54318b21eea438b', '28f3d82a55db4e6f83a30b88a2a3b280', '7f8ed4a2c45448b1b838d6da801bc645', '7bbc61572cfb425b8a337bf5dddc522e', '705a81f6407446eab3e32a6549b2f763', '2df2ef5b658646568caf045ade308239', '7e06202be27d48a6b588971807bb5bbf', 'f81e364f88994ab6a27d30d70e5a75b2', '81dc884509734b03afc466c7a946e959', '3d665d6a55294b818c495e13bbfb4a56'],
    'flow': ['4ad05bd0eee34f17a1162947f2b14f13', 'e9eb47a215a44233bad664589aeb5a5e', '6d99c2a8aa2e401487338952f7feddab', 'f438db719d204f5c906a5c64583ada39', 'f4911905b7a44cb88b2d87adc3814392', '05fbfc635bd649189256e4e59b31d1e6', '3f48240f0fb04d63a65a373499a365cc', '2e60ea4ef1c24fa0a5ba4ea715089fd9', 'db325fb4e82e4f16acee31f46d53a2bf', 'ddda0215e2994739b373676221076957', '57c7de745db040dfb8da2a39fb6d6f6b', '7b5422a70b734abc892b59acde0f804f', 'bf88af28a31b4b35be83ace1df438236', '0174fb97a96d4649803bfe22c09993a8', '1d867eff108d4317aa6e2f094c2074e7', '756cfe0da2384e8a821416650726c7ee', 'fef8cb85a7bc4fcea9f04041f8b95b01', '1f1a21df75bd4bcf8c081e47bf11b294', 'ace27c879edc487e8852dc7590ef647a', 'f8f18ea6d3214bebacf920f79d7eef04'],
    'edge': ['4ba56f1b24274c78b1ea9c33753083eb', '1cfa3c9d98db43dda8be8262bf04d583', 'aa63f41a1f024017bf884dcdd6699bd0', '05bac82b84c9449cae09808f198e9085', 'cd335f283f814ea58c3733c23bb17a8a', '9377dcb1721c428cae229cf103daf188', '3e254c4a015f44518f87883d1bc3f7f7', '56fc6241a4d44e35b1290bd7a49e2c69', '9cc3f5e855d34df0b3411c9184f67a39', '311d49de08e549dc9f37ebd7985b0e16', '272877db4cb6457c8f2a679156c00837', '00d1a6c79fc34befbc1ab7f4a3cca8c3', '182f7132fc3f4bdcaaa91c6c5b7817e9', '32f919e3f9ab4be5bb93ad9a0b957cf0', '4e274c2ded064b4eba93f4201da9801f', 'bc4bd42bc70d409daebe27f2c0fae255', '0cde9741eb7c4b6f95da73eb1e0ae31d', 'dc6f14f3aecd4ed287c913250fd2ea60', 'a1e1bc2e759b4a54a50bcfd8da98aa0a', 'c89c8f670d6c4c85ac43d2393e690b24'],
}


# In[9]:


for k, v in voxel_config.items():
    print(k, len(v))


# In[11]:


dm = AlgonautsDataModule(datasets_dir='/home/huze/algonauts_datasets/',
                 additional_features='',
                 additional_features_dir='',
                 rois='WB', num_frames=16, resolution=288,
                 cached=True, track='full_track', subs='all',
                 preprocessing_type='mmit', voxel_idxs=None)

dm.setup('test')


# In[12]:


task_id = 'ebe86bc42eef4f0f99234f99e3ced141'


ckpt_dir = args.ckpt_dir
hp_path = f'csv_logs/{task_id}/default/version_0/hparams.yaml'
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


with open(hp_path) as f:
    hparams = yaml.safe_load(f)

backbone = modify_resnets_patrial_x_all(multi_resnet3d50(cache_dir=hparams['i3d_rgb_dir']))
plmodel = LitModel.load_from_checkpoint(ckpt_path, backbone=backbone, hparams=hparams, voxel_idxs=None)

plmodel = plmodel.to(DEVICE)
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
mean, std = torch.tensor(std).to(DEVICE), torch.tensor(mean).to(DEVICE)


# In[139]:


data_loader = dm.predict_dataloader()

num_voxels = len(voxel_config[roi])

# grads = torch.zeros((len(data_loader.dataset), num_voxels,
#                   resolution, resolution), dtype=torch.float32)

if args.reverse:
    idxs = idxs[::-1]

for idx in idxs:
    save_path_tmax = os.path.join(save_dir, f'video{idx}-tmax.npy')
    # save_path_full = os.path.join(save_dir, f'video{idx}-full.npy')

    # if os.path.exists(save_path_tmax) and os.path.exists(save_path_full):
    if os.path.exists(save_path_tmax):
        print('skipping ...', save_path_tmax)
        continue
    print('saving to ...', save_path_tmax)
    
    grads = torch.zeros((num_voxels, resolution, resolution), dtype=torch.float32)
    # full_grads = torch.zeros((num_voxels, hparams['video_frames'], resolution, resolution), dtype=torch.float32)

    vid = data_loader.dataset.__getitem__(idx)
    vid = torch.tensor(vid['video']).to(DEVICE).unsqueeze(0)
    vid.requires_grad = True
    out = plmodel({'video': vid})
    out = out[0]['WB'][0]
    
    for i_voxel in tqdm(range(num_voxels)):
#         with timeit('backward'):
        loss = out[voxel_config[roi][i_voxel]] * 1e5
        loss.backward(retain_graph=True)

        grad = vid.grad.detach().clone()
        vid.grad = None

        # downsample
        grad = F.interpolate(grad, size=(
            grad.shape[-3], resolution, resolution))
        
        grad = grad[0]
        grad = grad.permute(1, 2, 3, 0)
        grad = grad * std + mean
        grad = grad.mean(-1) # grey
#         grad = reject_outliers_torch(grad, m=3)
        grad = grad.cpu()
    
        grads[i_voxel] = grad.max(0)[0] # max-pooling
        # full_grads[i_voxel] = grad

#         if i_voxel == 10:
#             break

    with open(save_path_tmax, 'wb') as f:
        np.save(f, grads.numpy())
    # with open(save_path_full, 'wb') as f:
    #     np.save(f, full_grads.numpy())


# In[ ]:




