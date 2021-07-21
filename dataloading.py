import os
import types
import collections
import numpy as np
from random import shuffle
import torch
import pandas as pd
import pytorch_lightning as pl
from typing import Optional

from PIL import Image
from torch._utils import _accumulate
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, random_split, DataLoader, Subset
from torchvision.datasets import MNIST

import kornia as K

import decord
from decord import VideoReader, cpu, gpu

# decord.bridge.set_bridge('torch')
from utils import concat_and_mask


def load_video(file, num_frames, load_transform):
    vr = VideoReader(file, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int)
    # vid = vr.get_batch(indices)
    # vid = vid.moveaxis(-1, 1)
    # vid = load_transform(vid / 255)
    # vid = vid.moveaxis(0, 1)
    images = []
    for seg_ind in indices:
        images.append(load_transform(Image.fromarray(vr[seg_ind].asnumpy())))
    vid = torch.stack(images, 0)
    vid = vid.moveaxis(0, 1)
    return vid


def wrap_load_videos(root, file_lists, num_frames=16, resolution=288):
    # load all to memory

    # t = nn.Sequential(
    #     K.augmentation.Normalize([0.485, 0.456, 0.406],
    #                              [0.229, 0.224, 0.225]),
    #
    # )
    resize_normalize = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    vids = []
    for file in tqdm(file_lists):
        vid = load_video(os.path.join(root, file), num_frames, resize_normalize)
        vids.append(vid)
    vids = torch.stack(vids, 0)
    return vids


def wrap_load_fmris(root, file_list):
    fmris = []
    for file in file_list:
        fmri = np.load(os.path.join(root, file), )
        fmris.append(fmri)
    fmris = np.stack(fmris, 0)
    fmris = torch.tensor(fmris).float()
    return fmris


class AlgonautsDataset(Dataset):
    def __init__(self, dataset_dir, rois='EBA', num_frames=16, resolution=288,
                 train=True, cached=True, track='mini_track', subs='all'):
        self.resolution = resolution
        self.cached = cached
        self.rois = rois
        self.num_frames = num_frames
        self.train = train
        self.dataset_dir = dataset_dir
        self.subs = [f'sub{i + 1:02d}' for i in range(10)] if subs == 'all' else subs.split(',')
        csv_path = os.path.join(self.dataset_dir, 'train_val.csv' if train else 'predict.csv')
        self.file_df = pd.read_csv(csv_path)
        self.track = track
        if self.track == 'full_track':
            assert self.rois == 'WB'

        # load
        if self.cached:  # this can get big
            cache_dir = '/home/huze/.cache/'
            cache_file = cache_dir + f'videos_{self.num_frames}_{self.resolution}_{self.train}.pt'
            if os.path.exists(cache_file):
                self.videos = torch.load(cache_file)
            else:
                self.videos = wrap_load_videos(os.path.join(self.dataset_dir, 'videos'),
                                               self.file_df['vid'].values,
                                               self.num_frames, self.resolution)
                torch.save(self.videos, cache_file)
        else:
            self.videos = wrap_load_videos(os.path.join(self.dataset_dir, 'videos'),
                                           self.file_df['vid'].values,
                                           self.num_frames, self.resolution)
        if train:
            if self.track == 'mini_track':
                self.fmris = []
                for roi in self.rois.split(','):
                    fmri = wrap_load_fmris(os.path.join(self.dataset_dir, 'fmris'),
                                           self.file_df[roi].values)
                    self.fmris.append(fmri)
                self.fmris, self.idx_ends = concat_and_mask([wrap_load_fmris(os.path.join(self.dataset_dir, 'fmris'),
                                                                             self.file_df[roi].values)
                                                             for roi in self.rois.split(',')])
            elif self.track == 'full_track':
                self.fmris, self.idx_ends = concat_and_mask(
                    [wrap_load_fmris(os.path.join(self.dataset_dir, 'fmris'), self.file_df[sub].values)
                     for sub in self.subs])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        if self.train:
            return self.videos[index], self.fmris[index]
        else:
            return self.videos[index]


class AlgonautsDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=1,
                 datasets_dir='',
                 rois='EBA',
                 track='mini_track',
                 subs='all',
                 num_frames=16,
                 resolution=288,
                 val_ratio=0.1,
                 cached=True,
                 random_split=False):
        super().__init__()
        self.subs = subs
        self.track = track
        self.random_split = random_split
        self.cached = cached
        self.resolution = resolution
        self.val_ratio = val_ratio
        self.num_frames = num_frames
        self.rois = rois
        self.datasets_dir = datasets_dir
        self.train_full_len = 1000
        self.file_list = ['train_val.csv', 'predict.csv']
        self.file_list = [os.path.join(self.datasets_dir, f) for f in self.file_list]
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        for f in self.file_list:
            if not os.path.exists(f):
                raise Exception("dataset not ready")

    def setup(self, stage: Optional[str] = None):

        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, 'fit'):
            algonauts_full = AlgonautsDataset(
                self.datasets_dir,
                train=True,
                rois=self.rois,
                num_frames=self.num_frames,
                resolution=self.resolution,
                cached=self.cached,
                track=self.track,
                subs=self.subs
            )
            num_train = int(self.train_full_len * (1 - self.val_ratio))
            num_val = int(self.train_full_len * self.val_ratio)
            if self.random_split:
                self.train_dataset, self.val_dataset = random_split(algonauts_full, [num_train, num_val],
                                                                    generator=torch.Generator().manual_seed(42))
            else:
                lengths = [num_train, num_val]
                indices = np.arange(sum(lengths)).tolist()
                self.train_dataset, self.val_dataset = \
                    [Subset(algonauts_full, indices[offset - length: offset]) for offset, length in
                     zip(_accumulate(lengths), lengths)]
            self.num_voxels = self.train_dataset[0][1].shape[0]

        # Assign Test split(s) for use in Dataloaders
        if stage in (None, 'test'):
            self.test_dataset = AlgonautsDataset(
                self.datasets_dir,
                train=False,
                rois=self.rois,
                num_frames=self.num_frames,
                resolution=self.resolution,
                cached=self.cached,
                track=self.track,
                subs=self.subs
            )
            self.num_voxels = getattr(self, 'num_voxels')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=8, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=8, pin_memory=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=8, pin_memory=False)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...


if __name__ == '__main__':
    d = AlgonautsDataset('datasets/')
    a = AlgonautsDataModule(32, 'datasets/')
    b = AlgonautsDataModule(32, 'datasets_full/', track='full_track')
    a.setup()
    pass
