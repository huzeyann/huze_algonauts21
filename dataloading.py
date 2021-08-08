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
from sklearn.model_selection import KFold

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


class RGB2BGR(torch.nn.Module):
    def forward(self, tensor):
        return torch.flip(tensor, [0])

    def __repr__(self):
        return self.__class__.__name__


class TwoFiveFive(torch.nn.Module):
    def forward(self, tensor):
        return tensor * 255

    def __repr__(self):
        return self.__class__.__name__


def wrap_load_videos(root, file_lists, num_frames=16, resolution=288, preprocessing_type='mmit'):
    # load all to memory

    # t = nn.Sequential(
    #     K.augmentation.Normalize([0.485, 0.456, 0.406],
    #                              [0.229, 0.224, 0.225]),
    #
    # )
    if preprocessing_type == 'mmit':
        resize_normalize = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif preprocessing_type == 'bdcn':
        resize_normalize = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.4810938, 0.45752459, 0.40787055], [1, 1, 1]),
            RGB2BGR(),
            TwoFiveFive(),
        ])
    else:
        NotImplementedError()

    vids = []
    for file in tqdm(file_lists):
        vid = load_video(os.path.join(root, file), num_frames, resize_normalize)
        vids.append(vid)
    vids = torch.stack(vids, 0)
    return vids


def wrap_load_one_video(root, file, num_frames=16, resolution=288, preprocessing_type='mmit'):
    if preprocessing_type == 'mmit':
        resize_normalize = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif preprocessing_type == 'bdcn':
        resize_normalize = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.4810938, 0.45752459, 0.40787055], [1, 1, 1]),
            RGB2BGR(),
            TwoFiveFive(),
        ])
    else:
        NotImplementedError()

    vid = load_video(os.path.join(root, file), num_frames, resize_normalize)
    return vid


def wrap_load_fmris(root, file_list):
    fmris = []
    for file in file_list:
        fmri = np.load(os.path.join(root, file), )
        fmris.append(fmri)
    fmris = np.stack(fmris, 0)
    fmris = torch.tensor(fmris).float()
    return fmris


class ToTensor(object):

    def __call__(self, arr):
        return torch.tensor(arr)


class ScaleTo1_1(object):

    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        return (2 * tensor / 255) - 1


class Permute(object):

    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        return tensor.permute(1, 0, 2, 3)


class AlgonautsDatasetI3dFreeze(Dataset):
    def __init__(self, dataset_dir, rois='EBA',
                 train=True, cached=True, track='mini_track', subs='all',
                 voxel_idxs=None):
        self.voxel_idxs = voxel_idxs
        self.track = track
        self.cached = cached
        self.rois = rois
        self.train = train
        self.dataset_dir = dataset_dir
        self.subs = [f'sub{i + 1:02d}' for i in range(10)] if subs == 'all' else subs.split(',')
        if self.train:
            csv = 'train_val-mini.csv' if self.track == 'mini_track' else 'train_val-full.csv'
        else:
            csv = 'full_vid.csv'
        csv_path = os.path.join(self.dataset_dir, csv)
        self.file_df = pd.read_csv(csv_path)
        self.vid_file_list = self.file_df['vid'].values
        self.vid_file_list = [x.replace('.mp4', '_flow.pkl') for x in self.vid_file_list]
        self.vid_root = os.path.join(self.dataset_dir, 'numpy')
        if self.track == 'full_track':
            assert self.rois == 'WB'

        # load fmri
        if train:
            if self.track == 'mini_track':
                fmri_dir = 'fmris-mini'
                self.fmris, self.idx_ends = concat_and_mask(
                    [wrap_load_fmris(os.path.join(self.dataset_dir, fmri_dir),
                                     self.file_df[roi].values)
                     for roi in self.rois.split(',')])
            elif self.track == 'full_track':
                fmri_dir = 'fmris-full'
                assert self.rois == 'WB'
                self.fmris, self.idx_ends = concat_and_mask(
                    [wrap_load_fmris(os.path.join(self.dataset_dir, fmri_dir), self.file_df[sub].values)
                     for sub in self.subs])

                if self.voxel_idxs is not None:
                    self.fmris = self.fmris[:, self.voxel_idxs]

            if self.track == 'full_track':
                self.sub_idx_ends = self.idx_ends
                self.idx_ends = np.array([self.fmris.shape[-1]])

    def __len__(self):
        return len(self.vid_file_list)

    def __getitem__(self, index):

        x = np.load(os.path.join(self.vid_root, self.vid_file_list[index]), allow_pickle=True)
        x = {k: torch.tensor(v).squeeze(0) for k, v in x.items()}

        if self.train:
            y = self.fmris[index]
            return x, y
        else:
            return x


class AlgonautsDataset(Dataset):
    def __init__(self, dataset_dir,
                 additional_features='',
                 additional_features_dir='',
                 rois='EBA', num_frames=16, resolution=288,
                 train=True, cached=True, track='mini_track', subs='all',
                 preprocessing_type='mmit', voxel_idxs=None):
        self.voxel_idxs = voxel_idxs
        self.track = track
        self.preprocessing_type = preprocessing_type
        self.additional_features_dir = additional_features_dir
        self.additional_features = additional_features
        self.resolution = resolution
        self.cached = cached
        self.rois = rois
        self.num_frames = num_frames
        self.train = train
        self.dataset_dir = dataset_dir
        self.subs = [f'sub{i + 1:02d}' for i in range(10)] if subs == 'all' else subs.split(',')
        if self.train:
            csv = 'train_val-mini.csv' if self.track == 'mini_track' else 'train_val-full.csv'
        else:
            csv = 'full_vid.csv'
        csv_path = os.path.join(self.dataset_dir, csv)
        self.file_df = pd.read_csv(csv_path)
        self.vid_file_list = self.file_df['vid'].values
        self.vid_root = os.path.join(self.dataset_dir, 'videos')
        if self.track == 'full_track':
            assert self.rois == 'WB'

        # load freezed layers
        if self.additional_features:
            self.additional_features = self.additional_features.split(',')
            self.features = {}
            for af in self.additional_features:
                self.features[af] = torch.tensor(
                    np.load(os.path.join(self.additional_features_dir, f'{af}.npy'))).float()
        else:
            self.additional_features = []

        if self.cached:
            self.cached_dir = os.path.join(self.dataset_dir,
                                           f'{self.resolution}_{self.num_frames}_{self.preprocessing_type}_npy')
            # save to np
            os.makedirs(self.cached_dir, exist_ok=True)
            for file in tqdm(self.vid_file_list):
                name = os.path.basename(file).replace('.mp4', '.npy')
                path = os.path.join(self.cached_dir, name)
                if not os.path.exists(path):
                    vid = wrap_load_one_video(self.vid_root, file,
                                              num_frames=self.num_frames,
                                              resolution=self.resolution,
                                              preprocessing_type=self.preprocessing_type)
                    np.save(path, vid.numpy())
        else:
            NotImplementedError()

        self.np_paths = [os.path.join(self.cached_dir, os.path.basename(f).replace('.mp4', '.npy'))
                         for f in self.vid_file_list]

        # load fmri
        if train:
            if self.track == 'mini_track':
                fmri_dir = 'fmris-mini'
                self.fmris, self.idx_ends = concat_and_mask(
                    [wrap_load_fmris(os.path.join(self.dataset_dir, fmri_dir),
                                     self.file_df[roi].values)
                     for roi in self.rois.split(',')])
            elif self.track == 'full_track':
                fmri_dir = 'fmris-full'
                assert self.rois == 'WB'
                self.fmris, self.idx_ends = concat_and_mask(
                    [wrap_load_fmris(os.path.join(self.dataset_dir, fmri_dir), self.file_df[sub].values)
                     for sub in self.subs])

                if self.voxel_idxs is not None:
                    self.fmris = self.fmris[:, self.voxel_idxs]

            if self.track == 'full_track':
                self.sub_idx_ends = self.idx_ends
                self.idx_ends = np.array([self.fmris.shape[-1]])

    def __len__(self):
        return len(self.vid_file_list)

    def __getitem__(self, index):

        vid = np.load(self.np_paths[index])

        x = {'video': vid}
        additional_features = {af: self.features[af][index] for af in self.additional_features}
        x.update(additional_features)

        if self.train:
            y = self.fmris[index]
            return x, y
        else:
            return x


class AlgonautsDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=1,
                 datasets_dir='',
                 additional_features='',
                 additional_features_dir='',
                 rois='EBA',
                 track='mini_track',
                 subs='all',
                 num_frames=16,
                 resolution=288,
                 val_ratio=0.1,
                 cached=True,
                 random_split=False,
                 use_cv=False,
                 num_split=None,
                 fold=-1,
                 preprocessing_type='mmit',
                 load_from_np=False,
                 voxel_idxs=None):
        super().__init__()
        self.voxel_idxs = voxel_idxs
        self.load_from_np = load_from_np
        self.preprocessing_type = preprocessing_type
        self.additional_features_dir = additional_features_dir
        self.additional_features = additional_features
        self.fold = fold
        self.num_split = num_split
        self.use_cv = use_cv
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
        self.file_list = ['train_val-mini.csv', 'train_val-full.csv', 'full_vid.csv']
        self.file_list = [os.path.join(self.datasets_dir, f) for f in self.file_list]
        self.batch_size = batch_size

        self.algonauts_full = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._has_setup_predict_all = True

    def prepare_data(self) -> None:
        for f in self.file_list:
            if not os.path.exists(f):
                raise Exception("dataset not ready")

    def setup(self, stage: Optional[str] = None):

        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, 'fit'):
            if not self.load_from_np:
                self.algonauts_full = AlgonautsDataset(
                    self.datasets_dir,
                    additional_features=self.additional_features,
                    additional_features_dir=self.additional_features_dir,
                    train=True,
                    rois=self.rois,
                    num_frames=self.num_frames,
                    resolution=self.resolution,
                    cached=self.cached,
                    track=self.track,
                    subs=self.subs,
                    preprocessing_type=self.preprocessing_type,
                    voxel_idxs=self.voxel_idxs,
                )
            else:
                self.algonauts_full = AlgonautsDatasetI3dFreeze(
                    self.datasets_dir,
                    train=True,
                    rois=self.rois,
                    cached=self.cached,
                    track=self.track,
                    subs=self.subs,
                    voxel_idxs=self.voxel_idxs,
                )
            self.idx_ends = self.algonauts_full.idx_ends.tolist()

            if not self.use_cv:
                num_train = int(self.train_full_len * (1 - self.val_ratio))
                num_val = int(self.train_full_len * self.val_ratio)
                if self.random_split:
                    self.train_dataset, self.val_dataset = random_split(self.algonauts_full, [num_train, num_val],
                                                                        generator=torch.Generator().manual_seed(42))
                else:
                    lengths = [num_train, num_val]
                    indices = np.arange(sum(lengths)).tolist()
                    self.train_dataset, self.val_dataset = \
                        [Subset(self.algonauts_full, indices[offset - length: offset]) for offset, length in
                         zip(_accumulate(lengths), lengths)]
            else:
                assert self.num_split > 0
                kf = KFold(n_splits=self.num_split)
                train, val = list(kf.split(np.arange(self.train_full_len)))[self.fold]
                self.train_dataset = Subset(self.algonauts_full, train)
                self.val_dataset = Subset(self.algonauts_full, val)

            self.num_voxels = self.train_dataset[0][1].shape[0]

        # Assign Test split(s) for use in Dataloaders
        if stage in (None, 'test'):
            if not self.load_from_np:
                self.test_dataset = AlgonautsDataset(
                    self.datasets_dir,
                    additional_features=self.additional_features,
                    additional_features_dir=self.additional_features_dir,
                    train=False,
                    rois=self.rois,
                    num_frames=self.num_frames,
                    resolution=self.resolution,
                    cached=self.cached,
                    track=self.track,
                    subs=self.subs,
                    preprocessing_type=self.preprocessing_type,
                    voxel_idxs=self.voxel_idxs,
                )
            else:
                self.test_dataset = AlgonautsDatasetI3dFreeze(
                    self.datasets_dir,
                    train=False,
                    rois=self.rois,
                    cached=self.cached,
                    track=self.track,
                    subs=self.subs,
                    voxel_idxs=self.voxel_idxs,
                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=8, pin_memory=False, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=8, pin_memory=False, prefetch_factor=2)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=8, pin_memory=False, prefetch_factor=2)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        if stage in (None, 'fit'):
            delattr(self, 'train_dataset')
            delattr(self, 'val_dataset')
            delattr(self, 'algonauts_full')
            setattr(self, 'train_dataset', None)
            setattr(self, 'val_dataset', None)
            setattr(self, 'algonauts_full', None)
            # self.train_dataset = None
            # self.val_dataset = None
            # self.algonauts_full = None
        if stage in (None, 'test'):
            delattr(self, 'test_dataset')
            setattr(self, 'test_dataset', None)
            # self.test_dataset = None


if __name__ == '__main__':
    d = AlgonautsDataset('datasets/')
    a = AlgonautsDataModule(32, 'datasets/')
    b = AlgonautsDataModule(32, 'datasets_full/', track='full_track')
    a.setup()
    pass
