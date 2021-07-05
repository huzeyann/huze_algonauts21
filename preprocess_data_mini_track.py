import glob
import os
import numpy as np
from utils import load_fmri
import json
from tqdm import tqdm
import pandas as pd

e_args = {
    'base_fmri_dir': '/data_smr/huze/projects/Algonauts2021/participants_data_v2021',
    'video_dir': '/data_smr/huze/projects/Algonauts2021/AlgonautsVideos268_All_30fpsmax/',
    'rois': ['LOC', 'FFA', 'STS', 'EBA', 'PPA', 'V1', 'V2', 'V3', 'V4'],
    'subs': ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08', 'sub09', 'sub10'],
}

num_train = 1000
num_predict = 102

dataset_dir = './datasets'
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
fmri_dir = os.path.join(dataset_dir, 'fmris')
if not os.path.exists(fmri_dir):
    os.mkdir(fmri_dir)
video_dir = os.path.join(dataset_dir, 'videos')
if not os.path.exists(video_dir):
    os.mkdir(video_dir)

video_list = glob.glob(e_args['video_dir'] + '/*.mp4')
video_list.sort()

# load for each roi
subs = e_args['subs']
roi_fmri_dict = {}
roi_sub_lens_dict = {}
for roi in e_args['rois']:
    all_fmri, roi_keys, roi_idx, roi_lens = load_fmri(e_args['base_fmri_dir'], [roi], subs)
    roi_fmri_dict[roi] = all_fmri
    roi_sub_lens_dict[roi] = roi_lens

# save
roi_list_dict = {roi:[] for roi in e_args['rois']}
for i in tqdm(range(num_train + num_predict)):

    if i < num_train:
        # fmri
        for roi, fmri in roi_fmri_dict.items():
            base_name = f"{i + 1:04d}_{roi}.npy"
            np.save(os.path.join(fmri_dir, base_name), fmri[i])
            roi_list_dict[roi].append(base_name)
    # save vid
    os.system(f"cp {video_list[i]} {video_dir}")

# roi_lens
with open(os.path.join(dataset_dir, 'config.json'), 'w') as fp:
    json.dump(roi_sub_lens_dict, fp)

video_list = [os.path.basename(v) for v in video_list]
train_video_list = video_list[:num_train]
predict_video_list = video_list[num_train:num_train + num_predict]

roi_list_dict.update({'vid': train_video_list})
train_df = pd.DataFrame(roi_list_dict)
predict_df = pd.DataFrame({'vid': predict_video_list})

train_df.to_csv(os.path.join(dataset_dir, 'train_val.csv'))
predict_df.to_csv(os.path.join(dataset_dir, 'predict.csv'))

pass
