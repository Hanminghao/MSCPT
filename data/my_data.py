# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import glob
import torch.utils.data as data
import pandas as pd
import os
from PIL import Image

from torchvision import transforms
from sklearn.model_selection import train_test_split


class MyData(data.Dataset):
    def __init__(self, data_dir=r'data/ref',
                 target_size=224,
                 train=True,
                 csv_dir='',
                 seed=42,
                 dataset_name='',
                 label_dicts={},
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225), 
                 num_shots=None,
                 feat_data_dir='',
                 base_model=''):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.index_col = 'slide_id'
        self.target_col = 'OncoTreeCode'
        self.project_id = 'project_id'
        self.num_shots = num_shots
        self.seed = seed
        self.csv_dir = csv_dir
        self.label_dicts = label_dicts
        self.train = train
        self.feat_data_dir = feat_data_dir
        self.base_model = base_model
        self.roi_transforms =  transforms.Compose(
                            [
                            transforms.Resize((target_size, target_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = img_mean, std = img_std)
                            ]
                        )
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(data_dir, base_model, self.dataset_name)
        self.check_files()

    def check_files(self):
        # # This part is the core code block for load your own dataset.
        # # You can choose to scan a folder, or load a file list pickle
        # # file, or any other formats. The only thing you need to gua-
        # # rantee is the `self.path_list` must be given a valid value. 
        train_df = pd.read_csv(f'./numshots/{self.dataset_name}/{self.dataset_name}_train_{self.num_shots}_{self.seed}.csv')
        val_df = pd.read_csv(f'./numshots/{self.dataset_name}/{self.dataset_name}_val_{self.num_shots}_{self.seed}.csv')
        self.data =  train_df if self.train else val_df 


    def __len__(self):
        return len(self.data)
    
    def get_ids(self, ids):
        return self.data.loc[ids, self.index_col]

    def get_labels(self, ids):
        return self.data.loc[ids, self.target_col]
    
    def get_project_id(self, ids):
        return self.data.loc[ids, self.project_id].split('-')[-1]

    def __getitem__(self, idx):
        slide_id = self.get_ids(idx)
        label = self.get_labels(idx)
        project_id = self.get_project_id(idx)
        pt_path = os.path.join(self.feat_data_dir, project_id, self.base_model + '_20', slide_id+'.pt')
        pt_path = pt_path.replace('clip', 'ViT-B-16')
        cla_pt = torch.load(pt_path)


        if self.label_dicts is not None:
            label = self.label_dicts[label]
        cla_img = []
        nums = glob.glob(os.path.join(self.data_dir, slide_id, '*'))
        for num in nums:
            if  num[-3:] == 'png':
                img = Image.open(num).convert('RGB')
                img = self.roi_transforms(img)
                cla_img.append(img)
        cla_img = torch.stack(cla_img)#.view(len(self.label_dicts), len(tissue_types), len(nums)//2, img.shape[0], img.shape[1], img.shape[2])
        

        return (cla_img, cla_pt), label, slide_id