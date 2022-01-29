#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-28-22 11:38
# @Author  : Kelley Kan HUANG (kan.huang@connect.ust.hk)

import numpy as np
from torch.utils.data import Dataset
from datasets.cs231n.coco_utils import load_coco_data, sample_coco_minibatch


class COCODataset(Dataset):
    def __init__(self, base_dir, batch_size, max_train=None, pca_features=True, seed=231, split='train'):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.split = split
        np.random.seed(seed)
        self.data = load_coco_data(
            base_dir=base_dir, max_train=max_train, pca_features=pca_features)

    def __len__(self):
        split_size = self.data['%s_captions' % self.split].shape[0]
        return split_size

    def __getitem__(self, idx):
        raise NotImplementedError

    def sample(self):
        # Make a minibatch of training data
        minibatch = sample_coco_minibatch(self.data,
                                          batch_size=self.batch_size,
                                          split=self.split)
        captions, features, urls = minibatch

        return captions, features, urls
