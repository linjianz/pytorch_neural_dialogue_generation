#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-29 14:33:53
Program: 
Description: 
"""
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class DiscDataSet(Dataset):
    def __init__(self, x1, x2, y):
        self.query = x1
        self.answer = x2
        self.label = y

    def __len__(self):

        return len(self.query)

    def __getitem__(self, item):
        sample = dict()
        sample['query'] = self.query[item]
        sample['answer'] = self.answer[item]
        sample['label'] = self.label[item]

        return sample


if __name__ == '__main__':
    print('Hello world')
