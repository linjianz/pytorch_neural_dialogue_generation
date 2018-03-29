#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-27 18:28:46
Program: 
Description:
"""
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class OpenSub(object):
    def __init__(self, args):
        self.path = args.dir_data
        self.PAD_ID = args.PAD_ID
        self.SOS_ID = args.SOS_ID
        self.EOS_ID = args.EOS_ID
        self.max_len_s = 30
        self.max_len_t = 32
        self.sources_train, self.targets_train, self.sources_valid, self.targets_valid = self.load_data()
        assert len(self.sources_train) == len(self.targets_train), 'Length of sources and targets must be equal!'

    def load_data(self):
        """
        source: pad with 0, len is 30 [0 0 ... 0 0 source]
        target: pad with 1/2, len is 32 [1 target 2 0 0]
        """
        with open(self.path+'/t_given_s_dialogue_length2_6.txt', 'r') as f:
            source_target = f.read().strip().split('\n')
        sources = []
        targets = []
        for _, s_t in enumerate(tqdm(source_target)):
            s_t = s_t.split('|')
            source = [int(w) for w in s_t[0].split()]
            source = [self.PAD_ID] * (self.max_len_s - len(source)) + source
            sources.append(source)

            target = [int(w) for w in s_t[1].split()]
            target = [self.SOS_ID] + target + [self.EOS_ID] + [self.PAD_ID] * (self.max_len_t - len(target) - 2)
            targets.append(target)

        num = len(sources)
        interval = num * 9 // 10
        sources_train = np.array(sources[: interval])
        sources_valid = np.array(sources[interval:])
        targets_train = np.array(targets[: interval])
        targets_valid = np.array(targets[interval:])

        return sources_train, targets_train, sources_valid, targets_valid


class OpenSubDataSet(Dataset):
    def __init__(self, x, y):
        self.sources = x
        self.targets = y

    def __len__(self):

        return len(self.sources)

    def __getitem__(self, item):
        sample = dict()
        sample['source'] = self.sources[item]
        sample['target'] = self.targets[item]

        return sample


if __name__ == '__main__':
    from utils.conf import get_parser
    parsers = get_parser()

    data_set = OpenSub(parsers)
    data_set_train = OpenSubDataSet(data_set.sources_train, data_set.targets_train)
    data_set_valid = OpenSubDataSet(data_set.sources_valid, data_set.targets_valid)
    loader_train = DataLoader(data_set_train, batch_size=256, shuffle=False, num_workers=4)
    n_batch = data_set_train.__len__() // loader_train.batch_size
    for i_batch, sample_batch in enumerate(loader_train):
        print('{}/{} source batch: {}, target batch: {}'.format(i_batch, n_batch, sample_batch['source'].size(),
                                                                sample_batch['target'].size()))
