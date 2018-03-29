#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-29 16:45:26
Program: 
Description: 
"""
from torch.utils.data import DataLoader
from dataset.opensubdata import OpenSub, OpenSubDataSet
from utils.conf import get_parser


# TODO 1. 获取generator输入数据（source/target）
# TODO 2. 由generator产生answer，从而有（source,target,1）（source,answer,0）两组数据
# TODO 3. 由discriminator判别上述数据真假，计算误差（不要reward），更新D
# TODO 4. 重复1/2/3，保留reward，不更新D，在（source,answer）上更新G
# TODO 5. 在（source, target）上更新G


def train():
    data_set = OpenSub(path=args.dir_data, args=args)
    data_set_train = OpenSubDataSet(data_set.sources_train, data_set.targets_train)

    print('Prepare data loader')
    loader_train = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=True, num_workers=4)


if __name__ == '__main__':
    args = get_parser()
    train()
