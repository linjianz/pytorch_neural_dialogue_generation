#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-25 21:52:02
Program: 
Description: 
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import re
import argparse
import numpy as np
from time import time
from tensorboardX import SummaryWriter
from dataset.discdata import DiscDataSet
from net.hierarchical_encoder import HierarchicalEncoder
from utils.misc import adjust_learning_rate, pre_create_file_train, display_loss, to_var


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server',         default=None, type=int, help='[6099 / 6199 / 6499]')
    parser.add_argument('--net_name',       default=None, help='[seq2seq]')
    parser.add_argument('--dir_date',       default=None, help='Name it with date, such as 20180102')
    parser.add_argument('--batch_size',     default=256, type=int, help='Batch size')
    parser.add_argument('--lr_base',        default=1e-3, type=float, help='Base learning rate')
    parser.add_argument('--lr_decay_rate',  default=0.1, type=float, help='Decay rate of lr')
    parser.add_argument('--epoch_lr_decay', default=1000, type=int, help='Every # epoch, lr decay lr_decay_rate')

    parser.add_argument('--layer_num',      default=2, type=int, help='Lstm layer number')
    parser.add_argument('--vocab_size',     default=25003, type=int, help='Vocabulary size')
    parser.add_argument('--vec_size',       default=256, type=int, help='word embedding size')
    parser.add_argument('--hidden_size',    default=256, type=int, help='Lstm hidden units (the same as vec_size)')
    parser.add_argument('--tf_ratio',       default=0.5, type=float, help='Lstm hidden units')
    parser.add_argument('--clip',           default=5.0, type=float, help='clip')
    parser.add_argument('--gpu',            default='1', help='GPU id list')
    parser.add_argument('--workers',        default=4, type=int, help='Workers number')
    parser.add_argument('--PAD_ID',         default=0, type=int, help='pad id')
    parser.add_argument('--SOS_ID',         default=25001, type=int, help='start of sentence id')
    parser.add_argument('--EOS_ID',         default=25002, type=int, help='end of sentence id')

    return parser.parse_args()


def run_batch(sample, model, optimizer, loss_func, args, phase='Train'):
    if phase == 'Train':
        model.train()
    else:
        model.eval()

    query = to_var(sample['query'])
    answer = to_var(sample['answer'])
    label = to_var(sample['label'])
    logits = model(query, answer)
    loss = loss_func(logits, label)

    # BP
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()

    return loss.data[0]


def train(args):
    print('\n')
    print('Create Hierarchical Encoder Model'.center(100, '='))
    torch.set_default_tensor_type('torch.FloatTensor')
    model = HierarchicalEncoder(vocab_size=args.vocab_size,
                                batch_size=args.batch_size,
                                input_size=args.vec_size,
                                hidden_size=args.hidden_size,
                                layer_num=args.layer_num)
    print(model)
    if torch.cuda.is_available():
        model.cuda()

    loss_func = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base)

    print('Load Data'.center(100, '='))
    # TODO 1. 训练generator，2. 生成(query, answer)，3. 重写dataset/discdata.py
    data_set = HierarchicalEncoder()
    dir_model_date, dir_log_date = pre_create_file_train(dir_model, dir_log, args)
    writer = SummaryWriter(dir_log_date)

    print('Prepare data loader')
    loader_train = DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    step_per_epoch = data_set.__len__() // loader_train.batch_size
    loss_best = -1
    epoch_best = 0
    epoch_current = 0

    print('Start Training'.center(100, '='))
    while True:
        adjust_learning_rate(optimizer, epoch_current, args.lr_base, args.lr_decay_rate, args.epoch_lr_decay)
        loss_list = []
        for step, sample_batch in enumerate(loader_train):
            step_global = epoch_current * step_per_epoch + step
            tic = time()
            loss = run_batch(sample=sample_batch,
                             model=model,
                             optimizer=optimizer,
                             loss_func=loss_func,
                             args=args,
                             phase='Train')
            hour_per_epoch = step_per_epoch * ((time() - tic) / 3600)
            loss_list.append(loss)

            # display result and add to tensor board
            if (step + 1) % 1 == 0:
                display_loss(hour_per_epoch, epoch_current, args, step, step_per_epoch, optimizer, loss,
                             loss_list, writer, step_global)
        loss_mean = np.mean(loss_list)
        epoch_current += 1
        if loss_mean < loss_best or loss_best == -1:
            loss_best = loss_mean
            epoch_best = epoch_current
            torch.save(model.state_dict(), dir_model_date + '/model-best.pkl')
            print('>>>save current best model in {:s}\n'.format(dir_model_date + '/model-best.pkl'))
        else:
            if epoch_current - epoch_best == 5:
                break


if __name__ == '__main__':
    parser_args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu  # set visible gpu list, eg: '2,3,4'
    gpu_list = re.split('[, ]', parser_args.gpu)  # store the gpu id into a list
    parser_args.gpu = range(len(list(filter(None, gpu_list))))  # gpu for PyTorch

    dir_data = '/media/csc105/Data/dataset-jiange/nlp/OpenSubData/OpenSubData'
    if parser_args.server == 6199:
        dir_data = '/media/Data/dataset_jiange/OpenSubData/OpenSubData'
    dir_project = '/home/jiange/project/pytorch_neural_dialogue_generation'
    dir_model = dir_project + '/model'  # directory to save model
    dir_log = dir_project + '/log'  # directory to save log

    train(parser_args)
