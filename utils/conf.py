#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-29 20:13:34
Program: 
Description: 
"""
import os
import re
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server',         default=6099, type=int, help='[6099 / 6199 / 6499]')
    parser.add_argument('--net_name',       default='seq2seq', help='[seq2seq]')
    parser.add_argument('--dir_date',       default='20180330', help='Name it with date, such as 20180102')
    parser.add_argument('--batch_size',     default=128, type=int, help='Batch size')
    parser.add_argument('--lr_base',        default=1e-3, type=float, help='Base learning rate')
    parser.add_argument('--lr_decay_rate',  default=0.1, type=float, help='Decay rate of lr')
    parser.add_argument('--epoch_lr_decay', default=1000, type=int, help='Every # epoch, lr decay lr_decay_rate')

    parser.add_argument('--layer_num',      default=2, type=int, help='Lstm layer number')
    parser.add_argument('--vocab_size',     default=25003, type=int, help='Vocabulary size')
    parser.add_argument('--vec_size',       default=256, type=int, help='word embedding size')
    parser.add_argument('--hidden_size',    default=256, type=int, help='Lstm hidden units (the same as vec_size)')
    parser.add_argument('--tf_ratio',       default=0.5, type=float, help='Lstm hidden units')
    parser.add_argument('--clip',           default=5.0, type=float, help='clip')
    parser.add_argument('--gpu',            default='0,1', help='GPU id list')
    parser.add_argument('--workers',        default=4, type=int, help='Workers number')
    parser.add_argument('--PAD_ID',         default=0, type=int, help='pad id')
    parser.add_argument('--SOS_ID',         default=25001, type=int, help='start of sentence id')
    parser.add_argument('--EOS_ID',         default=25002, type=int, help='end of sentence id')

    parsers = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = parsers.gpu        # set visible gpu list, eg: '2,3,4'
    gpu_list = re.split('[, ]', parsers.gpu)                # store the gpu id into a list
    parsers.gpu = range(len(list(filter(None, gpu_list))))  # gpu for PyTorch

    # data file
    parsers.dir_data = '/media/csc105/Data/dataset-jiange/nlp/OpenSubData/OpenSubData'
    if parsers.server == 6199:
        parsers.dir_data = '/media/Data/dataset_jiange/OpenSubData/OpenSubData'

    # model file and log file
    dir_project = '/home/jiange/project/pytorch_neural_dialogue_generation'
    parsers.dir_model = dir_project + '/model'  # directory to save model
    parsers.dir_log = dir_project + '/log'  # directory to save log

    return parsers
