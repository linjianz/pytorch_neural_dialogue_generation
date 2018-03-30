#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-27 21:32:50
Program: 
Description: 
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from tensorboardX import SummaryWriter
from dataset.opensubdata import OpenSub, OpenSubDataSet
from net.seq2seq_attention import Seq2Seq
from utils.misc import adjust_lr, mk_dir_train, display_loss, to_var
from utils.conf import get_parser


args = get_parser()


def run_batch(sample, model, optimizer, loss_func, phase='Train'):
    if phase == 'Train':
        model.train()
    else:
        model.eval()

    source = to_var(sample['source'].transpose(0, 1))       # T x B
    target = to_var(sample['target'].transpose(0, 1))       # T x B
    loss = 0  # Added onto for each word
    time_step, _ = tuple(target.size())

    # Run words through encoder
    # TODO: notice that must use model.module to call class method in nn.DataParallel
    encoder_hidden = model.module.init_hidden(args.batch_size)
    encoder_outputs, encoder_hidden = model.module.encoder(source, encoder_hidden)

    decoder_context = to_var(torch.zeros(args.batch_size, args.hidden_size))
    decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

    # Choose whether to use teacher forcing
    # use_teacher_forcing = random.random() < args.tf_ratio
    use_teacher_forcing = True
    count = 0
    if use_teacher_forcing:
        # Teacher forcing: Use the ground-truth target as the next input
        for i in range(time_step-1):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = model(target[i, :], decoder_context,
                                                                                       decoder_hidden, encoder_outputs)
            # decoder_output: B x Vocab, target: B x T
            loss += loss_func(decoder_output, target[i+1, :])
            count += 1
            # if target[i+1] == args.EOS_ID:
            #     break

    else:
        # Without teacher forcing: use network's own prediction as the next input
        decoder_input = target[0]   # SOS_ID
        for i in range(time_step-1):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = model(decoder_input, decoder_context,
                                                                                       decoder_hidden, encoder_outputs)
            loss += loss_func(decoder_output[0], target[i])
            _, top_id = decoder_output.data.topk(1)
            ni = top_id[0][0]
            decoder_input = to_var(torch.LongTensor([[ni]]))  # Chosen word is next input
            count += 1
            if ni == args.EOS_ID:
                break

    # BP
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()

    return loss.data[0] / count


def train():
    print('\n')
    print('Create Encoder and Decoder Model'.center(100, '='))
    torch.set_default_tensor_type('torch.FloatTensor')
    model = Seq2Seq(attn_model='general',
                    vocab_size=args.vocab_size,
                    input_size=args.vec_size,
                    hidden_size=args.hidden_size,
                    layer_num=args.layer_num)
    print(model)
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda(), device_ids=args.gpu, dim=1)
        # model.cuda()
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base)

    print('Load Data'.center(100, '='))
    data_set = OpenSub(args)
    data_set_train = OpenSubDataSet(data_set.sources_train, data_set.targets_train)
    data_set_valid = OpenSubDataSet(data_set.sources_valid, data_set.targets_valid)
    dir_model_date, dir_log_date = mk_dir_train(args)
    writer = SummaryWriter(dir_log_date)

    print('Prepare data loader')
    loader_train = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    step_per_epoch = data_set_train.__len__() // loader_train.batch_size
    loss_best = -1
    epoch_best = 0
    epoch_current = 0

    print('Start Training'.center(100, '='))
    while True:
        adjust_lr(optimizer, epoch_current, args.lr_base, args.lr_decay_rate, args.epoch_lr_decay)
        loss_list = []
        for step, sample_batch in enumerate(loader_train):
            step_global = epoch_current * step_per_epoch + step
            tic = time()
            loss = run_batch(sample=sample_batch,
                             model=model,
                             optimizer=optimizer,
                             loss_func=loss_func,
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
    train()
