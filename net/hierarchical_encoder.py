#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-26 21:19:46
Program: 
Description: 
"""
import torch
import torch.nn as nn
import numpy as np
from utils.misc import to_var


def fc(in_features, out_features, activation=None):
    if activation == 'relu':
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
        )
    elif activation == 'sigmoid':
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
        )
    else:
        return nn.Linear(in_features, out_features),


class HierarchicalEncoder(nn.Module):
    def __init__(self, vocab_size, batch_size, input_size, hidden_size, layer_num=1):
        super(HierarchicalEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        # Define sentence layers
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm_encoder = nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=layer_num,
                                    batch_first=False,  # must pay attention here!!!
                                    dropout=0.5,
                                    bidirectional=False)

        # Define context layers
        self.lstm_decoder = nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=layer_num,
                                    batch_first=False,
                                    dropout=0.5,
                                    bidirectional=False)
        self.sigmoid = fc(hidden_size, 2, 'sigmoid')

    def encoder_sentence(self, words_input):
        """
        :param words_input: T x B
        :return:
            que_output: T x B x N
            hidden: layer*direction x B x N
        """
        embedded = self.embedding(words_input)
        h_0 = to_var(torch.zeros(self.layer_num, self.batch_size, self.hidden_size))
        c_0 = to_var(torch.zeros(self.layer_num, self.batch_size, self.hidden_size))
        output, hidden = self.lstm_encoder(embedded, (h_0, c_0))

        return hidden

    def encoder_context(self, vec_input):
        h_0 = to_var(torch.zeros(self.layer_num, self.batch_size, self.hidden_size))
        c_0 = to_var(torch.zeros(self.layer_num, self.batch_size, self.hidden_size))
        output, hidden = self.lstm_encoder(vec_input, (h_0, c_0))

        return hidden

    def forward(self, query, answer):
        """

        :param query: T1 x B
        :param answer: T2 x B
        :return:
            logits: B x Class
        """
        que_hidden = self.encoder_sentence(query)
        ans_hidden = self.encoder_sentence(answer)
        que_hidden = torch.unsqueeze(que_hidden[-1][0], dim=0)
        ans_hidden = torch.unsqueeze(ans_hidden[-1][0], dim=0)
        context_input = torch.cat((que_hidden, ans_hidden), dim=0)  # 2 x B x N
        con_hidden = self.encoder_context(context_input)
        out_hidden = con_hidden[-1][1]  # B x N
        logits = self.sigmoid(out_hidden)   # B x C

        return logits


if __name__ == '__main__':
    # TODO 准备数据：1. query [None, 30], 2. answer [None, 32], 3. label [None, 1]
    model = HierarchicalEncoder(vocab_size=1000,
                                batch_size=8,
                                input_size=256,
                                hidden_size=256,
                                layer_num=2)
    model.cuda()
    que_batch = to_var(torch.from_numpy(np.arange(8 * 30).reshape(30, 8)))
    ans_batch = to_var(torch.from_numpy(np.arange(8 * 30).reshape(30, 8)))
    label_batch = to_var(torch.IntTensor([0, 1, 1, 1, 0, 0, 0, 1]))
    logits = model(que_batch, ans_batch)
    print(logits.size())
