#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-28 21:00:01
Program: 
Description: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.misc import to_var


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, outputs):
        """
        input:
            hidden: output of decoder just one time step > B x N
            encoder_outputs: outputs of encoder > T_encoder x B x N
        return:
            weights: T x B
        """
        ts, bs, _ = tuple(outputs.size())

        # Create variable to store attention energies
        attn_energies = to_var(torch.zeros(bs, ts))

        # Calculate energies for each encoder output
        for i in range(ts):
            for j in range(bs):     # 整个batch下hidden与encoder前面所有时刻的输出计算分数
                attn_energies[j, i] = self.score(hidden[j, :], outputs[i, j, :])

        # Normalize energies to weights in range 0 to 1, resize to B x 1 x T
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        # hidden: N, encoder_output: N
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class Seq2Seq(nn.Module):
    """
    encoder和decoder共用embedding，所以把两个放同一个class里好管理
    """
    def __init__(self, attn_model, vocab_size, input_size, hidden_size, layer_num=1):
        super(Seq2Seq, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        # Define encoder layers
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.gru_encoder = nn.GRU(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=layer_num,
                                  batch_first=False,  # must pay attention here!!!
                                  dropout=0.5,
                                  bidirectional=False)

        # Define decoder layers
        # self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru_decoder = nn.GRU(input_size=hidden_size*2,
                                  hidden_size=hidden_size,
                                  num_layers=layer_num,
                                  batch_first=False,
                                  dropout=0.5,
                                  bidirectional=False)
        self.out = nn.Linear(hidden_size * 2, vocab_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, outputs_encoder):
        """ Note: we run this one step at a time
        Input:
            word_input: one time step of decoder input (word index) > B
            last_context: init zeros > B x N
            last_hidden: init is encoder's last hidden > layers*direction x B x N
            outputs_encoder: T_encoder x B x N
        Return:
            output: B x N
        """
        word_embedded = self.embedding(word_input.unsqueeze(0)).contiguous()     # 1 x B x N
        last_context = last_context.contiguous()

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)       # 1 x B x 2N
        rnn_output, hidden = self.gru_decoder(rnn_input, last_hidden)   # rnn_output: 1 x B x N, hidden: l*d x B x N

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), outputs_encoder)    # output: B x 1 x T
        context = attn_weights.bmm(outputs_encoder.transpose(0, 1))  # B x 1 x T (bmm) B x T x N -> B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)      # 1 x B x N -> B x N
        context = context.squeeze(1)            # B x 1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), dim=1)

        return output, context, hidden, attn_weights

    def encoder(self, words_input, hidden):
        """Note: we run this all at once (over the whole input sequence)
        Input:
            words_input: words sequence > T x B
            hidden: init hidden state > layer*direction x B x N
        Return:
            output: all time step > T x B x N
            hidden: last time step state > layer*direction x B x N
        """
        word_embedded = self.embedding(words_input).contiguous()
        hidden = hidden.contiguous()
        output, hidden = self.gru_encoder(word_embedded, hidden)

        return output, hidden

    def init_hidden(self, batch_size):
        hidden = to_var(torch.zeros(self.layer_num, batch_size, self.hidden_size))

        return hidden

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.FloatTensor')
    model = Seq2Seq(attn_model='general',
                    vocab_size=25003,
                    input_size=256,
                    hidden_size=256,
                    layer_num=2)
    print(model)
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda(), device_ids=[0, 1])

    encoder_hidden = model.module.init_hidden(128)
    source = to_var(torch.from_numpy(np.arange(128*30).reshape(30, 128)))
    encoder_outputs, encoder_hidden = model.module.encoder(source, encoder_hidden)
    print(encoder_outputs.size(), encoder_hidden.size())
