#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-27-22 18:31
# @Author  : Kelley Kan HUANG (kan.huang@connect.ust.hk)

import torch
import torch.nn as nn
from temporal import TemporalAffine, TemporalSoftmaxLoss


class CaptioningRNN(nn.Module):
    def __init__(self,
                 word_to_idx,
                 input_dim=512,
                 wordvec_dim=128,
                 hidden_dim=128,
                 cell_type='rnn'):
        super(CaptioningRNN, self).__init__()
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.word_to_idx = word_to_idx
        self.input_dim = input_dim
        self.wordvec_dim = wordvec_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        self.temporal_affine_1 = TemporalAffine(input_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim)
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        if cell_type == "rnn":
            self.rnn = nn.RNN(wordvec_dim, hidden_dim, dim_mul)
        self.temporal_affine_2 = TemporalAffine(hidden_dim, vocab_size)

        self.softmax_loss = TemporalSoftmaxLoss()

    def forward(self, features, captions):
        """forward
        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) 
        """
        captions_in = captions[:, :-1]  # (N, T)
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        N, D = features.shape

        # Initial hidden state
        h_prev = self.temporal_affine_1(features.reshape(
            N, 1, D))  # (N, 1, H)
        h_prev = h_prev.transpose(0, 1)  # (N, 1, H) -> (1, N, H)

        # Use a word embedding layer to transform the words in captions_in from indices to vectors, giving an array of shape (N, T, W).
        word_vectors = self.embedding(captions_in)  # word_vectors, (N, T, W)
        # Must transpose first before input into nn.RNN
        word_vectors = word_vectors.transpose(0, 1)  # (N, T, W) -> (T, N, W)

        # process the sequence of input word vectors and produce hidden state vectors for all timesteps
        (T, N, W) = word_vectors.shape
        H = self.hidden_dim
        h = torch.empty(T, N, H).to(
            next(self.parameters()).device)  # (T, N, H)
        for i in range(T):
            # step once
            output, h_next = self.rnn(word_vectors[i].unsqueeze(0), h_prev)
            h[i] = h_next
            h_prev = h_next
        h = h.transpose(0, 1)  # (T, N, H) -> (N, T, H)

        scores = self.temporal_affine_2(h)

        loss = self.softmax_loss(scores, captions_out, mask)

        return loss


def main():
    """Test CaptioningRNN
    """
    import numpy as np

    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    N, D, W, H = 32, 20, 30, 40

    word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
    V = len(word_to_idx)
    T = 13  # max_length

    batch_size = N
    input_dim = D
    timesteps = T
    hidden_dim = H
    vocab_size = V
    wordvec_dim = W

    model = CaptioningRNN(word_to_idx,
                          input_dim=input_dim,
                          wordvec_dim=wordvec_dim,
                          hidden_dim=hidden_dim,
                          cell_type='rnn')
    model.to(device)

    # captions: List[Int]
    captions = torch.randint(vocab_size, size=(
        batch_size, timesteps)).to(device)
    features = torch.randn(batch_size, input_dim).to(device)

    loss = model(features, captions)
    print(loss.item())


if __name__ == "__main__":
    main()
