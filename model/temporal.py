#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-17-22 19:42
# @Author  : Kelley Kan HUANG (kan.huang@connect.ust.hk)

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAffine(nn.Module):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """

    def __init__(self, in_features, out_features):
        super(TemporalAffine, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        N, T, D = x.shape
        assert D == self.in_features
        M = self.out_features

        out = self.fc(x.reshape(N * T, D)).reshape(N, T, M)

        return out


class TemporalSoftmaxLoss(nn.Module):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    """

    def __init__(self):
        super(TemporalSoftmaxLoss, self).__init__()

    def forward(self, x, y, mask):

        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        # Method 1
        # Tensor.max return a tuple(values, index) when dim is given
        probs = F.softmax(x_flat-x_flat.max(dim=1, keepdim=True)
                          [0], dim=1)
        # Method 2
        # probs = (x_flat - x_flat.max(dim=1, keepdim=True)[0]).exp()
        # probs /= probs.sum(dim=1, keepdims=True)

        loss = -torch.sum(mask_flat *
                          torch.log(probs[torch.arange(N * T), y_flat])) / N

        return loss


def TemporalAffine_test():
    N, T, D, M = 2, 3, 4, 5
    my_temporal_affine = TemporalAffine(D, M)
    x = torch.randn(N, T, D)
    out = my_temporal_affine(x)
    print(out.shape)


def TemporalSoftmaxLoss_test(N, T, V, p):
    my_temporal_softmax_loss = TemporalSoftmaxLoss()

    x = 0.001 * torch.randn(N, T, V)
    y = torch.randint(V, size=(N, T))
    mask = torch.rand(N, T) <= p

    loss = my_temporal_softmax_loss(x, y, mask)
    print(loss)


def main():
    TemporalSoftmaxLoss_test(100, 1, 10, 1.0)   # Should be about 2.3
    TemporalSoftmaxLoss_test(100, 10, 10, 1.0)  # Should be about 23
    TemporalSoftmaxLoss_test(5000, 10, 10, 0.1)  # Should be about 2.3


if __name__ == "__main__":
    main()
