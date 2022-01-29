#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-27-22 19:57
# @Author  : Kelley Kan HUANG (kan.huang@connect.ust.hk)

import torch


def train(epochs, model, train_data, optimizer, scheduler, device, verbose=True, print_every=10):
    """
    Run optimization to train the model.
    """
    assert train_data.split == "train"

    loss_history = []

    num_train = len(train_data)
    iterations_per_epoch = max(num_train // train_data.batch_size, 1)
    num_iterations = epochs * iterations_per_epoch

    for epoch in range(epochs):
        for iter in range(iterations_per_epoch):
            optimizer.zero_grad()

            captions, features, urls = train_data.sample()
            captions, features = torch.from_numpy(captions).long().to(
                device), torch.from_numpy(features).to(device)

            # Compute loss and gradient
            loss = model(features, captions)
            loss_history.append(loss.item())

            # Perform a parameter update
            loss.backward()
            optimizer.step()

            t = epoch * iterations_per_epoch + iter
            if verbose and t % print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, loss_history[-1]))
        scheduler.step()

    return loss_history


def main():
    pass


if __name__ == "__main__":
    main()
