# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:45:28 2024

@author: jamyl
"""

import torch

def to_image(X):
    h = w = 32
    N_images = X.shape[0]
    assert X.shape[1] == h*w*3
    return X.reshape(N_images, 3, h, w).transpose((0, 2, 3, 1))


def accuracy(pred, gt):
    assert pred.ndim == 1
    assert gt.ndim == 1
    correct = (pred == gt).sum().item()
    return correct/pred.shape[0]


def hinge_loss(predictions, targets):
    device=predictions.device
    return torch.mean(torch.max(1 - targets * predictions, torch.zeros(1, device=device)))


def train_test_split(x, y, test_size=0.2, random_state=None):
    assert x.shape[0] == y.shape[0]
    # Set random seed for reproducibility
    if random_state is not None:
        torch.manual_seed(random_state)

    # Calculate the lengths of the train and test sets
    total_size = len(x)
    test_size = int(test_size * total_size)
    train_size = total_size - test_size

    # Use random permutation to shuffle indices
    indices = torch.randperm(total_size)

    # Split indices into train and test sets
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Use fancy indexing to get train and test tensors
    x_train = x[train_indices]
    x_test = x[test_indices]
    
    y_train = y[train_indices]
    y_test = y[test_indices]

    return x_train, x_test, y_train, y_test

def rescale_pixel_value(X):
    """
    Put back values of pixel in [0,1]
    """
    max_ = X.max(dim=-1, keepdims=True)[0].max(dim=-2, keepdims=True)[0].max(dim=-3, keepdims=True)[0]
    min_ = X.min(dim=-1, keepdims=True)[0].min(dim=-2, keepdims=True)[0].max(dim=-3, keepdims=True)[0]
    return ((X - min_) / (max_ - min_))
