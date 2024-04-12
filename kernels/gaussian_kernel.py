# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:25:00 2024

@author: jamyl
"""

import torch
import torch.nn as nn
from .kernel import Kernel

class GaussianKernel(Kernel):
    def __init__(self, gamma):
        super(GaussianKernel, self).__init__(gamma)
        self.gamma = gamma

    def forward(self, x1, x2, gamma_=None):
        if isinstance(self.gamma, float):
            gamma_ = self.gamma
        elif gamma_ is None:
            raise ValueError(f"Must specify gamma during inference of mode {self.gamma}")
        
        norm_x1 = x1.square().sum(axis=1, keepdims=True)
        norm_x2 = x2.square().sum(axis=1, keepdims=True)
        dist_matrix_sq = norm_x1 - 2 * x1 @ x2.T + norm_x2.T
        dist_matrix_sq = dist_matrix_sq.clip(min=0)
        
        return (- gamma_ * dist_matrix_sq).exp()