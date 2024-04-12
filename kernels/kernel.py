# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:23:27 2024

@author: jamyl
"""

import torch
import torch.nn as nn
import math

class Kernel(nn.Module):
    def __init__(self, gamma):
        super(Kernel, self).__init__()
        self.gamma = gamma
        
    def forward(self, x1, x2):
        raise NotImplementedError

    def forward_dist(self, dist):
        raise NotImplementedError
        
    def kernel_matrix(self, x1, x2, gamma=None):
        assert x1.ndim == x2.ndim == 2
        n, f = x1.shape
        m, f_ = x2.shape
        assert f == f_, "Feature size mismatch" 
        
        return self.forward(x1, x2, gamma)
        
    
        
        
        