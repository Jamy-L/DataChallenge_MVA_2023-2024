# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:25:00 2024

@author: jamyl
"""

import torch
import torch.nn as nn
from .kernel import Kernel

class LinearKernel(Kernel):
    def __init__(self, gamma=None):
        super(LinearKernel, self).__init__(gamma)

    def forward(self, x1, x2, gamma_=None):
        return x1 @ x2.T
    