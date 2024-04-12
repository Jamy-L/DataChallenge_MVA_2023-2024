# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:15:29 2024

@author: jamyl
"""


class StandardScaler():
    def __init__(self, center=True, rescale=True):
        self.center = center
        self.rescale = rescale
        
        self.f = None
    
    def fit(self, X):
        assert X.ndim == 2
        self.f = X.shape[1]
        
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        
        return self
    
    def transform(self, X):
        assert X.ndim == 2
        assert X.shape[1] == self.f, "feature shape mismatch"
        
        if self.center:
            X = X - self.mean
        
        if self.rescale:
            X = X / self.std
        
        return X
        
        
        
        
