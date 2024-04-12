# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:15:29 2024

@author: jamyl
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import accuracy, hinge_loss

from tqdm import tqdm

EPS_SUPPORT = 1e-8

class BinaryKLR(nn.Module):
    def __init__(self,
                 kernel: str="rbf",
                 gamma="scale",
                 C: float=1.):
        super(BinaryKLR, self).__init__()
        
        assert gamma in ["scale", "auto"] or isinstance(gamma, float)
        self.gamma = gamma
        
        self.C=C
        
        kernel = self.import_kernel(kernel)
        
        self.kernel = kernel(gamma=gamma)
        
        self.input_size = None
        self.Xtrain = None
        self.alphas = None
        
    def import_kernel(self, kernel_name):
        if kernel_name == 'rbf':
            from kernels.gaussian_kernel import GaussianKernel
            kernel = GaussianKernel
        else:
            raise ValueError(f"Invalid kernel : {kernel_name}")
        return kernel

    def fit(self, X, y, n_iter = 100):
        # x shape [N, f]
        assert X.ndim == 2
        dtype=X.dtype
        device = X.device

        n, f = X.size()
        self.input_size = f
        
        if self.gamma == "scale":
            gamma_ = 1 / (f * X.var())
        elif self.gamma == "auto":
            gamma_ = 1 / f
        else:
            gamma_ = self.gamma
        
        self.gamma_ = gamma_
        
        reg_lambda = 1 / (2 * self.C * n)
        
        #### Compute the kernel matrix
        K = self.kernel.kernel_matrix(X, X, gamma_)
        I = torch.eye(n)

        # Solve KLR using IRLS
        alpha = torch.zeros_like(y)
        for i in range(n_iter):
            m = K@alpha
            W = torch.sigmoid(-m) * torch.sigmoid(m)
            z = m + (y/torch.sigmoid(y * m))
            alpha = torch.linalg.solve(2*torch.diag(W)@K + 2 * n * reg_lambda * I, W*z)
        self.supports = X
        self.alphas = alpha
        return alpha 
    
    def predict(self, x):
        assert x.ndim == 2
        n, f = x.size()
        assert f == self.input_size
        
        K = self.kernel.kernel_matrix(x, self.supports, gamma=self.gamma_)
        
        y = K @ self.alphas
        
        return y
    
    def validation(self, Xtest, Ytest):
        with torch.no_grad():
            y_test_pred = self.predict(Xtest).squeeze(0)
            y_test_pred[y_test_pred >= 0] = 1
            y_test_pred[y_test_pred < 0] = -1
            val_acc = accuracy(y_test_pred, Ytest.squeeze(0))
            return val_acc
        
        
class MultiClassKLR(nn.Module):
    def __init__(self, 
                 n_classes: int,
                 C: float=1.,
                 kernel: str="rbf",
                 gamma="scale",
                 multiclass="ovo"):
                 
        super(MultiClassKLR, self).__init__()
        self.n_classes = n_classes
        self.multiclass = multiclass
        
        keys = [f"{i}{j}" for i in range(self.n_classes - 1) for j in range(i+1, self.n_classes) ]
        self.classifiers = dict(
            [(key, BinaryKLR(kernel, gamma, C)
              ) for key in keys])
        
    def fit(self, X, y):
        assert X.ndim == 2
        assert y.ndim == 1
        # Solves all the binary classifiers
        nc, _ = X.shape
        device = X.device
        
        outer_desc = "Training binary classifiers"
        with tqdm(total=self.n_classes * (self.n_classes - 1) // 2, desc=outer_desc) as pbar:
            for i in range(self.n_classes - 1):
                Xi = X[y==i, :]
                for j in range(i+1, self.n_classes):
                    Xj = X[y==j, :]
                    X_ = torch.cat((Xi, Xj), dim=0)
                    y_ = torch.cat((
                        torch.ones((Xi.shape[0]), device=device),
                        -torch.ones((Xj.shape[0]), device=device)
                        ),
                        dim=0)
                    
                    self.classifiers[f"{i}{j}"].fit(X_, y_)
                    pbar.update(1)

        
    def predict(self, Xtest):
        assert Xtest.ndim == 2
        device = Xtest.device
        
        prediction = torch.zeros((Xtest.shape[0], self.n_classes), device=device)
        
        for i in range(self.n_classes - 1):
            for j in range(i+1, self.n_classes):
                y_test_pred = self.classifiers[f"{i}{j}"].predict(Xtest).float()
                prediction[y_test_pred >= 0, i] += 1
                prediction[y_test_pred < 0, j] += 1
        prediction = prediction.argmax(dim = -1)
        return prediction
    
    def validation(self, Xtest, Ytest): 
        y_test_pred = self.predict(Xtest)
        return accuracy(y_test_pred, Ytest.squeeze(0))
