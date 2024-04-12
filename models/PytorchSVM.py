# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:15:29 2024

@author: jamyl
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from qpth.qp import QPFunction
from utils import accuracy, hinge_loss

from tqdm import tqdm

EPS_SUPPORT = 1e-8

class BinarySVM(nn.Module):
    def __init__(self,
                 kernel: str="rbf",
                 gamma="scale",
                 C: float=1.):
        super(BinarySVM, self).__init__()
        
        assert gamma in ["scale", "auto"] or isinstance(gamma, float)
        self.gamma = gamma
        
        self.C=C
        
        self.kernel_name = kernel
        
        if kernel == 'rbf':
            from kernels.gaussian_kernel import GaussianKernel
            kernel = GaussianKernel
            self.kernel = kernel(gamma=gamma)
        elif kernel == 'linear':
            from kernels.linear_kernel import LinearKernel
            kernel = LinearKernel
            self.kernel = kernel(gamma=None)
        else:
            raise ValueError(f"Invalid kernel : {kernel}")

        
        self.input_size = None
        self.Xtrain = None
        self.alphas = None
        
        
    def fit(self, X, y):
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
        
        ####  Solve the Quadratic Problem using differentiable solver
        # https://locuslab.github.io/qpth/
        # the solver solves for
        # z = argmin (zT Q z)/2 + pT z
        # s.t Az = b ;  Gz <= hz
        b = 1
        
        diag_y = torch.diag_embed(y, offset=0, dim1=-2, dim2=-1)
        
        Q = diag_y @ K @ diag_y / (2 * reg_lambda)
        
        p = - torch.ones((b, n), device=device, dtype=dtype)
        
        G = torch.cat([
              torch.eye(n, device=device, dtype=dtype),
            - torch.eye(n, device=device, dtype=dtype)],
            dim = 0)
        G = G.unsqueeze(0).expand((b, 2*n, n))
        # G : shape [b, 2n, n]
        
        h = torch.cat([
              torch.ones(n, device=device, dtype=dtype)/n,
            - torch.zeros(n, device=device, dtype=dtype)],
            dim = 0)
        h = h.unsqueeze(0).expand((b, 2*n))
        # h : shape [b, 2n]
        
        
        try:
            # raise ValueError()
            mu = QPFunction(verbose=False)(
                Q, p, G, h,
                Variable(torch.Tensor()), # Not used
                Variable(torch.Tensor())) # Not used
        
            assert not(torch.any(mu.isnan())) and not(torch.any(mu.isinf()))
        
        except:
            print("="*30+"\nWarning : binary SVM failed !!!")
            print("Falling back to qpsolvers")
            
            assert b == 1, "Cannot fallback to qpsolvers with batch os qp problems."
            ### Plan B !!

            from qpsolvers import solve_qp
            # https://pypi.org/project/qpsolvers/
            P_ = Q.cpu().detach().numpy().squeeze()
            q_ = p.cpu().detach().numpy().squeeze()
            G_ = G.cpu().detach().numpy().squeeze()
            h_ = h.cpu().detach().numpy().squeeze()
            mu = solve_qp(P_, q_, G_, h_, solver="osqp") # I have not investigated the solver used.
            mu = torch.from_numpy(mu).to(device).unsqueeze(0)
            
            print("Done !\n"+"="*30)
        
        
        alpha = (diag_y @ mu.squeeze(0).float()) / (2 * reg_lambda)
        
        
        
        mask = alpha.abs() >= EPS_SUPPORT
        self.supports = X[mask]
        self.alphas = alpha[mask]
        

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
        
        
class MultiClassSVM(nn.Module):
    def __init__(self, 
                 n_classes: int,
                 C: float=1.,
                 kernel: str="rbf",
                 gamma="scale",
                 multiclass="ovo"):
                 
        super(MultiClassSVM, self).__init__()
        self.n_classes = n_classes
        self.multiclass = multiclass
        
        keys = [f"{i}{j}" for i in range(self.n_classes - 1) for j in range(i+1, self.n_classes) ]
        self.classifiers = dict(
            [(key, BinarySVM(kernel, gamma, C)
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
