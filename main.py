# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:19:45 2024

@author: jamyl
"""


#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from utils import to_image, hinge_loss, train_test_split, rescale_pixel_value
from models.full_models import SIFT_DESCRIPTOR_SVM, HOG_SVM, SIFT_DICT_SVM
from models.ensemble_classifier import EnsembleClassifier

file_path_Xte = 'data/Xte.csv'
file_path_Xtr = 'data/Xtr.csv'
file_path_Ytr = 'data/Ytr.csv'

device= 'cuda' if torch.cuda.is_available() else 'cpu'

Xte = pd.read_csv(file_path_Xte, header=None).to_numpy()[:, :-1]
X = pd.read_csv(file_path_Xtr, header=None).to_numpy()[:, :-1]
Y = pd.read_csv(file_path_Ytr).to_numpy()[:, 1]

X = to_image(X)
X = X.transpose(0, -1, 1, 2)

Xte = to_image(Xte)
Xte = Xte.transpose(0, -1, 1, 2)

X = torch.from_numpy(X).to(device).float()
Xte = torch.from_numpy(Xte).to(device).float()
Y = torch.from_numpy(Y).to(device).float()


X = rescale_pixel_value(X)
Xte = rescale_pixel_value(Xte)

# Split the dataset into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
 
#%%

model_1 = SIFT_DESCRIPTOR_SVM(
    svm_conf={
        "C": 2,
        "gamma": "scale",
        "kernel": "rbf",
        },
    n_classes=10)



svm_conf={
    "C": 5,
    "gamma": "scale",
    "kernel": "rbf",
    }
model_2 = HOG_SVM(
    svm_conf={
        "C": 5,
        "gamma": "auto",
        "kernel": "rbf",
        },
    hog_conf={
        "pixels_per_cell": (8, 8),
        "cells_per_block": (4, 4),
        "orientations": 9,
        },
    n_classes=10,
    )

model_3 = SIFT_DICT_SVM(
    svm_conf={
        "C": 1,
        "gamma": "auto",
        "kernel": "rbf",
        },
    kmean_clusters=256,
    sift_conf={
        "nfeatures": 500,
        "contrastThreshold": 0.001,
        "edgeThreshold": 1_000,
        "sigma": 0.6
        },
    n_classes=10,
    )

total = EnsembleClassifier([model_1, model_2, model_3])
#%%
with torch.no_grad():
    model_1.fit(X_train, y_train)
    
    model_1.validation(X_train, y_train, label="Train")    
    model_1.validation(X_val, y_val, label="Val")    

#%%
with torch.no_grad():
    model_2.fit(X_train, y_train)
    
    model_2.validation(X_train, y_train, label="Train")    
    model_2.validation(X_val, y_val, label="Val")    

#%%
with torch.no_grad():
    model_3.fit(X_train, y_train)
    
    model_3.validation(X_train, y_train, label="Train")    
    model_3.validation(X_val, y_val, label="Val")    

#%%
with torch.no_grad():
    total.fit(X_train, y_train)
    
    total.validation(X_train, y_train, label="Train")    
    total.validation(X_val, y_val, label="Val")  


#%%

with torch.no_grad():
    
    print("Retraining on whole data set") 
    X_train, y_train = X, Y
    
    model_1.fit(X_train, y_train)
    prediction = model_1.predict(Xte).cpu().numpy()
    
    Yte = {'Prediction' : prediction}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv('Yte_pred.csv', index_label='Id')
    
    total.fit(X_train, y_train)
    prediction = total.predict(Xte).cpu().numpy()
    
    Yte = {'Prediction' : prediction}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv('Yte_pred_2.csv', index_label='Id')
    
    
    

