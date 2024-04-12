# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:41:25 2024

@author: jamyl
"""

import numpy as np
import torch
from utils import accuracy

class EnsembleClassifier():
    def __init__(self, classifiers: list=[]):
        self.n_classifiers = len(classifiers)
        assert self.n_classifiers > 0
        
        self.classifiers = classifiers
    
    def fit(self, X_train, y_train):
        for i, classifier in enumerate(self.classifiers):
            print(f"___ Training classifier {i+1}/{self.n_classifiers}")
            classifier.fit(X_train, y_train)
    
    def predict(self, X):
        predictions = []
        for i, classifier in enumerate(self.classifiers):
            print(f"___ Running classifier {i+1}/{self.n_classifiers}")
            predictions.append(classifier.predict(X))
        
        predictions = torch.stack(predictions, dim=0).cpu().numpy()
        prediction = self.majority_voting(predictions)
        
        return torch.from_numpy(prediction).to(X.device)

    def validation(self, X, y, label=""):
        for i, classifier in enumerate(self.classifiers):
            label_ = f"Classifier {i+1} " + label
            classifier.validation(X, y, label_)
        
        prediction = self.predict(X)
        acc = accuracy(prediction, y)

        print("="*10, "Total : ", f"{label} Accuracy : {acc}")

        
        
    def majority_voting(self, predictions):
        # Find the most voted class for each prediction
        most_voted_classes = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

        # Iterate over all positions in the input array
        for position in range(predictions.shape[1]):
            unique_votes = np.unique(predictions[:, position])

            # Check if all classifiers have different votes at this position
            if len(unique_votes) == predictions.shape[0]:
                most_voted_classes[position] = np.random.choice(unique_votes)

        return most_voted_classes