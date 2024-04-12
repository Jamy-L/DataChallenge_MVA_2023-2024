# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:43:04 2024

@author: jamyl
"""
import numpy as np
from skimage.feature import hog
import torch as th

from models.PytorchSVM import MultiClassSVM
from models.PytorchKLR import MultiClassKLR
from models.feature_extractors import SiftDescriptor, HOG, Sift, BagOfWordsExtractor
from models.scaler import StandardScaler
from utils import accuracy

class SIFT_DESCRIPTOR_SVM():
    def __init__(self,
                 svm_conf,
                 n_classes=10):
        self.n_classes = n_classes
        
        self.feature_extractor = SiftDescriptor()
        self.classifier = MultiClassSVM(n_classes,
                                        C = svm_conf["C"],
                                        gamma = svm_conf["gamma"],
                                        kernel = svm_conf["kernel"])
        self.scaler = StandardScaler()
        
    
    def fit(self, X_train, y_train):
        print("Extractig features ... ")
        feature_train = self.feature_extractor.fit(X_train).T

        # center features
        self.scaler.fit(feature_train)
        feature_train = self.scaler.transform(feature_train)

            
        print("training classifiers ... ")
        self.classifier.fit(feature_train, y_train)
    
    def predict(self, X):
        feature = self.feature_extractor(X).T
        feature = self.scaler.transform(feature)
        
        prediction = self.classifier.predict(feature)
        return prediction

    def validation(self, X, y, label=""):
        prediction = self.predict(X)
        acc = accuracy(prediction, y)
        print(f"{label} Accuracy : {acc}")

class HOG_SVM():
    def __init__(self,
                 svm_conf,
                 hog_conf,
                 n_classes=10):
        
        self.feature_extractor = HOG(
            orientations = hog_conf["orientations"],
            pixels_per_cell = hog_conf["pixels_per_cell"],
            cells_per_block = hog_conf["cells_per_block"])

        self.classifier = MultiClassSVM(n_classes,
                                        C = svm_conf["C"],
                                        gamma = svm_conf["gamma"],
                                        kernel = svm_conf["kernel"])
        self.stdslr=StandardScaler()
    
    def fit(self, X_train, y_train):
        print("Extracting features ...")
        im_features = self.feature_extractor(X_train)
        
        self.stdslr.fit(im_features)
        im_features=self.stdslr.transform(im_features)

        print("Fitting SVM ...")
        # self.classifier.fit(im_features.detach().cpu().numpy(), y_train.detach().cpu().numpy())
        self.classifier.fit(im_features, y_train)
        print("Done ...")

    def predict(self, X):
        print("Extracting features ...")
        test_features = self.feature_extractor(X)
                
        test_features=self.stdslr.transform(test_features)
        predict_classes = self.classifier.predict(test_features)
        
        return predict_classes.int()
        
    def validation(self, X, y, label=""):
        prediction = self.predict(X)
        acc = accuracy(prediction, y)
        print(f"{label} Accuracy : {acc}")
        
        
class SIFT_DICT_SVM():
    def __init__(self,
                 svm_conf,
                 kmean_clusters,
                 sift_conf,
                 n_classes=10):
        
        self.descriptor_extractor = Sift(
                        nfeatures = sift_conf["nfeatures"],
                        contrastThreshold = sift_conf["contrastThreshold"],
                        edgeThreshold = sift_conf["edgeThreshold"],
                        sigma = sift_conf["sigma"])
        
        self.stdslr_1=StandardScaler()
        self.word_extractor = BagOfWordsExtractor(kmean_clusters,
                                                  n_iters_kmeans=10)
        self.stdslr_2=StandardScaler()        

        
        self.classifier = MultiClassSVM(n_classes,
                                        C = svm_conf["C"],
                                        gamma = svm_conf["gamma"],
                                        kernel = svm_conf["kernel"])
        
        self.kmean_clusters = kmean_clusters
    
    def fit(self, X_train, y_train):
        descriptor_list= self.descriptor_extractor(X_train)
        descriptors = th.cat(descriptor_list, dim=0)
        
        self.stdslr_1.fit(descriptors)
        descriptors=self.stdslr_1.transform(descriptors)

        ###### scipy's way
        # print("Clustering ...")
        # from scipy.cluster.vq import kmeans, vq
        # a = descriptors.detach().cpu().numpy().copy()
        # self.voc, variance=kmeans(a,
        #                           self.kmean_clusters,
        #                           iter=1)
        # print("Done")
        
        
        # print("Histogram ...")
        # im_features = th.zeros((X_train.shape[0], self.kmean_clusters), device=X_train.device)
        
        # for i, descriptors in enumerate(descriptor_list):
        #     descriptors = self.stdslr_1.transform(descriptors)
            
        #     words, distance = vq(descriptors.cpu().numpy(), self.voc)
        #     for w in words:
        #         im_features[i][w] += 1
        # print("Done")
        ############## Our way
        # Fit kmeans
        print("Clustering ...")
        self.word_extractor.fit(descriptors)
        print("Done")

        # Pad feat
        N_feat = max([x.shape[0] for x in descriptor_list])
        feat_dim = descriptor_list[0].shape[1]

        ft = th.stack(
            [th.cat([x, th.zeros((N_feat - x.shape[0], feat_dim), device=x.device)], dim=0) for x in descriptor_list],
            dim=0
        )
        mask = th.stack(
            [th.cat([th.zeros(x.shape[0], device=x.device), th.ones(N_feat - x.shape[0], device=x.device)], dim=0) for x in descriptor_list],
            dim=0
        ).bool()

        im_features = self.word_extractor(ft, mask)

        
        
        self.stdslr_2.fit(im_features)
        im_features=self.stdslr_2.transform(im_features)

        print("Fitting SVM ...")
        self.classifier.fit(im_features, y_train)
        print("Done ...")
        
        return self.classifier.predict(im_features).int()

    def predict(self, X):
        descriptor_list = self.descriptor_extractor(X)
            
        ###### Dictionnary
        # from scipy.cluster.vq import vq
        # test_features=th.zeros((X.shape[0], self.kmean_clusters), device=X.device)
        # mean_dist = 0
        # for i, descriptors in enumerate(descriptor_list):
        #     descriptors = self.stdslr_1.transform(descriptors)
            
        #     words, distance = vq(descriptors.cpu().numpy(), self.voc)
        #     mean_dist += distance.mean()
        #     for w in words:
        #         test_features[i][w] += 1
                # Pad feat
        N_feat = max([x.shape[0] for x in descriptor_list])
        feat_dim = descriptor_list[0].shape[1]

        ft = th.stack(
            [th.cat([x, th.zeros((N_feat - x.shape[0], feat_dim), device=x.device)], dim=0) for x in descriptor_list],
            dim=0
        )
        mask = th.stack(
            [th.cat([th.zeros(x.shape[0], device=x.device), th.ones(N_feat - x.shape[0], device=x.device)], dim=0) for x in descriptor_list],
            dim=0
        ).bool()

        test_features = self.word_extractor(ft, mask)
                
        test_features=self.stdslr_2.transform(test_features)
        
        predict_classes = self.classifier.predict(test_features)
        
        return predict_classes.int()
    
    def validation(self, X, y, label=""):
        prediction = self.predict(X)
        acc = accuracy(prediction, y)
        print(f"{label} Accuracy : {acc}")


class SIFT_DESCRIPTOR_KLR():
    def __init__(self,
                 svm_conf,
                 n_classes=10):
        self.n_classes = n_classes
        
        self.feature_extractor = SiftDescriptor()
        self.classifier = MultiClassKLR(n_classes,
                                        C = svm_conf["C"],
                                        gamma = svm_conf["gamma"],
                                        kernel = svm_conf["kernel"])
        self.scaler = StandardScaler()
        
    
    def fit(self, X_train, y_train):
        print("Extractig features ... ")
        feature_train = self.feature_extractor.fit(X_train).T

        # center features
        self.scaler.fit(feature_train)
        feature_train = self.scaler.transform(feature_train)

            
        print("training classifiers ... ")
        self.classifier.fit(feature_train, y_train)
    
    def predict(self, X):
        feature = self.feature_extractor(X).T
        feature = self.scaler.transform(feature)
        
        prediction = self.classifier.predict(feature)
        return prediction

    def validation(self, X, y, label=""):
        prediction = self.predict(X)
        acc = accuracy(prediction, y)
        print(f"{label} Accuracy : {acc}")     