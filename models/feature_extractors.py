# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 23:14:23 2024

@author: jamyl
"""

import torch
import torch.nn as nn
import torch.optim as optim

from skimage.feature import hog
import kornia

from .kmeans import Kmeans

import cv2
import numpy as np
from tqdm import tqdm


SIFT_THRESH = 0.05
N_DESC_MAX = 20



class Sift():
    def __init__(self,
                nfeatures,
                contrastThreshold,
                edgeThreshold,
                sigma):
        self.sift = cv2.SIFT_create(
                               nfeatures = nfeatures, # max number of features
                               contrastThreshold = contrastThreshold,
                               edgeThreshold = edgeThreshold,
                               sigma = sigma)
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def forward(self, images):
        assert images.ndim == 4
        device = images.device
        
        print("Extracting features ...")
        des_list=[]
        
        # SIFT
        for i, x in enumerate(images):
            gray = x.mean(dim=0).detach().cpu().numpy()
            gray = (gray *255).astype(np.uint8)
            
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            assert descriptors is not None, f"Could not find feature in image {i}"
                
            # img=cv2.drawKeypoints(
            #     gray, keypoints, x.detach().cpu().numpy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # plt.imshow(img)
            
            des_list.append(torch.from_numpy(descriptors).to(device))
            
        return des_list



class SiftDescriptor(nn.Module):
    def __init__(self):
        super(SiftDescriptor, self).__init__()

        self.sift = kornia.feature.SIFTDescriptor(patch_size=32)
        self.flatten = nn.Flatten()


    def fit(self, x):
        return self.forward(x)
    
    def forward(self, x):
        assert x.ndim == 4
        
        nb, nc, h, w = x.shape

        # Run sift
        # desc = torch.cat([self.sift(x[:, i, None]) for i in range(nc)], dim=1)
        desc = self.sift(x.mean(dim=1, keepdims=True))

        out = self.flatten(desc)
        
        
        return out.T
    
    
    
    

class KorniaFeatureExtractor(nn.Module):
    def __init__(self, num_features=64):
        super(KorniaFeatureExtractor, self).__init__()

        # self.sift = kornia.feature.SIFTDescriptor(patch_size=32)
        # self.sift_feat = kornia.feature.SIFTFeatureScaleSpace(num_features=num_features)
        self.sift_feat = kornia.feature.SIFTFeature(num_features=num_features)


        # Add a linear layer to flatten the output
        self.flatten = nn.Flatten()

    def forward(self, x):
        assert x.ndim == 4
        
        nb, nc, h, w = x.shape

        # SIFT
        # This is not relevant
        # desc = torch.cat([self.sift(x[:, i, None]) for i in range(nc)], dim=1)

        # SIFT 2
        frame, resp, desc = self.sift_feat(x.mean(dim=1, keepdims=True))
        
        return desc


class BagOfWordsExtractor(nn.Module):
    def __init__(self,
                 num_bags=8,
                 n_iters_kmeans=10):
        super(BagOfWordsExtractor, self).__init__()
        self.num_bags = num_bags
        self.n_iters_kmeans = n_iters_kmeans
        self.k_means = Kmeans(self.num_bags, self.n_iters_kmeans)
        self.has_been_fit = False
    
    def fit(self, features):
        assert features.ndim == 2 # shape [B*N, D]
        self.k_means.fit(features)
        self.has_been_fit = True
        
    def forward(self, features, mask):
        assert self.has_been_fit, "BagOfWord.fit must be called before forward"
        assert features.ndim == 3
        nb, nc, D = features.shape # shape [B, N, D]
        
        features = features.view(-1, D) # shape [B*N, D]
        nearest_centroids = self.k_means(features) # shape [B*N, 1]
        nearest_centroids = nearest_centroids.view(nb, -1) # shape [B, N]
        
        # temporarily set centroid 0 to "Non existing descriptor"
        nearest_centroids += 1
        nearest_centroids[mask] = 0
        
        histograms = torch.nn.functional.one_hot(nearest_centroids,
                                                 num_classes=self.num_bags + 1).sum(dim = 1) # [B, num_bags] #torch.bincount only work on 1-D array
        # Remove the "non existing descriptor" class
        histograms = histograms[:, 1:]
        

        return histograms.float()
    
class HOG():
    def __init__(self,
                 orientations,
                 pixels_per_cell,
                 cells_per_block):
        
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        
    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)
    
    def extract(self, X):
        features=[]
        for i, x in enumerate(X):
            gray = x.mean(dim=0).detach().cpu().numpy()
            gray = (gray *255).astype(np.uint8)
            
            # HOG
            fd = hog(
                gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                # visualize=True,
                channel_axis=None)
            
            features.append(fd)
            
        im_features = np.stack(features, axis=0)
        return torch.from_numpy(im_features).float().to(X.device)
        
            
        
    
    
    
    