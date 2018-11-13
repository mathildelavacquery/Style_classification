#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:54:16 2018

@author: charlotte.caucheteux
"""

#import the packages

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models,transforms,datasets
import torch
import bcolz
import time
from utils import *


class FeaturesGenerator(object):
    
    def __init__(self, model_type, dataloader, output_dir, use_gpu = False):
        self.model_type = model_type #can be vgg or resnet
        self.dataloader = dataloader #dictionnaire avec train et valid dedans 
        self.output_dir = output_dir #path where to save the computed features
        self.use_gpu = use_gpu
        self.model = None #gives the group of layers that are going to be frozen
        self.features = None #features only for the train part, used for the PCA
        self.labels = None #labels only for the train part, used for the PCA



    #Returns the frozen layers, from which we are going to compute and save the features
    def get_preconvfeat_model(self):
        
        if self.model_type == 'resnet':
            #if the model is resnet, we freeze all the layers except the last one
            model_resnet = torchvision.models.resnet18(pretrained=True)
            if self.use_gpu:
                model_resnet = model_resnet.cuda()
            for param in model_resnet.parameters():
                param.requires_grad = False
                    
            # Parameters of newly constructed modules have requires_grad=True by default
            num_ftrs = model_resnet.fc.in_features
            model_features_block = nn.Sequential(*list(model_resnet.children())[:-1])
         
        if self.model_type == 'vgg16':
            #if the model is vgg, we freeze the first group of layers : classifiers
            model_vgg = models.vgg16(pretrained=True)
            if self.use_gpu:
                model_vgg = model_vgg.cuda()

            for param in model_vgg.parameters():
                param.requires_grad = False
            model_features_block = model_vgg.features
         
        
        self.model = model_features_block
        return(model_features_block)
        

    # compute the features at the end of the frozen layers    
    def compute_preconvfeat(self, model, dataset, save_dir_features, save_dir_labels, save_batch = 10):
        conv_features = []
        labels_list   = []
        i = 0

        for data in dataset:
            i = i + 1
            try : 
                inputs, labels = data
                if self.use_gpu:
                    inputs , labels = Variable(inputs.cuda()),Variable(labels.cuda())
                else:
                    inputs , labels = Variable(inputs),Variable(labels)
                x = model(inputs)
                conv_features.extend(x.data.cpu().numpy())
                labels_list.extend(labels.data.cpu().numpy())
                
                if i % save_batch == 0:
                    save_array(save_dir_features, conv_features)
                    save_array(save_dir_labels, labels_list)

            except (OSError, ValueError, IOError):
                print(str(i) + " image is not used ! " )
             
        conv_features = np.concatenate([[feat] for feat in conv_features])
        #save the features in the good directory
        save_array(save_dir_features, conv_features)
        save_array(save_dir_labels, labels_list)   
            
        return (conv_features, labels_list) 
  

    #generates and saves the features for the frozen layers, calling the previous functions
    def generate(self):
        model = self.get_preconvfeat_model()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        for phase in ['valid', 'train']:
            dataset = self.dataloader[phase]
            save_dir_features = os.path.join(self.output_dir, 'conv_feat_'+phase+'.bc')
            save_dir_labels = os.path.join(self.output_dir, 'labels_'+phase+'.bc')
            features, labels = self.compute_preconvfeat(model,dataset, save_dir_features, save_dir_labels)
        self.features, self.labels = features, labels
    
    

    #plots the PCA for the train dataset with the two classes : impressionism and realism
    def plotPCA(self,class_names):
        X = np.array([x.flatten() for x in self.features])
        y = np.array(self.labels)
        label_names = class_names
        
        pca = PCA(n_components=2)
        X_pca = pca.fit(X).transform(X)
        
        colors = ['navy', 'turquoise']
        
        plt.figure(figsize=(8, 8))
    #    for color, i, target_name in zip(colors, [0, 1], iris.target_names):
        for color, i, target_name in zip(colors, [0, 1], label_names):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                        color=color, lw=2, label=target_name, s = 1)
    
        plt.title("PCA")
        plt.legend()
        plt.show()




if __name__ == 'main':
    output_dir = 'data/classif_impressionism_realism/vgg'
    fg_vgg = FeaturesGenerator('vgg16', dataloaders, output_dir, use_gpu = True)
    fg_vgg.generate()
    fg_vgg.plotPCA(class_names)

    output_dir = 'data/classif_impressionism_realism/resnet'
    fg_resnet = FeaturesGenerator('resnet', dataloaders, output_dir, use_gpu = True)
    fg_resnet.generate()
    fg_resnet.plotPCA(class_names)
 
