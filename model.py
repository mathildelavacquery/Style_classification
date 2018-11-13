#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models,transforms,datasets
import bcolz
import time
import imp
import utils; imp.reload(utils)
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Model(object):
  
  def __init__(self, model_type, use_gpu = False):
      self.model_type = model_type #can be vgg16 or resnet
      self.use_gpu = use_gpu
      self.model = None #returns the whole model pretrained
      self.optimizer = None
      self.epoch = 0 #number of epochs, 10 for vgg16, 200 for resnet

  def load_and_tune(self):
      print("fonction to load and tune")
      if self.model_type == 'vgg16':
          model_vgg = models.vgg16(pretrained=True)
          if self.use_gpu:
              print("using gpu version for the model")
              model_vgg = model_vgg.cuda()
          # we update the last group of layers in order to have only 2 output classes    
          for param in model_vgg.parameters():
              param.requires_grad = False
              model_vgg.classifier._modules['6'] = nn.Linear(4096, 127)
          self.model = model_vgg
          self.optimizer = torch.optim.SGD(model_vgg.classifier[6].parameters(), lr = 0.001)
          self.epoch = 10
      
      if self.model_type == 'resnet':
          model_resnet = models.resnet18(pretrained=True)
          for param in model_resnet.parameters():
              param.requires_grad = False

          # Parameters of newly constructed modules have requires_grad=True by default
          num_ftrs = model_resnet.fc.in_features
          # update the last layer to have two classes in output
          model_resnet.fc = nn.Linear(512, 2)
          self.model = model_resnet
          self.optimizer = torch.optim.Adam(model_resnet.fc.parameters(), lr = 0.0001,eps=1e-08)
          self.epoch = 200

  @property
  #features_block is the block of frozen layers
  def features_block(self):
      if self.model_type == 'vgg16':
          return(self.model.features)
      if self.model_type == 'resnet':
          model_resnet_conv = nn.Sequential(*list(self.model.children())[:-1])
          return(model_resnet_conv)

  @property  
  #classif_block is the block of updated layers  
  def classif_block(self):
      if self.model_type == 'vgg16':
          return(self.model.classifier)
      if self.model_type == 'resnet':
          return(self.model.fc)

if __name__ == '__main__':
    m_vgg16 = Model('vgg16')
    m_vgg16.load_and_tune()
    m_resnet = Model('resnet')
    m_resnet.load_and_tune()
    
