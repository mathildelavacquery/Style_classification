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

from utils import *


class Trainer(object):
  
  def __init__(self, model,
               sizes, class_names,
               m_softm = nn.Softmax(dim=1),  
               criterion = nn.CrossEntropyLoss(), 
               learning_rate = 0.01,
               use_gpu = False):

      self.model = model #has to be from Model class
      self.sizes = sizes
      self.class_names = class_names
      self.m_softm = m_softm
      self.criterion = criterion
      #self.optimizer = optimizer
      self.learning_rate = learning_rate
      self.conv_feat_dic = None
      self.labels_dic = None
      self.losses = None
      self.accuracies = None
      self.use_gpu = use_gpu
    
  def load_inputs(self, save_dir_features_dic, save_dir_labels_dic):
      conv_feat_dic = {}
      labels_dic = {}
      for phase in ['train', 'valid']:
          conv_feat_dic[phase] = load_array(save_dir_features_dic[phase])
          labels_dic[phase] = load_array(save_dir_labels_dic[phase])
      self.conv_feat_dic = conv_feat_dic
      self.labels_dic = labels_dic
      return(conv_feat_dic, labels_dic)
    
  def data_gen(self, conv_feat, labels, batch_size=64,shuffle=True):
      labels = np.array(labels)
      if shuffle:
          index = np.random.permutation(len(conv_feat))
          conv_feat = conv_feat[index]
          labels = labels[index]
      for idx in range(0,len(conv_feat),batch_size):
          yield(conv_feat[idx:idx+batch_size],labels[idx:idx+batch_size]) 
        
   
  def train(self, shuffle=True):
      losses = {}
      accuracies = {}
      for phase in ['train', 'valid']:
          losses[phase] = []
          accuracies[phase] = []
          
      for epoch in range(self.model.epoch):
          for phase in ['train', 'valid']:
              if phase == 'train':
                  self.model.model.train()
              else:
                  self.model.model.eval()
              batches = self.data_gen(conv_feat=self.conv_feat_dic[phase],labels=self.labels_dic[phase],shuffle=True)
              total = 0
              running_loss = 0.0
              running_corrects = 0.0
              for inputs, classes in batches:
                  if self.use_gpu:
                      inputs , classes = torch.from_numpy(inputs).cuda(), torch.from_numpy(classes).cuda()
                  else:
                      inputs , classes = torch.from_numpy(inputs), torch.from_numpy(classes)
                 
                  if self.model.use_gpu:
                    self.model.model = self.model.model.cuda()
                  inputs = inputs.view(inputs.size(0), -1)
                  outputs = self.model.classif_block(inputs) #contain weights
                  loss = self.criterion(outputs, classes)  
                  #if train:
                  if phase == 'train':
                      if self.model.optimizer is None:
                          raise ValueError('Pass optimizer for train mode')
                      self.model.optimizer.zero_grad()
                      loss.backward() ##gradient
                      self.model.optimizer.step() ##step of the optimizer
                  _,preds = torch.max(outputs.data,1)
                  # statistics
                  running_loss += loss.data.item()
                  running_corrects += torch.sum(preds == classes.data)
                  

              epoch_loss = running_loss / self.sizes[phase]
              epoch_acc = float(running_corrects.data.item()) / float(self.sizes[phase])
              print(phase + ' - Loss: {:.4f} Acc: {:.4f}'.format(
                           epoch_loss, epoch_acc))
              losses[phase].append(epoch_loss)
              accuracies[phase].append(epoch_acc)
      self.losses = losses
      self.accuracies = accuracies
      return(losses, accuracies)
  
  def plot_measures(self, measure = 'loss'):
      if measure == 'loss':
          dic = self.losses
          title = 'Loss curves rgd the number of epochs'
      if measure == 'accuracy':
          dic = self.accuracies
          title = 'Accuracy curves rgd the number of epochs'

      yt = dic['train']
      yv = dic['valid']
      x = range(len(yt))
      plt.plot(x, yt, 'r', label = 'train')
      plt.plot(x, yv, 'b', label = 'valid')
      plt.title(title)
      plt.legend()
      plt.show()     