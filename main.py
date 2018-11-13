#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
# import urllib
from class_dataloader import *
from features_generator import *
from trainer import *
from model import *
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train/Run a ResNet/VGG Art Style classifier')
    parser.add_argument('-gpu', '--use_gpu', type = str, nargs = '?', help = 'True if GPU available', default = 'False')
    parser.add_argument('-o', '--is_organised', type = str, nargs = '?', help = 'False if the training directory is not organized yet', default = 'True')
    parser.add_argument('-c', '--class_names', type = list, nargs = '?', help = 'Name of the raw data directory', default = ['impressionism', 'realism'])
    parser.add_argument('-rd','--raw_data_dir', type = str, nargs= '?', help = 'Name of the raw data directory', default = 'None')
    parser.add_argument('-dd', '--data_dir', type = str, help = 'Name of the training directory')
    parser.add_argument('-sc', '--computeScalingFromScratch', type = str, nargs = '?', help = 'True if we need to compute the scaling parameters from scratch', default = 'False')
    parser.add_argument('-m', '--model', type = str, nargs = '?', help = 'model to use, either resnet or vgg16', default = 'resnet')
    parser.add_argument('-gf', '--generate_features', type = str, nargs = '?', help ='True to generate, False to use the already generated features', default = 'False')
    args = parser.parse_args()
    

    # download the raw data from kaggle competition - need to be registered and to approave project
    #urllib.urlretrieve("https://www.kaggle.com/c/painter-by-numbers/download/train_1.zip", filename= "")
    #urllib.urlretrieve("https://www.kaggle.com/c/painter-by-numbers/download/train_2.zip", filename= "")
    #urllib.urlretrieve("https://www.kaggle.com/c/painter-by-numbers/download/train_info.csv", filename= "")


    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1", "True")

    gpu = str2bool(args.use_gpu)
    print('Using GPU: ', gpu)

    # Part 0 - Organize the directory if needed
    loader = Dataloader(args.is_organised, args.class_names, args.raw_data_dir, args.data_dir)
    if loader.is_organised == "False":
        print('ok')
        df1 = loader.organise_dataset(0.8, 1)
        df2 = loader.organise_dataset(0.8, 2)
        df = df1.append(pd.DataFrame(data = df2), ignore_index=True)
        print('Train, valid repartition: ', df.groupby(['train_valid', 'style'])['filename'].count())

    # Part 1 - dataloader
    dataloader, dataset_sizes, class_names = loader.getDataLoader(loader.data_dir, args.computeScalingFromScratch)
    
    # Part 2 - Features Generator
    output_dir = args.data_dir + '/' + args.model
    if args.generate_features == 'True':
        fg = FeaturesGenerator(args.model, dataloader, output_dir, use_gpu = gpu)
        fg.generate()
        fg.plotPCA(class_names)
   
    # retrieve the features and labels already generated
    save_dir_features_dic = {'train' : output_dir+'/conv_feat_train.bc', 'valid' : output_dir+'/conv_feat_valid.bc'}
    save_dir_labels_dic   = {'train' : output_dir+'/labels_train.bc', 'valid' : output_dir+'/labels_valid.bc'}

    # Part 3 - set up the model
    print('Model chosen: ', args.model)
    m = Model(args.model)
    m.load_and_tune()
    
    #fb = m.features_block
    #classifier = m.classif_block

    # Part 4 - train the model
    train_model= Trainer(m,dataset_sizes, args.class_names,use_gpu = gpu)
    train_model.load_inputs(save_dir_features_dic,save_dir_labels_dic)
    train_model.train()
    print(train_model.plot_measures("loss"))
    print(train_model.plot_measures("accuracy"))
    
    




   
        
