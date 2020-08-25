# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:07:41 2020
made this file an API for batch_detector
v1.1 add an augument: model_dir @8/25/2020
V1.2 update the last figure: both pred of AVG-100 and mean of AVG-90
@author: lwang
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from numpy import save, load
from scipy.io import loadmat, savemat
from utils import normalize_matrix_1, plot1bar


def detector_subj (subj_train, subj_test, model_dir, save_dir, plot=1):
       
    labels_full= [4, 6, 27, 33, 35, 37, 39, 43, 45, 66, 72, 76, 78, 99,105,111,117]
    labels_subset = [6, 33, 37, 39, 43, 66, 72, 78, 99, 117] # (1)
    labels_subset2= [6, 33, 37, 39, 43, 45, 66, 72, 76, 78, 99, 117] # (2)
    labels_subset3= [6, 33, 35, 37, 39, 43, 45, 66, 72, 76, 78, 99, 117] # (3) 
    
    #%% laod trainded model (.h5)
    # Re-load the model with the best validation accuracy
    # model_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = subj_train + '_best_trained_model.h5'
    model= keras.models.load_model(os.path.join(model_dir, model_name))
    
    # choose one annotation from above
    labels2use = labels_full
    labels2plt = np.array(labels2use)-1 # python starts from 0
    
    #%% load test sets (.mat)
    # (1)test set: AVG-EEG on day 2
    filename = 'DL_trainset/AVG96/' + subj_test + '.mat'
    file = loadmat(filename)
    data_test_AVG = file['SNR3D'] # (96, 60, 120)
    
    # (2)test set: randomly-selected EEG on day 2
    filename = 'DL_trainset/AVG_ram100/' + subj_test + '.mat'
    file = loadmat(filename)
    data_test_ram = file['SNR3D_arr'] # (100, 60, 120) x 5
    
    #%% test (1) the best_trained_model on test set: AVG-EEG on day 2
    data_test_AVG =  normalize_matrix_1(data_test_AVG) # normalize to [0-1]
    predictions = model.predict(data_test_AVG)
    
    # save Matrix from Python to MATLAB (.mat data)
    pred_AVG = predictions
    mdic = {"predictions_3D": pred_AVG}
    savename = subj_test + "_pred_AVG96.mat"
    
    # save_dir = os.path.join(os.getcwd(), 'mat')
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
        
    savemat(os.path.join(save_dir, savename), mdic)
    
    
    if plot == 1:
        plt.figure(figsize=(16, 9))
        plt.imshow(predictions, cmap='gray')
        plot1bar(labels2plt, L=96)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Number of used trials')
        plt.title('Prediction on AVG trials ')    
        
        # plot probility on AVG-trials: [50, 60, 70, 80, 90, 100]
        nAVG = [50, 60, 70, 80, 90, 100]
        plt.figure(figsize=(16, 9))
        for i in range(6):  
            prediction_on_1AVG = predictions[nAVG[i]-5]
            plt.plot(prediction_on_1AVG, label= 'AVG {} trials'.format(nAVG[i]), linestyle='--')
           
        plot1bar(labels2plt, L=2)
        plt.grid(True)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('probability')
        plt.title('prediction on AVG trials (with Ann)')
        plt.legend(loc='best', fontsize='x-large')
         
    
    #%% test (2) the best_trained_model on randomly-selected EEG, 
    x_test_ram = data_test_ram[0, 4] # ntrials = [90]  
    x_test_ram =  normalize_matrix_1(x_test_ram)
    predictions = model.predict(x_test_ram)
    predictions_mean = np.mean(predictions, axis = 0)
     
    # save Matrix from Python to MATLAB (.mat data)
    pred_AVG = predictions_mean
    mdic = {"predictions_3D": pred_AVG}
    savename = subj_test + "_pred_ram90.mat"  
    savemat(os.path.join(save_dir, savename), mdic)
    
    
    if plot == 1:
        # plot probility on AVG trials = [90]
        plt.figure(figsize=(16, 9))
        plt.plot(predictions_mean, label= 'mean of 100 AVG-90 trials', linestyle='--')
        plt.plot(prediction_on_1AVG, label= 'AVG-100 trials', linestyle='--')    
        plot1bar(labels2plt, L=2)
        plt.grid(True)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('probability')
        plt.title('prediction on AVG trials (with Ann)')
        plt.legend(loc='best', fontsize='x-large')
        
    
    







