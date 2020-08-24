# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:03:45 2020
helper functions
@author: lwang
"""


from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# print ('tf:', tf.__version__)
import time
from numpy import save, load
from generate_images_v2 import prepare_one_dataset, plotbars,plot1bar, xloc2binaryvec

# sklearn for PR and ROC curves
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve, auc, RocCurveDisplay


def shuffle_trainset(x_train, y_train):
# shuffle [x_train, y_train] along the dimenstion of frequency '120', so that 
# trained NN does not predict by 'remembering' f order. 

    # option A: keep the same sample size
    # sn, fn = y_train.shape #500, 120
    # for i in range(sn):
    #     indices = np.array(range(fn))
    #     np.random.shuffle(indices)   
    #     y_train[i,:] = y_train[i,:][indices]
    #     x_train[i,:,:] = x_train[i,:,:][:,indices]

    # option B: 100 times ramdom shuffle for each training sample (sample size*100)
    sn, fn = y_train.shape #500, 120
    shuffle_times = 100 # for each sample
    new_size = sn*shuffle_times
    d2, d3 = x_train[0].shape
    x_train2 = np.zeros((new_size, d2, d3))
    y_train2 = np.zeros((new_size, fn))
    k=0
    for i in range(sn):
        indices = np.array(range(fn))
        for j in range(shuffle_times):
            np.random.shuffle(indices)   
            y_train2[k] = y_train[i,:][indices]
            x_train2[k] = x_train[i,:,:][:,indices]
            k+=1
            
    return x_train2, y_train2


def normalize_matrix_1(x_train): 
    # normalization [0 - 1] 
    x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
    return x_train
 
    
def plot1bar(xloc, color='green', L = 60):
    yy = np.array(range(L))
    xx = np.ones(L)
    for i in range(len(xloc)):
        plt.plot((xloc[i])*xx, yy, '--',color=color, linewidth=1)


def plot_loss(history, label, n):
    # Use a log scale (plt.semilogy) to show the wide range of values. 
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.semilogy(history.epoch,  history.history['loss'],
                 color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch,  history.history['val_loss'],
            color=colors[n], label='Val '+label,
            linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


def plot_metrics(history):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # plot metrics over epochs
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0,1])
        else:
            plt.ylim([0,1])
        plt.legend()