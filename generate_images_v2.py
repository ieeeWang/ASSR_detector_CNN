# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:57:08 2020
generate 120*12 grey images with random vertical bars, add noise
@author: lwang
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import save
# print(np.__version__)

# genearate loc of vertical bars (x1, x2, ..., x7)
def get_bar_locs():
    # genearate x of 2*MB between [30, 39] Hz
    r1= 30; r2=40
    locR = np.array(range(r1, r2))
    # loc = np.random.randint(10, size= n)# with repeting
    loc = np.random.choice(r2-r1, 2, replace=False) # without repeting
    locMB = np.sort(locR[loc]) # assending
    locMB = np.reshape(locMB, (1,2))
    c = np.ones_like(locMB) # 1: the types of MB
    locMB = np.concatenate((locMB,c), axis=0)

    # generate other x
    locMB2 = 2*locMB
    locMB3 = 3*locMB
    locAll = np.concatenate((locMB, locMB2, locMB3),axis = 1)
    
    return locAll #(2,6), x locs@0 row; catergary@1 row 

# generate grey images using bars: locAll
def get_one_image(locAll):
    ndim = (12, 120) # image dim
    image0 = np.zeros(ndim) 
    for i in range(locAll.shape[1]):
        image0[:,locAll[0,i]] = 255/locAll[1,i]        
    snr=1/1 # noise 
    noise2add = np.random.normal(loc=255*snr, scale=60, size=ndim)
    image1 = image0 + noise2add
    image1= np.abs(image1)
    image1 *= (255.0/(image1.max())) #normalized within a range [0 255]
    return image1

# generate grey image dataset
def prepare_one_dataset(N = 1000):
    images = np.zeros((N, 12, 120), dtype=int)
    labels = np.zeros((N, 2, 6),dtype=int)
    for i in range(N): 
        locAll = get_bar_locs()
        image1 = get_one_image(locAll)

        images[i]=image1
        labels[i]=locAll
    return images,labels

def plotbars(xloc, color='green'):
    L = 12
    yy = np.array(range(L))
    xx = np.ones(L)
    for i in range(len(xloc)):
        plt.plot((xloc[i]-1)*xx, yy, '--',color=color, linewidth=1)
        plt.plot((xloc[i]+1)*xx, yy, '--',color=color, linewidth=1)

def plot1bar(xloc, color='green'):
    L = 30
    yy = np.array(range(L))
    xx = np.ones(L)
    for i in range(len(xloc)):
        plt.plot((xloc[i])*xx, yy, '--',color=color, linewidth=1)


        
def xloc2binaryvec(xloc):
    binaryvec = np.zeros(120, dtype ='int32')
    binaryvec[xloc]=1
    return binaryvec    

#%%
if __name__=='__main__':  
    locAll = get_bar_locs()
    image = get_one_image(locAll)
        
    # plot 1 sample
    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    plotbars(locAll[0,:])
    plt.title(locAll[0,:])
    
    images,labels = prepare_one_dataset(100)
    # plot 10 samples
    #%% pick first N samples
    num = 20
    images_num = images[:num]
    labels_num = labels[:num]
    fig = plt.figure(figsize=(5, 10))
    for i in range(num):
        # plot the sample
        plt.subplot(num,1,i+1)
        plt.imshow(images_num[i], cmap='gray')
        plotbars(labels_num[i][0,:])

    #%% generate a random dataset & save
    # start_time = time.time()
    # images_all, labels_all = prepare_one_dataset(10000) # ~10 sec
    # save('images_10k_v2_snr1.npy', images_all)
    # save('labels_10k_v2_snr1.npy', labels_all)
    # print('elapsed_time:',  time.time() - start_time)
        
    
















