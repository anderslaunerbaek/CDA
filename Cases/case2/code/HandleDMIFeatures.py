# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:48:14 2017

@author: dnor
"""

import mahotas as mh
import numpy as np
import os
import matplotlib.pyplot as plt

class DMI_Handling:
    ''' A class containing some methods for dealing with the DMI data '''
    def build_X(images):
        # Create all data containers first
        # Number of features predefined for speed
        # Move to own file later
        feat_num = 41
        X = np.empty((len(images), feat_num))
    
        # For each image, extract some features
        for i, image in enumerate(images):
            # load first, rgb format
            im = mh.imread(image)
            # for each color channel
            for color in range(3):
                hist, _ = np.histogram(im[:,:,color], range = [0,255] , bins = 10, normed = True)
                X[i, (color* 10):(color + 1) * 10] = hist
            
            # Extract some features on the greyscale
            im_grey = mh.colors.rgb2grey(im)
            hist, _ = np.histogram(im_grey, range = [0, 255], bins = 10, normed = True)
            X[i, 30:40] = hist
            X[i, 40] = np.mean(DMI_Handling.get_dark_channel(im, 15))
    
        return X
    
    def get_dark_channel(image, win):
        ''' produces the dark channel prior in RGB space. 
        Parameters
        ---------
        Image: M * N * 3 numpy array
        win: Window size for the dark channel prior
        '''
        M = np.size(image, 0)
        N = np.size(image, 1)
        
        pad = int(win / 2)
        
        # Pad all axis, but not color space
        padded = np.pad(image, ((pad, pad), (pad, pad), (0,0)), 'edge')
        padded = np.pad(image, ((pad, pad), (pad, pad), (0,0)), 'edge')
        
        dark_channel = np.zeros((M, N))
        
        for i, j in np.ndindex(dark_channel.shape):
            dark_channel[i,j] = np.min(padded[i:i + win, j:j + win, :])
            
        return dark_channel

    def show_importances(clf, X):
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()
        
    def get_im_files(_dir):
        ''' Load all files in _dir given'''
        f = []
        for (dirpath, dirnames, filenames) in os.walk(_dir):
            for file in filenames:
                f.append(os.path.join(dirpath, file))
        return f
    
    def show_images(_dir):
        ''' show 24 images as given in _dir '''
        largest = 24
        fig = plt.figure()
        
        for i in range(1,largest + 1):
            
            if i > largest or i > len(_dir) - 1: 
                break
            a = plt.subplot(4, 6,i)
            plt.imshow(plt.imread(_dir[i]))
    
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_aspect("equal")
            a.xaxis.set_ticks_position('none')
            a.yaxis.set_ticks_position('none')
            
            a.set_title("Image " + str(i))
        fig.subplots_adjust(hspace = 0.2, wspace = 0.2)
        
        
    def create_subplots(_images):
        for i in range(1,len(_images), 24):
            DMI_Handling.show_images(_images[(i-1):(i+24)])
            
    