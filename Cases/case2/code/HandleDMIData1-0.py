# -*- coding: utf-8 -*-
"""
Created by dnor

This is a script for handling dmi image data stored in a respective folder,
with some images in the clear and foggy folders.

Building X is very computationally heavy (10 min on my pc), as images contain a large amount of data,
consider down-sampling images for speed if needed.

Notice that no proper Crossvalidation is implemented, this is needed.

"""
#%%
import sys
# Append path that contains HandleDMIFeatures
sys.path.append("C:\\Users\\dnor\\Desktop\\DMI projekt\\")
from HandleDMIFeatures import DMI_Handling as HDMI

import numpy as np
import os
import mahotas as mh
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rndForest
from sklearn.metrics import accuracy_score
from scipy.stats import norm

import matplotlib.pyplot as plt

# Load data
top_folder = "C:\\Users\\dnor\\Desktop\\DMI projekt\\online_rep\\data\\"

foggy_im_path = top_folder + "foggy\\"
clear_im_path = top_folder + "clear\\"
net_im_path = top_folder + "net\\"

foggy_im = HDMI.get_im_files(foggy_im_path)
clear_im = HDMI.get_im_files(clear_im_path)
net_im = HDMI.get_im_files(net_im_path)

all_images = [foggy_im, clear_im]

# Flatten, so one list has all paths
all_images_flat = [item for sublist in all_images for item in sublist]

# visualize all images in 4 x 6 subplots
create_subs = False
if create_subs:
    HDMI.create_subplots(foggy_im)
    HDMI.create_subplots(clear_im)
    #create_subplots(net_im)
    
# Build X and y
y_listed = np.zeros(len(all_images_flat))
run_index = 0
for i, images in enumerate(all_images):
    if i == 0:
        # First response is 0
        run_index = len(images)
    else:
        # Speghetti code
        y_listed[run_index:(run_index + len(images))] = i
        run_index += len(images)
          
# Notice that it takes time to build this, as it is memory heavy, and computationally heavy
X_listed = HDMI.build_X(all_images_flat)

# Randomise such that classes are not ordered
in_random = list(range(0,len(all_images_flat)))
np.random.shuffle(in_random)
X = X_listed[in_random, :]
y = y_listed[in_random]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.33, random_state = 42)

# A simple fitted random forest, no cross validation here - you need to implement that
clf = rndForest(n_estimators = 1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)

HDMI.show_importances(clf, X)
# Accuracy is very high - is it trustworthy?
print("Accuracy at %0.2f %% \n" % score)

#%% make some box plots of each bin of colors, example of exploratory analysis
# these boxplots points at something that might be "off"
fig, ax1 = plt.subplots()
foggy_X = X[np.where(y == 0), :]
nonfoggy_X = X[np.where(y == 1), :]
plt.boxplot((np.ravel(foggy_X[:,1:10]), np.ravel(nonfoggy_X[:,1:10]), np.ravel(foggy_X[:,10:20]),np.ravel(nonfoggy_X[:,10:20]),
             np.ravel(foggy_X[:,20:30]), np.ravel(nonfoggy_X[:,20:30]), np.ravel(foggy_X[:,30:40]), np.ravel(nonfoggy_X[:,30:40])))
plt.xticks([1,2,3,4,5,6,7,8], ["red fog","red nf", "green fog","green nf", "blue fog", "blue nf", "gray fog", "gray nf"])

ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

#%% Distribution of each color channel
# So generally images gets less color saturated when there is fog, or what?
titles = ["Red channel" , "Green channel", "Blue channel"]
fig = plt.figure()
for i in range(3):
    ax = fig.add_subplot(1,3,i + 1)
    data = np.ravel(foggy_X[:, (10*i):((i+1) * 10)])
    ndata = np.ravel(nonfoggy_X[:, (10*i):((i+1) * 10)])
    plt.hist(data, bins = 15, normed = True, alpha = 0.5, color = "blue")
    plt.hist(ndata, bins = 15, normed = True, alpha = 0.5, color ="red")
    
    mu, std = norm.fit(data)
    nmu, nstd = norm.fit(ndata)
    
    x = np.linspace(0, np.max(data), 100)
    p = norm.pdf(x, mu, std)
    _np = norm.pdf(x, nmu, nstd)
    fog = plt.plot(x, p, 'k', linewidth=2, color = 'blue', label = "foggy")
    nonfog = plt.plot(x, _np, 'k', linewidth=2, color = 'red', label = "not foggy")
    plt.legend(["foggy", "not foggy"])
    plt.title(titles[i])
    
