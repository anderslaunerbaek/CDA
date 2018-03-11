# -*- coding: utf-8 -*-
"""
Fisher discriminant line classification for use in 02582, not working starting point

@author: dnor
"""

import numpy as np
from sklearn.metrics import confusion_matrix

def produceDiscriminantLine(X, S, mu, pi):
    Sinv = np.linalg.inv(S) # How to inverse S
    # First "part" of the calc, something like X * inv(S) * mu' ...
    first = 0
    # Find the "second" part in notes
    second = 0
    return 0 # First - second and something more...

path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture4\\Exercises 4\Data"
dataPath = path + '\\FisherIris.csv'
attributeNames = []

# Dump data file into an array
with open(dataPath, "r") as ins:
    listArray = []
    for line in ins:
        # Remove junk, irritating formating stuff
        listArray.append(line.replace('\n', '').split('\t'))

# Encode data in desired format
n = len(listArray) - 1
p = len(listArray[0][0].split(',')) - 1
X = np.zeros((n, p))
y = np.zeros(n)
for i, data in enumerate(listArray):
    dataTemp = data[0].split(',')
    if i == 0: # first row is attribute names
        attributeNames = dataTemp[0:4]
    else:
        X[i - 1,:] = dataTemp[0:4]
        flowerInd = dataTemp[4]
        if flowerInd == 'Setosa':
            y[i-1] = 0
        elif flowerInd == "Versicolor":
            y[i-1] = 1
        else:
            y[i-1] = 2

# Actual Fisher discriminant done after here
# Looping over pi, mu and S, since calculations are similar for all classes
pi = np.zeros(3)
mu = np.zeros((3, p))
S = np.zeros((p,p))
for i in range(3):
    XSubset = X[np.where(y == i)[0], :] # We only need to work on some of X
    pi[i]  = 0# A fraction, a part, or something else?
    mu[i,:] = 0# Averages of each feature
    S += 0 # Oh no - matrix multiplication is needed here, something with the
    # means and a transposed XSubset
# And remember to scale with number of DF, check result with size of S
# S[0,0] should be something like 0.265
S = S / (n-3)

# Discriminants
d = np.zeros((3, n))
for i in range(3): # We are lazy programmers, therefor we loop
    d[i,:] = produceDiscriminantLine(X, S, mu[i,:], pi[i])
    
# Classify according to discriminant
yhat = np.unravel_index(np.argmax(d, axis=0), d.shape)[1] # index on "1" to get indexes
confusion_matrix(y, yhat) # can also do manually by just checking if y == yhat and counting