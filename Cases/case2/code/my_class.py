# -*- coding: utf-8 -*-
"""
Created on tir 17 apr 2018

@author: anders launer bæk
"""

import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.metrics import confusion_matrix

# filtering
from scipy.signal import convolve2d

class my_class:
	''' produces the dark channel prior in RGB space. 
	Parameters
	---------
	Image: M * N * 3 numpy array
	win: Window size for the dark channel prior
	'''
	def performance(pred_test, Y_test):
		cm_test = confusion_matrix(y_pred = pred_test,
			y_true = np.argmax(Y_test, axis=1), 
			labels = list(range(Y_test.shape[1])))

		acc_test = np.sum(np.diag(cm_test)) / np.sum(cm_test)
		print("\n\nAccuracy:\t{0}".format(acc_test))
		print("\nConfusion matrix (pred x true)\n", cm_test)

	def accuracy(pred_test, Y_test):
		cm_test = confusion_matrix(y_pred = pred_test,
			y_true = np.argmax(Y_test, axis=1), 
			labels = list(range(Y_test.shape[1])))

		return(np.sum(np.diag(cm_test)) / np.sum(cm_test))


	def get_dark_channel(image, win = 20):
		''' produces the dark channel prior in RGB space. 
		Parameters
		---------
		Image: M * N * 3 numpy array
		win: Window size for the dark channel prior
		'''
		M, N, _ = image.shape
		pad = int(win / 2)

		# Pad all axis, but not color space
		padded = np.pad(image, ((pad, pad), (pad, pad), (0,0)), 'edge')

		dark_channel = np.zeros((M, N))        
		for i, j in np.ndindex(dark_channel.shape):
			dark_channel[i,j] = np.min(padded[i:i + win, j:j + win, :])

		return dark_channel

	def batch_normalization(X_train, X_val, epsilon=.0001):
		variance = np.var(X_train,0)
		mean = np.mean(X_train,0)
		#
		X_train_norm = (X_train - mean) / (variance + epsilon)
		X_val_norm = (X_val - mean) / (variance + epsilon)
		#
		return X_train_norm, X_val_norm

	def list_pics(start_dir):
		pic_path = []
		for root, directories, filenames in os.walk(start_dir):
			for filename in filenames: 
				pic_path.append(os.path.join(root,filename))       
		#
		return(pic_path)

	def balance(array):
		tmp = dict(Counter(array))
		return(dict(zip([0.0, 1.0],[tmp[0.0] / sum(tmp.values()),tmp[1.0] / sum(tmp.values())])))

	def img_to_nparr(pic_path, img_height, img_width, rat = 1, ch = 3, verbose = True):
		# import
		from matplotlib.image import imread

		# precalculations
		image_height_r = int(img_height / rat)
		image_width_r = int(img_width / rat)
		pics = np.ndarray(shape=(len(pic_path), image_height_r, image_width_r, ch), dtype=np.float64)

		# loop each pics in path
		for ii, pic in enumerate(pic_path):
			#
			try:
				img = imread(pic)
			except:
				print(pic)
			# Convert to Numpy Array
			try:	
				pics[ii] = img.reshape((image_height_r,image_width_r, ch))
			except:
				print(pic)
				print(img.shape)
			if ii % 100 == 0 and verbose:
				print("%d images to array" % ii)
		
		print("All images to array!")
		#
		return(pics)

	def show_img(image, G_xy=False):
		# check if it is a string
		if isinstance(image, str):
			image = imread(image)
		# create picture
		plt.axis("off")
		if image.dtype == np.float and not G_xy:
			plt.imshow(image / 255.0)
		elif G_xy:
			plt.imshow(image, cmap=plt.cm.gray)
		else:
			plt.imshow(image)
		plt.show()

	"""
	asd
	"""
	def VARsob(S):
		return(np.sum(np.power(S - np.mean(S), 2)))
	def TEN(S):
		return(np.sum(np.power(S, 2)))
	
	def rgb_to_grey(img, method = "luminosity"):
		"""
		img = image 
		metod = "luminosity" (as default), "average" or "lightness"
		https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/

		"""
		_, _, ch = img.shape
		assert(ch == 3)
		#
		if method == "luminosity":
			img_grey = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
		elif method == "average":
			img_grey = np.mean(img, axis=2)
		elif method == "lightness":
			img_grey = (np.max(img, axis=2) + np.min(img, axis=2)) / 2
		else:
			img_grey = None
		# 
		return(img_grey)


	def sobel_filter(im, k_size = 3):
		# https://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy
		#
		im = im.astype(np.float)
		width, height, c = im.shape
		# force grayscale...
		if c > 1:
			img = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]
		else:
			img = im
		# check filter sizes
		assert(k_size == 3 or k_size == 5);
		# define filters
		if k_size == 3:
			kh = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
			kv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float)
		else:
			kh = np.array([[-1, -2, 0, 2, 1], [-4, -8, 0, 8, 4], [-6, -12, 0, 12, 6],
				[-4, -8, 0, 8, 4], [-1, -2, 0, 2, 1]], dtype = np.float)
			kv = np.array([[1, 4, 6, 4, 1],  [2, 8, 12, 8, 2], [0, 0, 0, 0, 0], 
				[-2, -8, -12, -8, -2], [-1, -4, -6, -4, -1]], dtype = np.float)

		gx = convolve2d(img, kh, mode='same', boundary = 'symm', fillvalue=0)
		gy = convolve2d(img, kv, mode='same', boundary = 'symm', fillvalue=0)

		S = np.sqrt(gx * gx + gy * gy)
		# normalize
		S *= 255.0 / np.max(S)
		#
		return(S)

	def lapalce_filter(im, k_size = 3):
		# 
		im = im.astype(np.float)
		width, height, c = im.shape
		# force grayscale...
		if c > 1:
			img = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]
		else:
			img = im
		# check filter sizes
		assert(k_size == 3)

		# define filter
		kernel = 1/6 * np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype = np.float)
		L = convolve2d(img, kernel, mode='same', boundary = 'symm', fillvalue=0)
		#
		return(L)

	def overexposed_pixels(img):
		# sum RGB if overexposed
		tmp = np.sum(img == 255.0, 2)
		# max one for each pixel
		tmp[tmp > 1] = 1
		#
		tmp_dict = dict(Counter(tmp.reshape(np.multiply(tmp.shape[0],tmp.shape[1]))))
		return(tmp_dict[1] / tmp_dict[0])



