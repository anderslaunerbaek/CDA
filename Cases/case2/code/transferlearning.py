
# coding: utf-8

# # Transferlearning

# In[ ]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from my_class import my_class as my
get_ipython().magic('matplotlib inline')


# In[ ]:


# Set directories
data_path_clear = "./../data/data/clear/billund/2016/"
data_path_foggy = "./../data/data/foggy/billund/2017/"


# In[ ]:


pic_path_clear = my.list_pics(data_path_clear)
pic_path_foggy = my.list_pics(data_path_foggy)
pic_path_clear = [pic_path_clear[ii] for ii in range(len(pic_path_clear)) if ".jpg" in pic_path_clear[ii]]
pic_path_foggy = [pic_path_foggy[ii] for ii in range(len(pic_path_foggy)) if ".jpg" in pic_path_foggy[ii]]


#n_sample = 10
#pic_path_clear = np.random.choice(pic_path_clear, size=n_sample, replace=False)
#pic_path_foggy = np.random.choice(pic_path_foggy, size=n_sample, replace=False)

pic_path = np.concatenate((pic_path_clear, pic_path_foggy))
# pic_path = [pic_path[ii] for ii in range(len(pic_path)) if ".jpg" in pic_path[ii]]

n_clear = len(pic_path_clear)
n_foggy = len(pic_path_foggy)
n = len(pic_path)


# ## Create target variable and feature matrix

# In[ ]:


Y_clear = np.zeros(n_clear, dtype=int)
Y_foggy = np.ones(n_foggy, dtype=int)
Y = np.concatenate((Y_clear, Y_foggy), axis=0)
# balance(Y)
# one hot
b = np.zeros((len(Y), len(set(Y))))
b[np.arange(len(Y)), Y] = 1
Y = b
n_classes = Y.shape[1]
classes = ["clear", "foggy"]


# In[ ]:


ratio = 1
channels = 3
update = True
#
if update:
    pics = my.img_to_nparr(pic_path=pic_path, 
                           img_height = 288, 
                           img_width = 384, 
                           rat = ratio,
                           ch = channels,
                           verbose = False)

    # only consider the 3 /5 top of the picture...    
    #pics = pics[:, 0:int(pics.shape[1] / 5 * 4),:,:]
    # dimensions picture
    image_height, image_width, _ = pics[1].shape
    
n_pixels = image_height * image_width


# ## Feature extraction

# In[ ]:


features = ["Dark channel", "sobel_VARsob", "sobel_TEN", 
            "laplace_sum", "laplace_var", "pct_overexposed"]
n_features = len(features)


# In[ ]:


if update:
    X = np.zeros((n, n_features))
    for ii in range(n):
        print(str(ii + 1) + " of " + str(n), end="\r")
        feature_list = []
        # dark channel
        dc = my.get_dark_channel(pics[ii], win=20)
        # close to 1 -> presents of fog
        feature_list.append(np.mean(dc / 255.0)) 

        # sobel edge filtering
        S = my.sobel_filter(pics[ii]),
        feature_list.append(my.VARsob(S))
        feature_list.append(my.TEN(S) / n_pixels)
        
        # laplace
        L = my.lapalce_filter(pics[ii])
        feature_list.append(np.sum(abs(L)) / n_pixels)
        feature_list.append(np.var(abs(L)) / n_pixels)
        
        # pct. overexposed pixels
        feature_list.append(my.overexposed_pixels(pics[ii]) / n_pixels)
        
        # add to design matrix
        X[ii,:] = feature_list
    # if updated save new... 
    print("Updated...")
    np.save("./../data/tmp/X_transfer.npy", X)
else:
    X = np.load("./../data/tmp/X_transfer.npy")
    


# ## Model assessment
# ### RF

# In[ ]:


# load model
from sklearn.externals import joblib
clf = joblib.load("./../data/tmp/clf.pkl")
# make predictions
pred = clf.predict(X)
my.performance(pred, Y)


# ### KNN

# In[ ]:


# load model
neigh = joblib.load("./../data/tmp/neigh.pkl")
# make predictions
pred = neigh.predict(X)
my.performance(pred, Y)


# ### Create TF graph

# In[ ]:


# reset
tf.reset_default_graph()

# Parameters
learning_rate = 0.01
training_epochs = 50000
batch_size = 100
display_step = 1000
display_step_state = False

# tf Graph Input
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_classes])

# Set model weights
W = tf.Variable(tf.zeros([n_features, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Test model
prediction = tf.argmax(pred, 1)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


# ## Model assessment

# In[ ]:


# get the test accuracy
with tf.Session() as sess:
    # restore variables
    saver.restore(sess, "./../data/models/model_final.ckpt")
    print("Model restored.")
    # batch norm
    X_model = np.load("./../data/tmp/X_model.npy")
    _, X_bn = my.batch_normalization(X_model, X, epsilon=.0001)
    
    # Check the values of the variables
    acc_test, pred_test = sess.run([accuracy, prediction], feed_dict={x: X_bn, y: Y})
    my.performance(pred_test, Y)
    
    Weights, bias = sess.run([W, b])
    

