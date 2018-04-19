
# coding: utf-8

# # Supervised Learning

# In[1]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from my_class import my_class as my
get_ipython().magic('matplotlib inline')


# In[2]:


# Set directories
data_path_clear = "./../data/data/clear/skive/2016/"
data_path_foggy = "./../data/data/foggy/skive/2016/"


# In[3]:


pic_path_clear = my.list_pics(data_path_clear)
pic_path_foggy = my.list_pics(data_path_foggy)
pic_path = pic_path_clear + pic_path_foggy
pic_path = [pic_path[ii] for ii in range(len(pic_path)) if ".jpg" in pic_path[ii]]

n_clear = len(pic_path_clear)
n_foggy = len(pic_path_foggy)
n = len(pic_path)


# ## Create target variable and feature matrix

# In[4]:


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


# In[5]:


ratio = 1
channels = 3
pics = my.img_to_nparr(pic_path=pic_path, 
                       img_height = 576, 
                       img_width = 704, 
                       rat = ratio,
                       ch = channels,
                       verbose = False)
# only consider the 3 /5 top of the picture...
# pics = pics[:, 0:int(pics.shape[1] / 5 * 4),:,:]
# dimensions picture
image_height, image_width, _ = pics[1].shape

n_pixels = image_height * image_width


# ## Feature extraction

# In[6]:


features = ["Dark channel", "sobel_VARsob", "sobel_TEN", 
            "laplace_sum", "laplace_var", "pct_overexposed"]
n_features = len(features)


# In[ ]:


update = True
#
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
    np.save("./../data/tmp/X.npy", X)
else:
    X = np.load("./../data/tmp/X.npy")


# ## Modelling

# In[ ]:


# Randomize the order of the pictures
idx = np.arange(n)
np.random.shuffle(idx)
#
pic_path = np.array(pic_path)[idx]
Y = Y[idx]
X = X[idx]


# In[ ]:


#
test_size = 0.3
rand_state = 22
K = 2
#Splitting 
#X_model, X_test, Y_model, Y_test = train_test_split(X, Y,
#                                                    test_size = test_size,
#                                                    random_state = rand_state)
idx_X_model, idx_X_test, idx_Y_model, idx_Y_test = train_test_split(np.arange(n),np.arange(n),
                                                                    test_size = test_size,
                                                                    random_state = rand_state)
# devide data
X_model, X_test, Y_model, Y_test = X[idx_X_model], X[idx_X_test], Y[idx_Y_model], Y[idx_Y_test]
pic_path_model, pic_path_test = pic_path[idx_X_model], pic_path[idx_X_test]

np.save("./../data/tmp/X_model.npy", X_model)


print("Train and val. size:\t{0}\nTest set size:\t\t{1}".format(len(X_model), len(X_test)))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#
clf = RandomForestClassifier(n_estimators = 500,
                             n_jobs = -1,
                             random_state=rand_state,
                             max_features = None,
                             min_samples_split = 2,
                             class_weight = {0: Y_model.shape[0] / (n_classes * np.bincount(np.argmin(Y_model, 1)))[0], 
                                             1: Y_model.shape[0] / (n_classes * np.bincount(np.argmin(Y_model, 1)))[1]},
                             max_depth=None)
# fit tree
clf.fit(X_model, np.argmax(Y_model, 1))

from sklearn.externals import joblib
joblib.dump(clf, "./../data/tmp/clf.pkl") 

# clf = joblib.load("./../data/tmp/clf.pkl")

pred_test = clf.predict(X_test)
my.performance(pred_test, Y_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
#
kk_neigh = [2,3,4,5,6,7,8,9,10]
class_error_rate = []
for kk in kk_neigh:
    neigh = KNeighborsClassifier(n_neighbors=kk,n_jobs=-1)
    neigh.fit(X_model, np.argmax(Y_model, 1)) 
    #
    pred_test = neigh.predict(X_test)
    class_error_rate.append(my.accuracy(pred_test, Y_test))
#
neigh = KNeighborsClassifier(n_neighbors=kk_neigh[np.argmax(class_error_rate)],
                             n_jobs=-1)
neigh.fit(X_model, np.argmax(Y_model, 1)) 
joblib.dump(neigh, "./../data/tmp/neigh.pkl") 

pred_test = neigh.predict(X_test)
my.performance(pred_test, Y_test)


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


# ### Train model

# In[ ]:


# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

# Start training
with tf.Session() as sess:
    # run the initializer
    sess.run(init)
    X_model_bn, X_test_bn = my.batch_normalization(X_model, X_test, epsilon=.0001)
    # Training cycle starts
    avg_cost = 0
    for epoch in range(training_epochs):  
        # display logs per epoch step
        if ((epoch+1) % display_step == 0) and display_step_state:
            print("Ep:", '%04d' % (epoch+1), 
                  "\n\tcost = {:.5f}\t{:.5f}\t{:.5f}".format(avg_cost/epoch,
                                                             sess.run(accuracy, feed_dict={x: X_model_bn, y: Y_model}),
                                                             sess.run(accuracy, feed_dict={x: X_test_bn, y: Y_test})))
        # 
        _, c = sess.run([optimizer, cost], feed_dict={x: X_model_bn, y: Y_model})

        # aggregate loss
        avg_cost += c

    # model performance
    acc_model = sess.run(accuracy, feed_dict={x: X_model_bn, y: Y_model})
    acc_test = sess.run(accuracy, feed_dict={x: X_test_bn, y: Y_test})

    # save model for folds
    save_path = saver.save(sess, "./../data/models/model_final.ckpt")
    print("Model final saved in path: %s" % save_path)

print("\n\nFinal model performance: \n\tAccuracy model:\t{:.5f}\n\tAccuracy test:\t{:.5f}".format(acc_model, acc_test))


# ## Model assessment

# In[ ]:


# get the test accuracy
with tf.Session() as sess:
    # restore variables
    saver.restore(sess, "./../data/models/model_final.ckpt")
    print("Model restored.")
    # batch norm
    _, X_test_bn = my.batch_normalization(X_model, X_test, epsilon=.0001)
    
    # Check the values of the variables
    acc_test, pred_test = sess.run([accuracy, prediction], feed_dict={x: X_test_bn, y: Y_test})
    my.performance(pred_test, Y_test)
    
    Weights, bias = sess.run([W, b])
    

