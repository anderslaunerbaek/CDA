
# coding: utf-8

# # Transferlearning

# In[1]:


import numpy as np
from matplotlib.image import imread
from collections import Counter
from scipy.signal import convolve2d



def VARsob(S):
    return(np.sum(np.power(S - np.mean(S), 2)))
def TEN(S):
    return(np.sum(np.power(S, 2)))
    
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

def img_to_nparr(pic_path, img_height, img_width, rat = 1, ch = 3, verbose = True):
    # import
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


# In[2]:


pic_path = np.load("./../data/tmp/pic_path.npy")
pic_path = pic_path[0:2]
n = len(pic_path)


# ## Create target variable and feature matrix

# In[3]:


ratio = 1
channels = 3

pics = img_to_nparr(pic_path=pic_path,
                    img_height = 288, 
                    img_width = 384, 
                    rat = ratio,
                    ch = channels,
                    verbose = False)
#
image_height, image_width, _ = pics[1].shape    
n_pixels = image_height * image_width


# In[4]:


features = ["Dark channel", "sobel_VARsob", "sobel_TEN", 
            "laplace_sum", "laplace_var", "pct_overexposed"]
n_features = len(features)

X = np.zeros((n, n_features))
for ii in range(n):
    print(str(ii + 1) + " of " + str(n), end="\r")
    feature_list = []
    # dark channel
    dc = get_dark_channel(pics[ii], win=20)
    # close to 1 -> presents of fog
    feature_list.append(np.mean(dc / 255.0)) 

    # sobel edge filtering
    S = sobel_filter(pics[ii]),
    feature_list.append(VARsob(S))
    feature_list.append(TEN(S) / n_pixels)

    # laplace
    L = lapalce_filter(pics[ii])
    feature_list.append(np.sum(abs(L)) / n_pixels)
    feature_list.append(np.var(abs(L)) / n_pixels)

    # pct. overexposed pixels
    feature_list.append(overexposed_pixels(pics[ii]) / n_pixels)

    # add to design matrix
    X[ii,:] = feature_list
# if updated save new... 
print("Updated...")
np.save("./../data/tmp/X_transfer.npy", X)

