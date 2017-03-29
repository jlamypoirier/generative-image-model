import numpy as np
import tensorflow as tf
#import os
#from IPython.display import clear_output, Image, display, HTML
#import PIL.Image
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#from io import BytesIO


def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap



#k = np.float32([np.exp(-(i)**2/2) for i in range(-2,3)])#np.float32([1,4,6,4,1])
k = np.float32([np.exp(-(i/2)**2/2) for i in range(-4,5)])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()#*np.eye(1, dtype=np.float32)

def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi#, lo2
#lapSplit = tffunc(np.float32)(lap_split)

def _lap_blur(img):
    return tf.nn.conv2d(img, k5x5, [1,1,1,1], 'SAME')
def _lap_down(img):
    return tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
def _lap_up(img):
    return tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(img)*[1,2,2,1], [1,2,2,1])
lap_blur2 = tffunc(np.float32)(_lap_blur)
lap_down2 = tffunc(np.float32)(_lap_down)
lap_up2 = tffunc(np.float32)(_lap_up)
def lap_blur(img):
    return lap_blur2(img[None,:,:,None])[0,:,:,0]
def lap_down(img):
    return lap_down2(img[None,:,:,None])[0,:,:,0]
def lap_up(img):
    return lap_up2(img[None,:,:,None])[0,:,:,0]


def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]
#lapSplitN = tffunc(np.float32)(lap_split_n)

def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    #img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    muls=[1.8**3,1.8**2,1.8,1.]
    for i in range(len(tlevels)):
        tlevels[i]=tlevels[i]*muls[i]
    out = lap_merge(tlevels)
    #return out[0,:,:,:]
    return out













