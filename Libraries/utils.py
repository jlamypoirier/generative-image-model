import tensorflow as tf
import numpy as np
import os
from IPython.display import Image
import PIL.Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def defaultinDict(dic,entry,default=None):
    if entry in dic:
        return dic[entry]
    return default

def fixDir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    return dir_
def safeDir(dir_):
    assert(not os.path.exists(dir_))
    return dir_

def showarray(a, fmt='jpeg'):
    a = np.uint8((a-a.min())*255./(a.max()-a.min()))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
def plot(img,cmap=plt.cm.Greys):
    return plt.imshow(img,cmap=cmap, interpolation='nearest')
def plot3d(img=None,cmap='terrain',reg=10):
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    img[0,0]+=reg
    X, Y = np.meshgrid(range(img.shape[0]),range(img.shape[1]))
    plot=ha.plot_surface(X, Y, img,cmap=cmap)
    img[0,0]-=reg
    return plot
def stats(img):
    return (img.min(),img.max(),img.mean(),img.std())