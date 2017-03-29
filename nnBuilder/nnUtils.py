import numpy as np
import tensorflow as tf
import os
from IPython.display import clear_output, Image, display, HTML
import PIL.Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from io import BytesIO

def defaultinDict(dic,entry,default=None):
    if entry in dic:
        return dic[entry]
    return default

def itercopy(x):#Copies lists and dictionaries but not objects
    if type(x)==list:
        y=x.copy()
        for i in range(len(y)):
            y[i]=itercopy(y[i])
    elif type(x)==dict:
        y=x.copy()
        for k in y:
            y[k]=itercopy(y[k])
    else:
        y=x
    return y

def fixDir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    return dir_
def safeDir(dir_):
    assert(not os.path.exists(dir_))
    return dir_

def show(a, fmt='jpeg'):
    a = np.uint8((a-a.min())*255./(a.max()-a.min()))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
def plot(img,cmap=plt.cm.Greys):
    y=plt.imshow(img,cmap=cmap, interpolation='nearest')
    plt.grid(None)
    return y
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

def composite(imgs):
    return np.concatenate([np.concatenate([img for img in _imgs],axis=1) for _imgs in imgs],axis=0)

def convertData(File):
    print("Converting file %s"%File)
    assert(os.path.isfile(File+'.npy'))
    with tf.device('/cpu:0'):
        a=np.load(File+'.npy').astype(np.float32)
        a=(a-a.mean())/a.std()
        b=tf.Variable(tf.expand_dims(a,2))
        c=tf.train.Saver({"data":b},max_to_keep=10000)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        #print(sess.run(b))
        c.save(sess,File,write_meta_graph=False)
    return list(a.shape)+[1]
    #with tf.Session() as sess:
        #tf.initialize_all_variables().run()
        #c.restore(sess,File)
        #d=sess.run(b)










