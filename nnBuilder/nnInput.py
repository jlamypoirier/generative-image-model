import numpy as np
import tensorflow as tf
import warnings
from nnLayer import *
#import os
from tensorflow.examples.tutorials.mnist import input_data as mnist




class DataLayer(SimpleLayer):
    type="Abstract"
    def __init__(self,batch=1,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        self.batch=batch                #The number of elements in a batch
    def save(self,**kwargs):
        kwargs["batch"]=self.batch
        return Layer.save(self,**kwargs)

class MNISTLayer(DataLayer):
    type="MNIST"
    def __init__(self,folder='/tmp/tensorflow/mnist/input_data',**kwargs):
        DataLayer.__init__(self,**kwargs)
        self.folder=folder
        self.data=mnist.read_data_sets(self.folder, one_hot=True)
        self.y, self.labels=tf.py_func(lambda :self.data.train.next_batch(self.batch), [], [tf.float32,tf.float64], stateful=True)
        self.y=tf.reshape(self.y, [-1,28,28,1])
        self.labels=tf.cast(self.labels,tf.float32)
    def save(self,**kwargs):
        kwargs["folder"]=self.folder
        return Layer.save(self,**kwargs)