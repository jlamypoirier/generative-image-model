import numpy as np
import tensorflow as tf
import warnings
from nnLayer import *
import os
import PIL.Image
from tensorflow.examples.tutorials.mnist import input_data as mnist

def load_data(file):
    assert(os.path.isfile(file)), 'Cannot find file "%s"'%file
    try:#Numpy file
        return np.load(file).astype(np.float32)
    except:
        pass
    try:#Image file
        return np.array(PIL.Image.open(file),dtype=np.float32)
    except:
        pass
    try:#GeoTiff file
        import gdal
        return gdal.Open(file).GetRasterBand(1).ReadAsArray().astype(np.float32)
    except:
        pass
    raise Exception('Cannot load file "%s": unsupported file type'%file)
    
def saveTf(file,data):#No proper cleanup
    with tf.device('/cpu:0'):#Avoid wasting gpu memory
        tf_data=tf.Variable(data)
        saver=tf.train.Saver({"data":tf_data},max_to_keep=10000)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(file)
        saver.save(sess,os.path.abspath(file),write_meta_graph=False)







###Input and random layers
class BasicInputLayer(SimpleLayer):
    type="Abstract"
    _noShapeError="Can't deduce the shape from the input: no input tensor given"
    _dimensionError="The input tensor and the given shape have different dimension: %s vs %s"
    def __init__(self,shape=None,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        self._shape=shape                    #Shape of the tensor, or None to keep arbitrary, or -1 to deduce from x
        self.shape=shape                     #Can set individual components to None or -1
        self.var_dim=self.shape==None
        if self.shape==None:
            pass
        elif type(self.shape) == int and self.shape<0:
            assert(x!=None), self._noShapeError
            self.shape=tf.shape(x)
        else:
            shape=self.shape
            self.var_dim=self.var_dim or None in self.shape
            for i,s in enumerate(self.shape):
                if s!=None and s<0:
                    assert(x!=None), self._noShapeError
                    assert(tf.shape(x).ndims==len(shape)), _dimensionError%(tf.shape(x).ndims,len(shape))
                    shape[i]=tf.shape(x)[i]
            self.shape=tf.concat(0,shape)
    def save(self,**kwargs):
        kwargs["shape"]=self._shape
        return SimpleLayer.save(self,**kwargs)


class PlaceholderLayer(BasicInputLayer):     
    #Warning: TensorFlow doesn't always like these:
    #    Doesn't work with derivatives even if no input is fed
    #    Whole shape is forgotten if any component is set to None
    type="Placeholder"
    def __init__(self,ignore_shape=False,**kwargs):
        BasicInputLayer.__init__(self,**kwargs)
        self.y=tf.placeholder_with_default(self.y,self.shape)

class InputLayer(BasicInputLayer):
    type="Input"
    def __init__(self,**kwargs):
        BasicInputLayer.__init__(self,**kwargs)
        self.y=tf.placeholder(dtype=tf.float32,shape=self.shape)
    
class RandomLayer(BasicInputLayer):
    type="Random"
    _shapeError="Random layer must have a fixed shape"
    def __init__(self,rand_type="normal",scale=1.,mean=0.,**kwargs):
        BasicInputLayer.__init__(self,**kwargs)
        self.rand_type=rand_type             #normal or uniform generator
        self.scale=scale                     #scale for the distribution (std or half-range)
        self.mean=mean                       #mean for the distribution
        assert(not self.var_dim), self._shapeError
        if self.rand_type=="normal":
            self.y=tf.random_normal(self.shape, mean=self.mean, 
                                    stddev=self.scale,dtype=tf.float32)
        elif self.rand_type=="uniform":
            self.y=tf.random_uniform(self.shape,minval=self.mean-self.scale,
                                     maxval=self.mean+self.scale,dtype=tf.float32)
    def save(self,**kwargs):
        kwargs["rand_type"]=self.rand_type
        kwargs["scale"]=self.scale
        kwargs["mean"]=self.mean
        return BasicInputLayer.save(self,**kwargs)
    
class ConstantLayer(SimpleLayer):     #A constant, loaded at startup from either x or the given file
    type="Constant"
    _noInputError="No initial value given to constant layer"
    def __init__(self,folder="",file=None,convert_file=True,normalize=False,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        self.folder=folder                   #(Optional) the folder for the file
        self.file=file                       #The file name, if loading form a file
        self.convert_file=convert_file       #Convert the file to Tensorflow format for efficiency (memory, startup time)
        self.normalize=normalize
        self.file_data=None
        self.file_path=os.path.join(folder,file)
        assert(os.path.isfile(file)), 'Cannot find file "%s"'%file
        if self.convert_file:
            self.tf_file_path=self.file_path+".tf"
            self.shape_file_path=self.file_path+".shape"
            if os.path.isfile(self.tf_file_path) and os.path.isfile(self.shape_file_path):
                self.shape=np.load(self.shape_file_path)
            else:
                self.load_file()
                saveTf(self.tf_file_path,self.file_data)
                np.save(self.shape_file_path,self.shape)
            self.y=tf.Variable(tf.zeros(shape=self.shape, dtype=tf.float32))
            self.saver=tf.train.Saver({"data":self.y},max_to_keep=10000)
        else:
            self.load_file()
            self.y=tf.Variable(self.file_data)
        self.file_data=None #Cleanup
    def start(self,sess):
        SimpleLayer.start(self,sess)
        if self.convert_file:
            self.saver.restore(self.sess,os.path.abspath(self.tf_file_path))
    def load_file(self):
        if self.file_data==None:
            self.file_data=load_data(self.file_path)
            self.shape=self.file_data.shape
            if self.normalize:
                self.mean=self.file_data.mean()
                self.std=self.file_data.std()
                self.file_data=(self.file_data-self.mean)/self.std
    def save(self,**kwargs):
        kwargs["folder"]=self.folder
        kwargs["file"]=self.file
        kwargs["convert_file"]=self.convert_file
        kwargs["normalize"]=self.normalize
        return SimpleLayer.save(self,**kwargs)
        

















#Batch Generators
class DataLayer(SimpleLayer):
    type="Abstract"
    def __init__(self,batch=None,**kwargs):
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


class RandomCropLayer(DataLayer):
    type="Random_Crop"
    def __init__(self,**kwargs):
        DataLayer.__init__(self,shape,**kwargs)
        self.shape=shape
        
        
    























