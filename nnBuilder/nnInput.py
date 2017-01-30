import numpy as np
import tensorflow as tf
import warnings
from nnLayer import *
from _nnUtils import *
import os
import sys
import tarfile
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
        tf_data=tf.Variable(data,dtype=tf.float32)
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
        super().__init__(**kwargs)
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
        return super().save(self,**kwargs)


class PlaceholderLayer(BasicInputLayer):     
    #Warning: TensorFlow doesn't always like these:
    #    Doesn't work with derivatives even if no input is fed
    #    Whole shape is forgotten if any component is set to None
    type="Placeholder"
    def __init__(self,ignore_shape=False,**kwargs):
        super().__init__(**kwargs)
        self.y=tf.placeholder_with_default(self.y,self.shape)

class InputLayer(BasicInputLayer):
    type="Input"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.y=tf.placeholder(dtype=tf.float32,shape=self.shape)
    
class RandomLayer(BasicInputLayer):
    type="Random"
    _shapeError="Random layer must have a fixed shape"
    def __init__(self,rand_type="normal",scale=1.,mean=0.,**kwargs):
        super().__init__(**kwargs)
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
        return super().save(self,**kwargs)
    
class ConstantLayer(SimpleLayer):     #A constant, loaded at startup from either x or the given file
    type="Constant"
    _noInputError="No initial value given to constant layer"
    def __init__(self,folder="",file=None,label_file=None,convert_file=True,normalize=False,**kwargs):
        super().__init__(**kwargs)
        self.folder=folder                   #(Optional) the folder for the file
        self.file=file                       #The file name, if loading form a file
        self.label_file=label_file           #A file containing the labels
        self.convert_file=convert_file       #Convert the file to Tensorflow format for efficiency (memory, startup time)
        self.normalize=normalize
        self.file_data=None
        self.file_path=os.path.join(folder,file)
        if self.convert_file:
            self.tf_file_path=self.file_path+".tf"
            self.shape_file_path=self.file_path+".shape.npy"
            if os.path.isfile(self.tf_file_path+".index") and os.path.isfile(self.shape_file_path):
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
        if self.label_file:
            self.labels=self.add_sublayer(ConstantLayer(x=None,folder=folder,file=label_file,convert_file=convert_file))
    def start(self,sess):
        SimpleLayer.start(self,sess)
        if self.convert_file:
            self.saver.restore(self.sess,os.path.abspath(self.tf_file_path))
    def load_file(self):
        if self.file_data==None:
            assert(os.path.isfile(self.file)), 'Cannot find file "%s"'%self.file
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
        return super().save(self,**kwargs)





#Batch Generators
class DataLayer(SimpleLayer):
    type="Abstract"
    def __init__(self,batch=32,**kwargs):
        super().__init__(**kwargs)
        self.batch=batch                #The number of elements in a batch
    def save(self,**kwargs):
        kwargs["batch"]=self.batch
        return super().save(self,**kwargs)

class BatchIdentityLayer(DataLayer): #Broadcasts a tensor into a batch of identical tensors
    type="Batch_Identity"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.y=tf.tile(tf.expand_dims(self.y,axis=0), [self.batch]+[1]*self.y.get_shape().ndims)

class BatchSliceLayer(DataLayer): #Slices a tensor into batches without shuffling
    type="Batch_Slice"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.data_n=tf.shape(self.y)[0]
        self.y=tf.tile(self.y, [2]+[1]*(self.y.get_shape().ndims-1))#(safely) assumes self.batch < self.y.get_shape().values[0]
        self.batch_n=tf.Variable(-1,dtype=tf.int32) #Will be 0 on first batch
        self.batch_update=self.batch_n.assign_add(1)
        self.ndims=self.y.get_shape().ndims-1  #Excludes batch dimension
        self.begin_no_update=tf.pack([tf.mod(self.batch_n*self.batch,self.data_n)]+[0]*self.ndims)
        self.begin=tf.pack([tf.mod(self.batch_update*self.batch,self.data_n)]+[0]*self.ndims)
        self.size=size=[self.batch]+[-1]*self.ndims
        #The epoch of the last element of the last generated batch (starts at 0)
        self.epoch=tf.div((self.batch_n+1)*self.batch-1,self.data_n)
        self.last_y=tf.slice(self.y, begin=self.begin_no_update,size=self.size)
        self.y=tf.slice(self.y, begin=self.begin,size=self.size)
        labels=self.get_input_labels()
        if labels!=None:
            labels=tf.tile(labels,[2,1])
            self.label_begin_no_update=tf.pack([tf.mod(self.batch_n*self.batch,self.data_n),0])
            self.label_begin=tf.pack([tf.mod(self.batch_update*self.batch,self.data_n),0])
            self.label_size=size=[self.batch,-1]
            self.labels=tf.slice(labels, begin=self.label_begin,size=self.label_size)
            self.last_labels=tf.slice(labels, begin=self.label_begin_no_update,size=self.label_size)
        
        
        
        
#Input processing
class RandomCropLayer(SimpleLayer):
    type="Random_Crop"
    def __init__(self,shape=None,batch=False,**kwargs):
        super().__init__(**kwargs)
        self.shape=shape
        self.batch=batch                #If true, perform the operation on a batch
        if self.batch:
            fn=lambda x:tf.random_crop(x,self.shape)
            self.y=tf.map_fn(fn, self.y, parallel_iterations=1024)
        else:
            self.y=tf.random_crop(self.y,self.shape)



#Standard datasets
class MNISTLayer(DataLayer): 
    type="MNIST"
    def __init__(self,folder='/tmp/tensorflow/mnist/input_data',**kwargs):
        super().__init__(**kwargs)
        self.folder=folder
        self.data=mnist.read_data_sets(self.folder, one_hot=True)
        self.y, self.labels=tf.py_func(lambda :self.data.train.next_batch(self.batch), [], [tf.float32,tf.float64], stateful=True)
        self.y=tf.reshape(self.y, [-1,28,28,1])
        self.labels=tf.cast(self.labels,tf.float32)
    def save(self,**kwargs):
        kwargs["folder"]=self.folder
        return super().save(self,**kwargs)


class CIFARLayer(DataLayer):
    type="CIFAR_10"
    def __init__(self,folder='/tmp/cifar10_data',**kwargs):
        super().__init__(**kwargs)
        self.folder=folder
        self.url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.file_name = self.url.split('/')[-1]
        self.file_path = os.path.join(self.folder, self.file_name)
        self.file_base_name=self.file_name.split(".")[0]
        self.file_base_path=os.path.join(self.folder,self.file_base_name)
        self.tf_file_path=self.file_base_path+".tf"
        self.label_file_path=self.file_base_path+".labels"
        self.shape_file_path=self.file_base_path+".shape.npy"
        if not (os.path.isfile(self.tf_file_path+".index") and 
                os.path.isfile(self.label_file_path+".tf.index") and os.path.isfile(self.shape_file_path)):
            self.convert_data()
        self.data=self.add_sublayer(ConstantLayer(
                folder=self.folder,file=self.file_base_name,label_file=self.file_base_name+".labels",convert_file=True,normalize=True))
        self.y=self.data.get()
        self.labels=self.data.labels
    def save(self,**kwargs):
        kwargs["folder"]=self.folder
        return super().save(self,**kwargs)
    def start(self,sess):
        SimpleLayer.start(self,sess)
        self.data.start(sess) 
    def stop(self):
        SimpleLayer.stop(self)
        self.data.stop() 
    def download(self):#Based on https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
        if not os.path.exists(self.file_path):
            from six.moves import urllib
            fixDir(self.folder)
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (self.file_name,
                                                                 float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            urllib.request.urlretrieve(self.url, self.file_path, _progress)
            print()
            statinfo = os.stat(self.file_path)
            print('Successfully downloaded', self.file_name, statinfo.st_size, 'bytes.')
        tarfile.open(self.file_path, 'r:gz').extractall(self.folder)
    def convert_data(self):
        import pickle
        self.download()
        files=["data_batch_%i"%i for i in range(1,6)]
        folder=os.path.join(self.folder,"cifar-10-batches-py")
        def unpickle(f):
            file = open(f, 'rb')
            dic = pickle.load(file,encoding='latin1')
            file.close()
            return dic
        dics=[unpickle(os.path.join(folder,file)) for file in files]
        data=np.moveaxis(np.array([dic['data'] for dic in dics]).reshape([50000,3,32,32]),1,3).astype(np.float32)
        labels=np.array([dic['labels'] for dic in dics]).reshape([50000])
        labels_one_hot=np.array([[label==i for i in range(10)] for label in labels],dtype=np.float32)
        saveTf(self.tf_file_path,data)
        saveTf(self.label_file_path+".tf",labels_one_hot)
        np.save(self.shape_file_path,[50000,32,32,3])
        np.save(self.label_file_path+".shape.npy",[50000,10])
        
        
        














