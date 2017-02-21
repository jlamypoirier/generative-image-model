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

def writeTf(file,data):
    writer=tf.python_io.TFRecordWriter(file)
    writer.write(data.tostring())
    writer.close()




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
            assert(self.y!=None), self._noShapeError
            self.shape=tf.shape(self.y)
        else:
            shape=self.shape
            self.var_dim=self.var_dim or None in self.shape
            for i,s in enumerate(self.shape):
                if s!=None and s<0:
                    assert(self.y!=None), self._noShapeError
                    assert(tf.shape(self.y).ndims==len(shape)), _dimensionError%(tf.shape(self.y).ndims,len(shape))
                    shape[i]=tf.shape(self.y)[i]
            self.shape=tf.concat(0,shape)
    def save(self,**kwargs):
        kwargs["shape"]=self._shape
        return super().save(**kwargs)


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
    

class ConstantLayer(SimpleLayer):     #A constant, loaded at startup from either x or the given file
    type="Constant"
    _noInputError="No initial value given to constant layer"
    def __init__(self,folder="",file=None,label_file=None,convert_file=True,normalize=False,**kwargs):
        super().__init__(**kwargs)
        self.folder=folder                   #(Optional) the folder for the file
        self.file=file                       #The file name, if loading form a file
        self.label_file=label_file           #A file containing the labels
        self.convert_file=convert_file       #Convert the file to Tensorflow format for efficiency (memory, startup time) 
        #                                     (Tensorflow still wastes memory)
        self.normalize=normalize
        self.file_data=None
        self.file_path=os.path.join(folder,file)
        if self.convert_file:
            '''self.tf_file_path=self.file_path+".tf"
            self.shape_file_path=self.file_path+".shape.npy"
            if os.path.isfile(self.tf_file_path+".index") and os.path.isfile(self.shape_file_path):
                self.shape=np.load(self.shape_file_path)
            else:
                self.load_file()
                saveTf(self.tf_file_path,self.file_data)
                np.save(self.shape_file_path,self.shape)
            #self.y=tf.Variable(tf.zeros(shape=self.shape, dtype=tf.float32))
            self.tf_var=tf.Variable(0,dtype=tf.float32,validate_shape=False)
            self.y=tf.reshape(self.tf_var,shape=self.shape)
            self.saver=tf.train.Saver({"data":self.tf_var},max_to_keep=10000)'''
            self.tf_file_path=self.file_path+".tf"
            self.shape_file_path=self.file_path+".shape.npy"
            if os.path.isfile(self.tf_file_path) and os.path.isfile(self.shape_file_path):
                self.shape=np.load(self.shape_file_path)
            else:
                self.load_file()
                writeTf(self.tf_file_path,self.file_data)
                np.save(self.shape_file_path,self.shape)
            reader=tf.read_file(self.tf_file_path)
            decode=tf.decode_raw(reader, tf.float32)[3:-1]#Better way?
            self.y=tf.Variable(tf.reshape(decode,self.shape))
            #self.y=tf.reshape(self.tf_var,shape=self.shape)
        else:
            self.load_file()
            self.y=tf.Variable(self.file_data)
        self.file_data=None #Cleanup
        if self.label_file:
            self.labels=self.add_sublayer(ConstantLayer(x=None,folder=folder,file=label_file,convert_file=convert_file))
    def start(self,sess):
        SimpleLayer.start(self,sess)
        #if self.convert_file:
        #    self.saver.restore(self.sess,os.path.abspath(self.tf_file_path))
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
        return super().save(**kwargs)

#Random layers
class RandomLayer(BasicInputLayer):
    type="Random"
    _shapeError="Random layer must have a fixed shape"
    def __init__(self,rand_type="normal",scale=1.,mean=0.,**kwargs):
        super().__init__(shape=-1,**kwargs)
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
        return super().save(**kwargs)

class NoiseLayer(CombineLayer):
    type="Noise"
    def __init__(self,rand_type="normal",scale=1.,mean=0.,combine_op="add",**kwargs):
        self.rand_type=rand_type             #normal or uniform generator
        self.scale=scale                     #scale for the distribution (std or half-range)
        self.mean=mean                       #mean for the distribution
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_COMBINE_MANAGED,type="Network")
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_COMBINE_MANAGED,type="Random",
                              rand_type=rand_type,scale=scale,mean=mean)
        super().__init__(combine_op=combine_op,**kwargs)
    def save(self,**kwargs):
        kwargs["rand_type"]=self.rand_type
        kwargs["scale"]=self.scale
        kwargs["mean"]=self.mean
        return super().save(**kwargs)

class SoftDropoutLayer(NoiseLayer):#Adapted NoiseLayer (no output normalization)
    type="Soft_Dropout"
    DROP_ON_TEST=True
    def __init__(self,scale=.5,mean=1.,combine_op="mult",**kwargs):#Make adjustable during training?
        super().__init__(combine_op=combine_op,scale=scale,mean=mean,**kwargs)
        









#Input Pipeline (not working)
class PipelineLayer(SimpleLayer):
    type="Pipeline"
    _extra_args=SimpleLayer._extra_args+["min_after_dequeue","batch"]
    def __init__(self,make_batch=False,shuffle=False,num_threads=16,capacity=None,batch_capacity=None,**kwargs):
        super().__init__(**kwargs)
        self.shuffle=shuffle
        self.num_threads=num_threads
        self.make_batch=make_batch
        self._capacity=capacity
        if self.make_batch:
            self.batch="batch" in kwargs and kwargs["batch"] or 128
        else:
            if "batch" in kwargs:
                warnings.warn(self._uncaughtArgumentWarning%"batch")
            self.batch=self.y.get_shape()[0].value
        self.capacity=self._capacity or min(4096,self.batch*16) #Samples, not batches (weird when make_batch=False?)
        data=[self.y]
        labels=self.get_labels()
        if labels!=None:
            data.append(labels)
        if self.shuffle:
            self.min_after_dequeue="min_after_dequeue" in kwargs and kwargs["min_after_dequeue"] or self.capacity//4
            output=tf.train.shuffle_batch(data, batch_size=self.batch, num_threads=self.num_threads, capacity=self.capacity,
                                          enqueue_many=not self.make_batch,min_after_dequeue=self.min_after_dequeue)
        else:
            if "min_after_dequeue" in kwargs:
                warnings.warn(self._uncaughtArgumentWarning%"min_after_dequeue")
            output=tf.train.batch(data, batch_size=self.batch, num_threads=self.num_threads, capacity=self.capacity, 
                                  enqueue_many=not self.make_batch)
        if labels!=None:
            self.y=output[0]
            self.labels=output[1]
        else:
            self.y=output
    def save(self,**kwargs):
        kwargs["make_batch"]=self.make_batch
        kwargs["shuffle"]=self.shuffle
        if self.num_threads!=16:kwargs["num_threads"]=self.num_threads
        if self._capacity!=None:kwargs["capacity"]=self.capacity
        if self.make_batch and self.batch!=128:kwargs["batch"]=self.batch
        if self.shuffle and self.min_after_dequeue!=self.capacity//4:kwargs["min_after_dequeue"]=self.min_after_dequeue
        return super().save(**kwargs)
            

#Label Generators
class IdentityLabelLayer(SimpleLayer):
    type="Identity_Label"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.labels=self.y
    def save(self,**kwargs):
        return super().save(**kwargs)
    
class ConstantLabelLayer(SimpleLayer):
    type="Constant_Label"
    def __init__(self,label,**kwargs):
        super().__init__(**kwargs)
        self.label=np.float32(label)    #Must be numpy array
        tile_n=[self.y.get_shape()[0].value]+[1]*self.label.ndim
        self.labels=tf.constant(np.tile(np.expand_dims(self.label,axis=0),tile_n))
    def save(self,**kwargs):
        kwargs["label"]=self.label
        return super().save(**kwargs)
    
class OneHotLabelLayer(ConstantLabelLayer):
    type="One_Hot_Label"
    def __init__(self,i,n,**kwargs):
        self.i=i
        self.n=n
        label=[0]*n
        label[i]=1
        super().__init__(label=np.float32(label),**kwargs)
    def save(self,**kwargs):
        kwargs["i"]=self.i
        kwargs["n"]=self.n
        return SimpleLayer.save(self,**kwargs)

#Batch Generators
class DataLayer(SimpleLayer):
    type="Abstract"
    def __init__(self,batch=32,**kwargs):
        super().__init__(**kwargs)
        self.batch=batch                #The number of elements in a batch
    def save(self,**kwargs):
        kwargs["batch"]=self.batch
        return super().save(**kwargs)

class BatchIdentityLayer(DataLayer): #Broadcasts a tensor into a batch of identical tensors
    type="Batch_Identity"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.y=tf.tile(tf.expand_dims(self.y,axis=0), [self.batch]+[1]*self.y.get_shape().ndims)

class BatchSliceLayer(DataLayer): #Slices a tensor into batches without shuffling
    type="Batch_Slice"
    def __init__(self,thread_safe=False,**kwargs):
        super().__init__(**kwargs)
        self.thread_safe=thread_safe #Allows running in multiple threads, but breaks self.last_y and and related
        self.data_n=self.y.get_shape()[0].value#tf.shape(self.y)[0]
        self.y=tf.tile(self.y, [2]+[1]*(self.y.get_shape().ndims-1))#(safely) assumes self.batch < self.y.get_shape().values[0]
        self.batch_n=tf.Variable(-1,dtype=tf.int32) #Will be 0 on first batch
        if self.thread_safe:
            self.queue=tf.train.range_input_producer(np.int32(self.batch*self.data_n),shuffle=False)#Not very good
            self.batch_update=self.batch_n.assign(self.queue.dequeue())
        else:
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
            #self.label_begin=tf.pack([tf.mod(self.batch_n*self.batch,self.data_n),0])
            self.label_begin=tf.pack([tf.mod(self.batch_update*self.batch,self.data_n),0])
            self.label_size=size=[self.batch,-1]
            self.labels=tf.slice(labels, begin=self.label_begin,size=self.label_size)
            self.last_labels=tf.slice(labels, begin=self.label_begin_no_update,size=self.label_size)
    def save(self,**kwargs):
        if self.thread_safe:kwargs["thread_safe"]=self.thread_safe
        return super().save(**kwargs)
        
        
'''class RandomCropLayer(SimpleLayer):
    type="Random_Crop"
    def __init__(self,shape=None,batch=False,**kwargs):
        super().__init__(**kwargs)
        self.shape=shape
        self.batch=batch                #If true, perform the operation on a batch
        if self.batch:
            fn=lambda x:tf.random_crop(x,self.shape)
            self.y=tf.map_fn(fn, self.y, parallel_iterations=1024)
        else:
            self.y=tf.random_crop(self.y,self.shape)'''
        
#Input processing
Whitening,BatchWhitening=make_batch_layer(name="Whitening",fun=tf.image.per_image_standardization)
RandomCrop,BatchRandomCrop=make_batch_layer(name="Random_Crop",fun=lambda x,shape:tf.random_crop(x,shape),args=["shape"],shape=None)
CentralCrop,BatchCentralCrop=make_batch_layer(name="Central_Crop",
    fun=lambda x,shape:tf.image.resize_image_with_crop_or_pad(x, target_height=shape[0], target_width=shape[1]),
    args=["shape"],shape=None)
VerticalRandomFlip,BatchVerticalRandomFlip=make_batch_layer(name="Random_Vertical_Flip",fun=tf.image.random_flip_up_down,drop_on_test=True)
HorizontalRandomFlip,BatchHorizontalRandomFlip=make_batch_layer(name="Random_Horizontal_Flip",
    fun=tf.image.random_flip_left_right,drop_on_test=True)
RandomBrightness,BatchRandomBrightness=make_batch_layer(name="Random_Brightness",
    fun=lambda x,max_delta:tf.image.random_brightness(x,max_delta=max_delta),
    args=["max_delta"],max_delta=63,drop_on_test=True)
RandomContrast,BatchRandomContrast=make_batch_layer(name="Random_Contrast",
    fun=lambda x,lower,upper:tf.image.random_contrast(x,lower=lower,upper=upper),
    args=["lower","upper"],lower=0.2, upper=1.8,drop_on_test=True)


#Standard datasets
class MNISTLayer(DataLayer): 
    type="MNIST"
    _def_folder="/tmp/tensorflow/mnist/input_data"
    def __init__(self,folder=_def_folder,**kwargs):
        super().__init__(**kwargs)
        #self.test=test
        self.folder=folder
        self.data=mnist.read_data_sets(self.folder, one_hot=True)
        if self.test:
            print("test")
            self.y, self.labels=self.data.test.images, self.data.test.labels
            self.y=tf.reshape(self.y, [10000,28,28,1])
        else:
            print("train")
            self.y, self.labels=tf.py_func(lambda :self.data.train.next_batch(self.batch), [], [tf.float32,tf.float64], stateful=True)
            self.y=tf.reshape(self.y, [self.batch,28,28,1])
        self.labels=tf.cast(self.labels,tf.float32)
    def save(self,**kwargs):
        if self.folder!=self._def_folder:kwargs["folder"]=self.folder
        #if self.test:kwargs["test"]=self.test
        return super().save(**kwargs)


class CIFARLayer(DataLayer):
    type="CIFAR_10"
    url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    files_train=["data_batch_%i"%i for i in range(1,6)]
    files_test=["test_batch"]
    n_train=50000
    n_test=10000
    n_classes=10
    sample_shape=[32,32,3]
    def __init__(self,folder='/tmp/cifar10_data',**kwargs):
        super().__init__(**kwargs)
        #self.test=test
        self.folder=folder
        self.file_name = self.url.split('/')[-1]
        self.file_path = os.path.join(self.folder, self.file_name)
        self.file_base_name=self.file_name.split(".")[0]
        if self.test:
            self.file_data_path=fixDir(os.path.join(self.folder,"test"))
        else:
            self.file_data_path=fixDir(os.path.join(self.folder,"train"))
        self.file_base_path=os.path.join(self.file_data_path,self.file_base_name)
        self.label_base_path=self.file_base_path+".labels"
        #self.tf_file_path=self.file_base_path+".tf"
        #self.label_file_path=self.file_base_path+".labels"
        #self.shape_file_path=self.file_base_path+".shape.npy"
        #if not (os.path.isfile(self.tf_file_path+".index") and 
        #        os.path.isfile(self.label_file_path+".tf.index") and os.path.isfile(self.shape_file_path)):
        if not (os.path.isfile(self.file_base_path+".tf") and os.path.isfile(self.file_base_path+".shape.npy") and 
                os.path.isfile(self.label_base_path+".tf") and os.path.isfile(self.label_base_path+".shape.npy")):
            self.convert_data()
        self.data=self.add_sublayer(ConstantLayer(
                folder=self.file_data_path,file=self.file_base_name,label_file=self.file_base_name+".labels",
                convert_file=True,normalize=True))
        self.y=self.data.get()
        self.labels=self.data.labels
    def save(self,**kwargs):
        kwargs["folder"]=self.folder
        #if self.test:kwargs["test"]=self.test
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
        files=self.test and self.files_test or self.files_train
        n=self.test and self.n_test or self.n_train
        folder=os.path.join(self.folder,"cifar-10-batches-py")
        def unpickle(f):
            file = open(f, 'rb')
            dic = pickle.load(file,encoding='latin1')
            file.close()
            return dic
        dics=[unpickle(os.path.join(folder,file)) for file in files]
        data=np.moveaxis(np.array([dic['data'] for dic in dics]).reshape(
                        [n,self.sample_shape[2]]+self.sample_shape[:2]),1,3).astype(np.float32)
        labels=np.array([dic['labels'] for dic in dics]).reshape([n])
        labels_one_hot=np.array([[label==i for i in range(self.n_classes)] for label in labels],dtype=np.float32)
        writeTf(self.file_base_path+".tf",data)
        writeTf(self.label_base_path+".tf",labels_one_hot)
        #saveTf(self.tf_file_path,data)
        #saveTf(self.label_file_path+".tf",labels_one_hot)
        np.save(self.file_base_path+".shape.npy",[n]+self.sample_shape)
        np.save(self.label_base_path+".shape.npy",[n,self.n_classes])
        
        
        














