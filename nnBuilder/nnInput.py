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
    def init(self,kwargs):
        super().init(kwargs)
        self.shape=kwargs.pop("shape",None)  #Shape of the tensor, or None to keep arbitrary, or -1 to deduce from x
                                             #Can set individual components to None or -1
    def call(self):
        super().call()
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
                    #assert(tf.shape(self.y).ndims==len(shape)), _dimensionError%(tf.shape(self.y).ndims,len(shape))
                    #shape[i]=tf.shape(self.y)[i]
                    tf_shape=self.y.get_shape().as_list()
                    assert(len(tf_shape)==len(shape)), _dimensionError%(tf.shape(self.y).ndims,len(shape))
                    shape[i]=tf_shape[i]
            #print(shape)
            #self.shape=tf.concat(0,shape)


class PlaceholderLayer(BasicInputLayer):     
    #Warning: TensorFlow doesn't always like these:
    #    Doesn't work with derivatives even if no input is fed
    #    Whole shape is forgotten if any component is set to None
    type="Placeholder"
    def call(self):
        super().call()
        self.y=tf.placeholder_with_default(self.y,self.shape)

class InputLayer(BasicInputLayer):
    type="Input"
    def call(self):
        super().call()
        self.y=tf.placeholder(dtype=tf.float32,shape=self.shape)
    

class ConstantLayer(SimpleLayer):     #A constant, loaded at startup from either x or the given file
    type="Constant"
    _noInputError="No initial value given to constant layer"
    def init(self,kwargs):
        super().init(kwargs)
        self.folder=kwargs.pop("folder","")  #(Optional) the folder for the file
        self.file=kwargs.pop("file")         #The file name
        self.label_file=kwargs.pop("label_file",None)#A file containing the labels
        self.convert_file=kwargs.pop("convert_file",True)#Convert the file to Tensorflow format for efficiency (memory, startup time) 
        #                                     (Tensorflow still wastes memory)
        self.normalize=kwargs.pop("normalize",False)
        self.file_data=None
        self.file_path=os.path.join(self.folder,self.file)
    def call(self):
        super().call()
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
            self.labels=self.add_sublayer(ConstantLayer(x=None,folder=self.folder,file=self.label_file,convert_file=self.convert_file))
    def load_file(self):
        if self.file_data==None:
            assert(os.path.isfile(self.file)), 'Cannot find file "%s"'%self.file
            self.file_data=load_data(self.file_path)
            self.shape=self.file_data.shape
            if self.normalize:
                self.mean=self.file_data.mean()
                self.std=self.file_data.std()
                self.file_data=(self.file_data-self.mean)/self.std

#Random layers
class RandomLayer(BasicInputLayer):
    type="Random"
    _shapeError="Random layer must have a fixed shape"
    def init(self,kwargs):
        if "shape" not in kwargs:
            kwargs["shape"]=-1
        super().init(kwargs)
        self.rand_type=kwargs.pop("rand_type","normal") #normal or uniform generator
        self.scale=kwargs.pop("scale",1.)               #scale for the distribution (std or half-range)
        self.mean=kwargs.pop("mean",0.)                 #mean for the distribution
    def call(self):
        super().call()
        assert(not self.var_dim), self._shapeError
        if self.rand_type=="normal":
            self.y=tf.random_normal(self.shape, mean=self.mean, 
                                    stddev=self.scale,dtype=tf.float32)
        elif self.rand_type=="uniform":
            self.y=tf.random_uniform(self.shape,minval=self.mean-self.scale,
                                     maxval=self.mean+self.scale,dtype=tf.float32)

class NoiseLayer(CombineLayer):
    type="Noise"#GaussianNoise
    def init(self,kwargs):
        if "combine_op" not in kwargs:
            kwargs["combine_op"]="add"
        super().init(kwargs)
        self.rand_type=kwargs.pop("rand_type","normal") #normal or uniform generator
        self.scale=kwargs.pop("scale",1.)               #scale for the distribution (std or half-range)
        self.mean=kwargs.pop("mean",0.)                 #mean for the distribution
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_COMBINE_MANAGED,type="Network")
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_COMBINE_MANAGED,type="Random",
                              rand_type=self.rand_type,scale=self.scale,mean=self.mean)

class SoftDropoutLayer(NoiseLayer):#Adapted NoiseLayer (no output normalization)
    type="Soft_Dropout"#GaussianDropout
    DROP_ON_TEST=True
    def init(self,kwargs):
        if "scale" not in kwargs:
            kwargs["scale"]=.5
        if "mean" not in kwargs:
            kwargs["mean"]=1.
        if "combine_op" not in kwargs:
            kwargs["combine_op"]="mult"
        super().init(kwargs)
        


#Input Pipeline (bugs?)
class PipelineLayer(SimpleLayer):
    type="Pipeline"
    def init(self,kwargs):
        super().init(kwargs)
        self.shuffle=kwargs.pop("shuffle",False)
        self.num_threads=kwargs.pop("num_threads",16)
        self.make_batch=kwargs.pop("make_batch",False)
        if self.make_batch:
            self.batch=kwargs.pop("batch",128)
        else:
            self.batch=self.y.get_shape()[0].value
        self.capacity=kwargs.pop("capacity",min(4096,self.batch*16))#Samples, not batches (weird when make_batch=False?)
    def call(self):
        super().call()
        data=[self.y]
        labels=self.get_labels()
        if labels!=None:
            data.append(labels)
        if self.shuffle:
            self.min_after_dequeue=kwargs.pop("min_after_dequeue",self.capacity//4)
            output=tf.train.shuffle_batch(data, batch_size=self.batch, num_threads=self.num_threads, capacity=self.capacity,
                                          enqueue_many=not self.make_batch,min_after_dequeue=self.min_after_dequeue)
        else:
            output=tf.train.batch(data, batch_size=self.batch, num_threads=self.num_threads, capacity=self.capacity, 
                                  enqueue_many=not self.make_batch)
        if labels!=None:
            self.y=output[0]
            self.labels=output[1]
        else:
            self.y=output
            

#Label Generators
class IdentityLabelLayer(SimpleLayer):
    type="Identity_Label"
    def call(self):
        super().call()
        self.labels=self.y
    
class ConstantLabelLayer(SimpleLayer):
    type="Constant_Label"
    def init(self,kwargs):
        super().init(kwargs)
        self.label=np.float32(kwargs.pop("label"))    #Must be numpy array
    def call(self):
        super().call()
        tile_n=[self.y.get_shape()[0].value]+[1]*self.label.ndim
        self.labels=tf.constant(np.tile(np.expand_dims(self.label,axis=0),tile_n))
    
class OneHotLabelLayer(ConstantLabelLayer):
    type="One_Hot_Label"
    def init(self,kwargs):
        self.i=kwargs.pop("i")
        self.n=kwargs.pop("n")
        label=[0]*n
        label[i]=1
        kwargs["label"]=np.float32(label)
        super().init(kwargs)

class LabelLayer(SimpleLayer): #A layer acting on the labels (make into sublayer type instead?)
    type="Label"
    def init(self,kwargs):
        super().init(kwargs)
        self.layer_def=kwargs.pop("layer")
    def call(self):
        super().call()
        self.label_layer=self.add_sublayer(Layer(x=self.get_labels(),**self.layer_def))
        self.labels=self.label_layer.get()
    def export(self):
        exp=super().export()
        self.label_layer._export(exp["layer"])
        return exp

#Batch Generators
class DataLayer(SimpleLayer):
    type="Abstract"
    def init(self,kwargs):
        super().init(kwargs)
        self.batch=kwargs.pop("batch",32)  #The number of elements in a batch

class BatchIdentityLayer(DataLayer): #Broadcasts a tensor into a batch of identical tensors
    type="Batch_Identity"
    def call(self):
        super().call()
        self.y=tf.tile(tf.expand_dims(self.y,axis=0), [self.batch]+[1]*self.y.get_shape().ndims)

class BatchSliceLayer(DataLayer): #Slices a tensor into batches without shuffling
    type="Batch_Slice"
    def init(self,kwargs):
        super().init(kwargs)
        self.thread_safe=kwargs.pop("thread_safe",False) #Allows running in multiple threads, but breaks self.last_y and related (untested)
    def call(self):
        super().call()
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
'''CentralCrop,BatchCentralCrop=make_batch_layer(name="Central_Crop",
    fun=lambda x,shape:tf.image.resize_image_with_crop_or_pad(x, target_height=shape[0], target_width=shape[1]),
    args=["shape"],shape=None)'''
VerticalRandomFlip,BatchVerticalRandomFlip=make_batch_layer(name="Random_Vertical_Flip",fun=tf.image.random_flip_up_down,drop_on_test=True)
HorizontalRandomFlip,BatchHorizontalRandomFlip=make_batch_layer(name="Random_Horizontal_Flip",
    fun=tf.image.random_flip_left_right,drop_on_test=True)
RandomBrightness,BatchRandomBrightness=make_batch_layer(name="Random_Brightness",
    fun=lambda x,max_delta:tf.image.random_brightness(x,max_delta=max_delta),
    args=["max_delta"],max_delta=63,drop_on_test=True)
RandomContrast,BatchRandomContrast=make_batch_layer(name="Random_Contrast",
    fun=lambda x,lower,upper:tf.image.random_contrast(x,lower=lower,upper=upper),
    args=["lower","upper"],lower=0.2, upper=1.8,drop_on_test=True)

class CentralCrop(SimpleLayer): #Broadcasts a tensor into a batch of identical tensors
    type="Central_Crop"
    def init(self,kwargs):
        super().init(kwargs)
        self.shape=kwargs.pop("shape")
    def call(self):
        super().call()
        y_shape=self.y.get_shape().as_list()
        if len(y_shape)==3:
            begin=[(y_shape[0]-self.shape[0])//2,(y_shape[1]-self.shape[1])//2,0]
            size=shape+[y_shape[2]]
        else:
            begin=[0,(y_shape[1]-self.shape[0])//2,(y_shape[2]-self.shape[1])//2,0]
            size=[y_shape[0]]+self.shape+[y_shape[3]]
        self.y=tf.slice(self.y, begin, size)

#Standard datasets
class MNISTLayer(DataLayer): 
    type="MNIST"
    def init(self,kwargs):
        super().init(kwargs)
        self.folder=kwargs.pop("folder","/tmp/tensorflow/mnist/input_data")
        self.data=mnist.read_data_sets(self.folder, one_hot=True)
    def call(self):
        super().call()
        if self.test:
            self.y, self.labels=self.data.test.images, self.data.test.labels
            self.y=tf.reshape(self.y, [10000,28,28,1])
        else:
            self.y, self.labels=tf.py_func(lambda :self.data.train.next_batch(self.batch), [], [tf.float32,tf.float64], stateful=True)
            self.y=tf.reshape(self.y, [self.batch,28,28,1])
        self.labels=tf.cast(self.labels,tf.float32)


class CIFARLayer(DataLayer):
    type="CIFAR_10"
    url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    files_train=["data_batch_%i"%i for i in range(1,6)]
    files_test=["test_batch"]
    n_train=50000
    n_test=10000
    n_classes=10
    sample_shape=[32,32,3]
    def init(self,kwargs):
        super().init(kwargs)
        self.folder=kwargs.pop("folder",'/tmp/cifar10_data')
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
    def call(self):
        super().call()
        self.data=self.add_sublayer(ConstantLayer(
                folder=self.file_data_path,file=self.file_base_name,label_file=self.file_base_name+".labels",
                convert_file=True,normalize=True))
        self.y=self.data.get()
        self.labels=self.data.labels
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
        
        
        














