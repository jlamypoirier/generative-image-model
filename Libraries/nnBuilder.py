from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
from dataProducer import *


'''def defaultinDict(dic,entry,default=None):
    return (entry in dic and dic[entry]) or default


def fixDir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    return dir_'''


#Basic layers
class Layer:
    def __init__(self,x,dic={}):
        self._type="None"
        self.x=x
        self.batch_norm=defaultinDict(dic,"batch_norm",False)#Batch normalization for the input
        self.dropout=defaultinDict(dic,"dropout",False)#Dropout for the input
        if self.batch_norm:
            mean,std=tf.nn.moments(x,range(len(self.x.get_shape().as_list())))
            self.x=(self.x-mean)/std
        if self.dropout:
            self.tf_keep_rate=tf.placeholder_with_default(np.float32(1.0),[])
            self.x=tf.nn.dropout(self.x,self.tf_keep_rate)
    def save(self):
        dic={}
        dic["type"]=self._type
        dic["dropout"]=self.dropout
        dic["batch_norm"]=self.batch_norm
        return dic
    def get(self):
        return self.y
    def getVars(self):
        return []
        #return self.w,self,b
    def getDropout(self):
        if self.dropout:
            return [self.tf_keep_rate]
        return []
    def getIn(self):
        return self.x
    def start(self,sess):
        #print("Starting "+self._type)
        return
    def stop(self):
        return
    
class IdentityLayer(Layer):
    def __init__(self,x,dic={}):
        Layer.__init__(self,x,dic)
        self._type="identity"
        self.y=self.x
        
class PlaceholderLayer(Layer):
    def __init__(self,x,ignore_batch=False,ignore_size=False,dic={}):
        Layer.__init__(self,x,dic)
        self.ignore_batch=defaultinDict(dic,"ignore_batch",ignore_batch)
        self.ignore_size=defaultinDict(dic,"ignore_size",ignore_batch)
        self._type="placeholder"
        self.t_shape=self.x.get_shape().as_list()
        if self.ignore_batch:
            self.t_shape[0]=None
        if self.ignore_size:
            self.t_shape[1:3]=[None,None]
        self.y=tf.placeholder_with_default(self.x,self.t_shape)
        
class LinearLayer(Layer):
    def __init__(self,x,size=None,dic={}):
        Layer.__init__(self,x,dic)
        self._type="linear"
        size=defaultinDict(dic,"size",size)
        self.randScale=defaultinDict(dic,"rand_scale",0.1)
        assert(size!=None)
        shape=self.x.get_shape()[1:]
        self.y=self.x
        if len(shape)>1:
            self.y=tf.reshape(self.y,[-1,shape.num_elements()])
        self.shape=[shape.num_elements(),size]
        self.w=tf.Variable(tf.random_normal(self.shape, 0, self.randScale),name='Weights')
        self.b=tf.Variable(tf.random_normal([size], self.randScale, self.randScale),name='Biases')
        self.y=tf.matmul(self.y,self.w)+self.b
    def save(self):
        dic=Layer.save(self)
        dic["size"]=self.shape[1]
        dic["rand_scale"]=self.randScale
        return dic
    def getVars(self):
        return [self.w,self.b]
            
class QuadLayer(LinearLayer):
    def __init__(self,x,size=None,dic={}):
        LinearLayer.__init__(self,x,size,dic)
        self._type="quadratic"
        self.y=tf.mul(self.y,self.y)

class BasicRelu(Layer):
    def __init__(self,x,dic={}):
        Layer.__init__(self,x,dic)
        self._type="basic_relu"
        self.y=tf.nn.relu(self.y)
        
class BatchNormalize(Layer):
    def __init__(self,x,dic={}):
        Layer.__init__(self,x,dic)
        self._type="batch_normalize"
        self.mean,self.std=tf.nn.moments(x,range(len(self.x.get_shape().as_list())))
        self.y=(self.x-self.mean)/self.std

class ReluLayer(LinearLayer):
    def __init__(self,x,size=None,dic={}):
        LinearLayer.__init__(self,x,size,dic)
        self._type="relu"
        self.relu=BasicRelu(self.x)
        self.y=self.relu.y
        
class BasicSoftmax(Layer):
    def __init__(self,x,dic={}):
        Layer.__init__(self,x,dic)
        self._type="basic_softmax"
        self.y=tf.nn.softmax(self.x)
        
class BasicSigmoid(Layer):
    def __init__(self,x,dic={}):
        Layer.__init__(self,x,dic)
        self._type="basic_sigmoid"
        self.y=tf.nn.sigmoid(self.x)
        
class SigmoidLayer(LinearLayer):
    def __init__(self,x,size=None,dic={}):
        LinearLayer.__init__(self,x,size,dic)
        self._type="sigmoid"
        self.sigmoid=BasicSigmoid(self.x)
        self.y=self.sigmoid.y
        
class SoftmaxLayer(LinearLayer):
    def __init__(self,x,size=None,dic={}):
        LinearLayer.__init__(self,x,size,dic)
        self._type="softmax"
        self.softmax=BasicSoftmax(self.x)
        self.y=self.softmax.y

class ConvLayer(Layer):
    def __init__(self,x,size=None,fSize=3,relu=True,dic={}):
        Layer.__init__(self,x,dic)
        self._type="convolution"
        self.size=defaultinDict(dic,"size",size)
        self.fsize=defaultinDict(dic,"fsize",fSize)
        self.stride=defaultinDict(dic,"stride",1)
        self.relu=defaultinDict(dic,"relu",relu)
        self.pad=defaultinDict(dic,"pad","VALID")
        self.input_channels=defaultinDict(dic,"input_channels",self.x.get_shape()[3].value)
        self.randScale=defaultinDict(dic,"rand_scale",0.1)
        assert(self.size!=None)
        self.filterShape=[self.fsize,self.fsize,self.input_channels,self.size]
        #print(self.filterShape)
        self.w=tf.Variable(tf.random_normal(self.filterShape, 0, self.randScale),name='Weights')
        self.b=tf.Variable(tf.random_normal([self.size], self.randScale, self.randScale),name='Biases')
        self.y=tf.nn.bias_add(tf.nn.conv2d(self.x,self.w,[1,self.stride,self.stride,1], self.pad),self.b)
        #print(self.y.get_shape())
        if self.relu:
            self.y=tf.nn.relu(self.y)
    def getVars(self):
        return [self.w,self.b]
    def save(self):
        dic=Layer.save(self)
        dic["size"]=self.size
        dic["fsize"]=self.fsize
        dic["relu"]=self.relu
        dic["pad"]=self.pad
        dic["stride"]=self.stride
        dic["rand_scale"]=self.randScale
        return dic
    
class PoolLayer(Layer):
    def __init__(self,x,poolStride=2,poolWindow=3,dic={}):
        Layer.__init__(self,x,dic)
        self._type="pool"
        self.poolStride=defaultinDict(dic,"stride",poolStride)
        self.poolWindow=defaultinDict(dic,"window",poolWindow)
        self.poolType=defaultinDict(dic,"pool_type","max")
        self.pad=defaultinDict(dic,"pad","VALID")
        shape=self.x.get_shape()[1:]
        if self.poolType=="max":
            self.y=tf.nn.max_pool(self.x, ksize=[1, self.poolWindow, self.poolWindow, 1], 
                      strides=[1, self.poolStride, self.poolStride, 1],padding=self.pad)
        elif self.poolType=="avg":
            self.y=tf.nn.avg_pool(self.x, ksize=[1, self.poolWindow, self.poolWindow, 1], 
                      strides=[1, self.poolStride, self.poolStride, 1],padding=self.pad)
        else: 
            assert(False)
    def save(self):
        dic=Layer.save(self)
        dic["stride"]=self.poolStride
        dic["window"]=self.poolWindow
        dic["pad"]=self.pad
        dic["pool_type"]=self.poolType
        return dic
    
class RescaleLayer(Layer):#Slow
    def __init__(self,x,scale=2,dic={}):
        Layer.__init__(self,x,dic)
        self._type="rescale"
        self.scale=defaultinDict(dic,"scale",scale)
        self.in_shape=np.array(x.get_shape().as_list()[1:3])
        self.out_shape=self.in_shape*self.scale
        self.y=tf.image.resize_bilinear(x, self.out_shape)
    def save(self):
        dic=Layer.save(self)
        dic["scale"]="scale"
        return dic
    
class ExpandLayer(Layer):#Fast but needs integer scale, no interpolation
    def __init__(self,x,scale=2,dic={}):
        Layer.__init__(self,x,dic)
        self._type="rescale"
        self.scale=defaultinDict(dic,"scale",scale)
        self.y=tf.depth_to_space(tf.tile(self.x,[1,1,1,self.scale**2]),self.scale)
    def save(self):
        dic=Layer.save(self)
        dic["scale"]="scale"
        return dic
    
class RandInput(Layer):
    def __init__(self,x=None,shape=None,dic={}):
        Layer.__init__(self,None,dic)
        self._type="random"
        self.shape=defaultinDict(dic,"shape",shape)
        self.channels=defaultinDict(dic,"channels",None)
        if self.shape=="x_shape":
            assert(x!=None)
            self.tf_shape=tf.shape(x)
            if self.channels!=None:
                self.tf_shape=tf.concat(0,[self.tf_shape[:-1],[self.channels]])
        else:
            self.tf_shape=self.shape
            if self.channels!=None:
                self.tf_shape[-1]=self.channels
        self.y=tf.random_normal(self.tf_shape, mean=0.0, 
                                         stddev=1.0,dtype=tf.float32)
        #print(self.y.get_shape(),self.channels)
    def save(self):
        dic=Layer.save(self)
        dic["shape"]=self.shape
        dic["channels"]=self.channels
        return dic
    
class BasicCombine(Layer):#Not a proper layer
    def __init__(self,x,combine_op="combine",dic={}):
        Layer.__init__(self,x,dic)
        self._type="combine"
        shapes=[_x.get_shape().as_list() for _x in x]
        #print(combine_op,shapes)
        """for shape in shapes:
            assert(len(shapes[0])==len(shape))
            if combine_op=="combine":
                assert(shapes[0][:-1]==shape[:-1])
            elif combine_op=="combine_batch":
                assert(shapes[0][1:]==shape[1:])
            else:
                assert(shapes[0]==shape)"""
        if combine_op=="combine":
            self.y=tf.concat(len(shapes[0])-1,self.x)
        elif combine_op=="combine_batch":
            self.y=tf.concat(0,self.x)
        elif combine_op=="sub":
            assert(len(x)==2)
            self.y=tf.sub(self.x[0],self.x[1])
        elif combine_op=="mult":
            if len(x)==2:
                self.y=tf.mul(self.x[0],self.x[1])
            else:
                self.y=tf.reduce_prod(tf.pack(self.x),[0])
        else:
            assert(False)
        
class BatchQueueLayer(Layer):#Simple interface to the BatchQueue class, comes with a placeholder
    def __init__(self,x=None,dic={}):
        Layer.__init__(self,x,dic)
        self._type="batchQueue"
        self.ignore_batch=defaultinDict(dic,"ignore_batch",True)
        self.ignore_size=defaultinDict(dic,"ignore_size",True)
        self.batchQueue=BatchQueue(dic=dic)
        self.y=self.batchQueue.batchData
        placeholder=PlaceholderLayer(self.y,ignore_batch=self.ignore_batch,ignore_size=self.ignore_size)
        self.y=placeholder.y
    def save(self):
        dic=Layer.save(self)
        self.batchQueue.save(dic)
        return dic
    def start(self,sess):
        self.batchQueue.start(sess)
    def stop(self):
        self.batchQueue.stop()

class BatchProducerLayer(Layer):
    def __init__(self,x=None,dic={}):
        Layer.__init__(self,x,dic)
        self._type="batchProducer"
        self.ignore_batch=defaultinDict(dic,"ignore_batch",True)
        self.ignore_size=defaultinDict(dic,"ignore_size",True)
        self.batchProducer=BatchProducer(dic=dic)
        self.y=self.batchProducer.batchData
        placeholder=PlaceholderLayer(self.y,ignore_batch=self.ignore_batch,ignore_size=self.ignore_size)
        self.y=placeholder.y
    def save(self):
        dic=Layer.save(self)
        self.batchProducer.save(dic)
        return dic
    def start(self,sess):
        self.batchProducer.start(sess)
    def stop(self):
        self.batchProducer.stop() 
        
'''class GenLayer(Layer):
    def __init__(self,x,n_rand=None,name=None,size=None,fSize=3,relu=True,dic={}):
        Layer.__init__(self,x,name,dic)
        self.size=defaultinDict(dic,"size",size)
        self.fSize=defaultinDict(dic,"fsize",fSize)
        self.n_rand=defaultinDict(dic,"n_rand",n_rand)
        self.relu=defaultinDict(dic,"relu",relu)
        self.pad=defaultinDict(dic,"pad","VALID")
        self.randScale=defaultinDict(dic,"rand_scale",0.1)
        self.scale=defaultinDict(dic,"scale",2)
        assert(self.size!=None and self.n_rand!=None)
        self.in_shape=np.array(x.get_shape().as_list()[1:3])
        self.out_shape=self.in_shape*2
        with tf.name_scope(self.name):
            if self.scale==1:
                self.out_shape=self.in_shape
                self.x_scaled=x
            else:          
                self.out_shape=self.in_shape*self.scale
                self.x_scaled=tf.image.resize_bilinear(x, self.out_shape)
            self.rand_layer=tf.random_normal(tf.concat(0,[self.x_scaled.get_shape()[:3],[self.n_rand]]), mean=0.0, 
                                         stddev=1.0,dtype=tf.float32, seed=None, name=None)
            
            
            self.ws=tf.Variable(tf.random_normal(
                    [self.fSize,self.fSize,self.x_scaled.get_shape()[3].value,self.n_rand], 0, self.randScale),name='ScaleWeights')
            self.bs=tf.Variable(tf.random_normal([self.n_rand], self.randScale, self.randScale),name='ScaleBiases')
            self.scale=tf.nn.bias_add(tf.nn.conv2d(self.x_scaled,self.ws,[1,1,1,1], "SAME"),self.bs)
            self.rand_scaled=tf.mul(self.rand_layer,self.scale)
            
            self.combined=tf.concat(3,[self.x_scaled,self.rand_scaled])
            self.filterShape=[self.fSize,self.fSize,self.combined.get_shape()[3].value,self.size]
            self.w=tf.Variable(tf.random_normal(self.filterShape, 0, self.randScale),name='Weights')
            self.b=tf.Variable(tf.random_normal([self.size], self.randScale, self.randScale),name='Biases')
            self.y=tf.nn.bias_add(tf.nn.conv2d(self.combined,self.w,[1,1,1,1], self.pad),self.b)
            if self.relu:
                self.y=tf.nn.relu(self.y)
    def save(self,dic={}):
        Layer.save(self,dic)
        dic["type"]="generator"
        dic["n_rand"]=self.n_rand
        dic["size"]=self.size
        dic["relu"]=self.relu
        return dic
    @staticmethod
    def load(x,dic):
        assert(dic["type"]=="generator")
        return GenLayer(x,dic=dic)'''

#Convenience Layers

    
def RandomLayer(x,dic):
    new_dic={"type":"combine_layer","layers":[]}
    new_dic["layers"].append({"type":"identity"})
    new_dic["layers"].append({"type":"random","shape":dic["shape"],"channels":dic["channels"]})
    return layerFactory(x,new_dic)

def RandomLayerScaled(x,dic):
    new_dic={"type":"combine_layer","layers":[]}
    new_dic["layers"].append({"type":"identity"})
    new_dic_2={"type":"combine_layer","layers":[],"combine_op":"mult"}
    new_dic_2["layers"].append({"type":"random","shape":dic["shape"],"channels":dic["channels"]})
    new_dic_2["layers"].append(dic["scale_layer"])
    new_dic["layers"].append(new_dic_2)
    return layerFactory(x,new_dic)

def MipmapLayer(x,dic):
    new_dic={"type":"pool","stride":dic["scale"],"window":dic["scale"],"pool_type":"avg","pad":"VALID"}
    return layerFactory(x,new_dic)





def layerFactory(x,dic):
    t=dic["type"]
    if t=="linear":
        return LinearLayer(x,dic=dic)
    elif t=="identity":
        return IdentityLayer(x,dic=dic)
    elif t=="quadratic":
        return QuadLayer(x,dic=dic)
    elif t=="basic_relu":
        return BasicRelu(x,dic=dic)
    elif t=="relu":
        return ReluLayer(x,dic=dic)
    elif t=="batch_normalize":
        return BatchNormalize(x,dic=dic)
    elif t=="basic_softmax":
        return BasicSoftmax(x,dic=dic)
    elif t=="softmax":
        return SoftmaxLayer(x,dic=dic)
    elif t=="basic_sigmoid":
        return BasicSigmoid(x,dic=dic)
    elif t=="sigmoid":
        return SigmoidLayer(x,dic=dic)
    elif t=="convolution":
        return ConvLayer(x,dic=dic)
    elif t=="pool":
        return PoolLayer(x,dic=dic)
    elif t=="mipmap":
        return MipmapLayer(x,dic=dic)
    elif t=="rescale":
        return RescaleLayer(x,dic=dic)
    elif t=="random":
        return RandInput(x,dic=dic)
    elif t=="random_layer":
        return RandomLayer(x,dic=dic)
    elif t=="random_layer_scaled":
        return RandomLayerScaled(x,dic=dic)
    elif t=="batchQueue":
        return BatchQueueLayer(x,dic=dic)
    elif t=="batchProducer":
        return BatchProducerLayer(x,dic=dic)
    elif t=="combine":
        assert(False)
    elif t=="combine_layer":
        return CombineLayer(x,dic=dic)
    elif t=="basic_network":
        return BasicNetwork(x,dic=dic)
    else:
        assert(False)
        
class CombineLayer(Layer):
    def __init__(self,x,dic={}):
        Layer.__init__(self,x,dic)
        self._type="combine_layer"
        self.combine_op=defaultinDict(dic,"combine_op","combine")
        self._vars=[]
        layers_def=dic["layers"]
        self.layers=[]
        for layer_def in layers_def:
            layer=layerFactory(self.x,layer_def)
            self.layers.append(layer)
            self._vars+=layer.getVars()
        self.combine=BasicCombine([layer.get() for layer in self.layers],combine_op=self.combine_op)
        self.y=self.combine.y
    def save(self):
        dic=Layer.save(self)
        dic["layers"]=[layer.save() for layer in self.layers]
        return dic
    def start(self,sess):
        for layer in self.layers: layer.start(sess) 
    def stop(self):
        for layer in self.layers: layer.stop() 
    def getVars(self):
        return self._vars
    def getDropout(self):
        rates=Layer.getDropout(self)
        for layer in self.layers: rates+=layer.getDropout()
        return rates
        
class BasicNetwork(Layer):
    def __init__(self,x,dic):
        Layer.__init__(self,x,dic)
        self._type="basic_network"
        self.layers=[]
        self._vars=[]
        self.y=self.x
        #self.dropout=defaultinDict(dic,"dropout",False)
        #self.tf_keep_rate=tf.placeholder_with_default(np.float32(1.0),[])
        layers_def=dic["layers"]
        for layer_def in layers_def:
            layer=layerFactory(self.y,layer_def)
            self.layers.append(layer)
            self._vars+=layer.getVars()
            self.y=layer.get()
    def save(self):
        dic=Layer.save(self)
        dic["layers"]=[layer.save() for layer in self.layers]
        return dic
    def start(self,sess):
        for layer in self.layers: layer.start(sess) 
    def stop(self):
        for layer in self.layers: layer.stop() 
    def getVars(self):
        return self._vars
    def getDropout(self):
        rates=Layer.getDropout(self)
        for layer in self.layers: rates+=layer.getDropout()
        return rates
    
class ConvNet(BasicNetwork):
    def __init__(self,x,dic):
        self.layer_n=dic["layer_n"]
        self.windows=dic["windows"]
        self.widths=dic["widths"]
        self.batch_norm=dic["batch_norm"]
        self.dropouts=defaultinDict(dic,"dropouts",[False]*self.layer_n)
        self.rand_scale=defaultinDict(dic,"rand_scale",.1)
        self.last_stride=defaultinDict(dic,"last_stride",1)
        self.last_relu=defaultinDict(dic,"last_relu",False)
        self.input_channels=defaultinDict(dic,"input_channels",None)
        dic["layers"]=[]
        for i in range(self.layer_n):
            dic["layers"].append({"type":"convolution","fsize":self.windows[i],"size":self.widths[i],"relu":True,
                                  "pad":"VALID","dropout":self.dropouts[i],"rand_scale":self.rand_scale,
                                  "batch_norm":self.batch_norm})
        if self.input_channels!=None:
            dic["layers"][0]["input_channels"]=self.input_channels
        dic["layers"][-1]["relu"]=self.last_relu
        dic["layers"][-1]["stride"]=self.last_stride
        BasicNetwork.__init__(self,x,dic)
        
class NetworkSaver:
    def __init__(self,network):#Takes a layer or network
        self.network=network
        self.saveDict={}
        for i,var in enumerate(network.getVars()):
            self.saveDict["var_%i"%i]=var
        self.saver = tf.train.Saver(self.saveDict,max_to_keep=10000)
        self.reset_op=tf.variables_initializer(network.getVars())
        self.sess=None
    def save(self,folder,safe=True):
        assert(self.sess!=None)
        File=fixDir(folder)+"/vars.ckpt"
        if safe:
            assert(not os.path.isfile(File))
        print(File)
        return self.saver.save(self.sess,File,write_meta_graph=False)
    def start(self,sess):
        self.sess=sess
    def stop(self):
        self.sess=None
    def load(self,folder):
        assert(self.sess!=None)
        return self.saver.restore(self.sess,folder+"/vars.ckpt")
    def init(self):
        self.sess.run(self.reset_op)
        
class SessManager:
    def __init__(self,networks):
        self.networks=networks #Doesn't need to be a network, just need start and stop functions
        self.coord = tf.train.Coordinator()
        self.running=False
    def add(self,networks):
        #assert(not self.running)
        self.networks+=networks
        if self.running:
            for network in networks:
                network.start(self.sess)
    def start(self):
        assert(not self.running)
        print("Starting new session")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        #config.gpu_options.per_process_gpu_memory_fraction=0.
        self.sess=tf.InteractiveSession(config=config)
        tf.global_variables_initializer().run()
        self.threads = tf.train.start_queue_runners(coord=self.coord, start=True)
        for network in self.networks:
            network.start(self.sess)
        self.running=True
        return self.sess
    def stop(self):
        assert(self.running)
        print("Ending session")
        for network in self.networks:
            network.stop()
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()
        self.running=False
    def maybe_end(self):
        if self.running:
            self.stop()
    def maybe_start(self):
        if not self.running:
            self.start()
    def get(self):
        self.maybe_start()
        return self.sess
    def clean(self):#Destroy everything
        global resize
        self.stop()
        tf.reset_default_graph()
        resize = tffunc(np.float32, np.int32)(resize_)
        gc.collect()
        
class NetworkTrainer:
    def __init__(self,network,loss,dic={}):
        self.network=network
        self.loss=loss
        self.learnRate=defaultinDict(dic,"learnRate",0.01)
        self.l2regMul=defaultinDict(dic,"l2regMul",1e-8)
        self.momentum=defaultinDict(dic,"momentum",.9)
        self.tf_dropout_rates=network.getDropout()
        self.keep_rate=defaultinDict(dic,"keep_rate",1.0)
        self.tf_l2regMul=tf.placeholder(tf.float32)
        self.tf_learnRate=tf.placeholder(tf.float32)
        self.tf_momentum=tf.placeholder(tf.float32)
        self.var_list=network.getVars()
        self.l2reg=tf.mul(tf.reduce_sum(tf.pack([tf.nn.l2_loss(var) for var in self.var_list])),self.tf_l2regMul)
        self.loss_full=tf.add(self.loss,self.l2reg)
        self.train_step = tf.train.MomentumOptimizer(self.tf_learnRate,self.tf_momentum).minimize(
            self.loss_full,var_list=self.var_list)
        self.sess=None
    def start(self,sess):
        self.sess=sess
    def stop(self):
        self.sess=None
    def train(self,n,dic={}):
        assert(self.sess!=None)
        feed_dict={self.tf_learnRate:self.learnRate,self.tf_l2regMul:self.l2regMul,
                                                self.tf_momentum:self.momentum}
        for rate in self.tf_dropout_rates:
            feed_dict[rate]=self.keep_rate
        feed_dict.update(dic)
        #print(feed_dict)
        for i in range(n):
            self.sess.run(self.train_step,feed_dict=feed_dict)
    def evaluate(self,n=10,print_=True):
        assert(self.sess!=None)
        loss=np.mean([self.sess.run(self.loss) for k in range(n)])
        if print_:
            print(loss)
        return (loss)
    

class ClassifierTrainer(NetworkTrainer):
    def __init__(self,network,labels,logits=None,dic={},extra_loss=None,sigmoid=False):
        self.labels=labels
        self.logits=logits
        if logits==None: self.logits=network.y
        if logits.get_shape()[-1].value in [1,None]:#Need fix
            self.cross_entropy=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits,self.labels))
        else:
            self.cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits,self.labels))
        loss=self.cross_entropy
        if extra_loss!=None:
            loss+=extra_loss
        NetworkTrainer.__init__(self,network,loss,dic)
        if sigmoid or logits.get_shape()[-1].value==1:
            print("sigmoid")
            self.sigmoidLayer=BasicSigmoid(self.logits)
            self.y=self.sigmoidLayer.get()
            self.wrong_prediction = tf.not_equal(tf.greater(self.y, 0.5), tf.greater(self.labels, 0.5))
        else:
            print("softmax")
            self.softmaxLayer=BasicSoftmax(self.logits)
            self.y=self.softmaxLayer.get()
            self.wrong_prediction = tf.not_equal(tf.argmax(self.y, 1), tf.argmax(self.labels, 1))
        self.error_rate = tf.reduce_mean(tf.cast(self.wrong_prediction, tf.float32))
        #self.error_rate=tf.sub(1.,self.accuracy)
    def evaluate(self,n=10,print_=True):
        assert(self.sess!=None)
        e=np.array([self.sess.run([self.cross_entropy,self.error_rate]) for k in range(n)])
        loss=np.mean(e[:,0])
        error=np.mean(e[:,1])
        if print_:
            print((loss,error))
        return (loss,error)
    
class ClassifierArrayTrainer(ClassifierTrainer):
    def __init__(self,network,labels,logits=None,dic={},extra_loss=None,sigmoid=False):
        if logits==None: logits=network.y
        #print(logits.get_shape().as_list())
        size=tf.shape(logits)
        labels=tf.tile(tf.expand_dims(tf.expand_dims(labels,1),1),tf.concat(0,[[1],size[1:3],[1]]))
        ClassifierTrainer.__init__(self,network,tf.reshape(labels,[-1,size[3]]),
                                   tf.reshape(logits,[-1,size[3]]),dic,extra_loss,sigmoid)
        
    
    
    
    
'''class Network:
    def __init__(self,x,layers=None,finish=True,softmax=True,dic={}):
        self.ready=False
        self.layers=[]
        self.w=[]
        self.b=[]
        self.x=x
        self.y=self.x
        self.running=False
        self.dropout=defaultinDict(dic,"dropout",False)
        self.tf_keep_rate=tf.placeholder_with_default(np.float32(1.0),[])
        layers=defaultinDict(dic,"layers",layers)
        if layers!=None: 
            self._load(layers,finish=finish,softmax=softmax)
        #self.shape=x.get_shape()
    def _addLayer(self,layer):
        assert(not self.ready)
        assert(layer.getIn()==self.y)
        self.layers.append(layer)
        self.y=layer.get()
        self.w.append(layer.w)
        self.b.append(layer.b)
    def addLinearLayer(self,size):
        layer=LinearLayer(self.y,"Layer%iLinear"%len(self.layers),size)
        self._addLayer(layer)
    def addReluLayer(self,size):
        layer=ReluLayer(self.y,"Layer%iRelu"%len(self.layers),size)
        self._addLayer(layer)
    def addConvLayer(self,fSize,size,convStride=1,poolStride=2,poolWindow=2):
        layer=ConvLayer(self.y,"Layer%iConv"%len(self.layers),fSize=fSize,size=size,
                        convStride=convStride,poolStride=poolStride,poolWindow=poolWindow)
        self._addLayer(layer)
    def finish(self,softmax=True):
        assert(not self.ready)
        self.ready=True
        self.logits=self.y
        if softmax:
            self.y=tf.nn.softmax(self.logits)
        self.saveDict={}
        for i in range(len(self.layers)):
            self.saveDict["Layer%iWeights"%i]=self.w[i]
            self.saveDict["Layer%iBiases"%i]=self.b[i]
        self.saver = tf.train.Saver(self.saveDict,max_to_keep=10000)
        self.reset_op=tf.initialize_variables(self.w+self.b)
    def startSess(self):
        assert(self.ready)
        assert(not self.running)
        print("Starting new session")
        self.sess=tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.running=True
        return self.sess
    def endSess(self):
        assert(self.running)
        print("Ending session")
        self.sess.close()
        self.running=False
    def ensureSess(self):
        if not self.running:
            self.startSess()
    def ensureNoSess(self):
        if self.running:
            self.endSess()
    #def resetSess(self):
    #    if self.running:
    #        print("Resetting variables")
    #        tf.initialize_all_variables().run()
    #    else:
     #       self.startSess()
    def saveVars(self,folder,safe=True):
        File=fixDir(folder)+"/vars.ckpt"
        if safe:
            assert(not os.path.isfile(File))
        print(File)
        return self.saver.save(self.sess,File,write_meta_graph=False)
    def loadVars(self,folder):
        return self.saver.restore(self.sess,folder+"/vars.ckpt")
    def reset_vars(self):
        self.sess.run(self.reset_op)
    def save(self,dic={}):
        dic["layers"]=[layer.save() for layer in self.layers]
        return dic
    def _load(self,layers,finish=True,softmax=True):
        for layer in layers:
            if layer["type"]=="linear":
                self._addLayer(LinearLayer.load(self.y,layer))
            elif layer["type"]=="relu":
                self._addLayer(ReluLayer.load(self.y,layer))
            elif layer["type"]=="conv":
                self._addLayer(ConvLayer.load(self.y,layer))
            elif layer["type"]=="quadratic":
                self._addLayer(QuadLayer.load(self.y,layer))
            elif layer["type"]=="random":
                self._addLayer(RandLayer.load(self.y,layer))
            elif layer["type"]=="generator":
                l=GenLayer.load(self.y,layer)
                self._addLayer(l)
                self.w.append(l.ws)
                self.b.append(l.bs)
            elif layer["type"]=="simple_conv":
                self._addLayer(SimpleConvLayer.load(self.y,layer))
            else:
                assert(False)
            if defaultinDict(layer,"dropout",False):
                self.y=tf.nn.dropout(self.y,self.tf_keep_rate)
        if finish: self.finish(softmax=softmax)
    @staticmethod
    def load(x,dic,finish=True,softmax=True):
        return Network(x,dic=dic,finish=finish,softmax=softmax)

class TrainingNetwork(Network):
    def __init__(self,batchQueue=None,layers=None,finish=True,l2regMul=5e-8,step=0,momentum=.90,learnRate=0.01,dic={},prob_score=False):
        if batchQueue==None:
            batchQueue=BatchQueue.load(dic,finish=True,prob_score=prob_score)
        layers=defaultinDict(dic,"layers",layers)
        self.prob_score=prob_score
        #assert(layers[-1]["type"]=="linear")
        assert(layers[-1]["size"]==batchQueue.numClasses)
        self.batchQueue=batchQueue
        #self.layersDef=layers
        self.step=defaultinDict(dic,"step",step)
        self.learnRate=defaultinDict(dic,"learnRate",learnRate)
        self.l2regMul=defaultinDict(dic,"l2regMul",l2regMul)
        self.momentum=defaultinDict(dic,"momentum",momentum)
        self.numClasses=batchQueue.numClasses
        self.size=batchQueue.size
        self.y_=tf.placeholder_with_default(batchQueue.get()[1],[None,self.numClasses],name='labels')#batch_labels
        x1=self,batchQueue.get()[0]
        self.x_=tf.placeholder_with_default(x1,[None,self.size,self.size,1],name='input')#batch_data
        Network.__init__(self.x_,layers,finish,softmax=True,dic=dic)
        self.x=x1
    def finish(self,softmax=True):
        Network.finish(self,softmax=not self.prob_score)
        #Loss and error rate
        self.tf_l2regMul=tf.placeholder(tf.float32)
        self.tf_learnRate=tf.placeholder(tf.float32)
        self.tf_momentum=tf.placeholder(tf.float32)
        self.l2reg=tf.mul(tf.add(tf.reduce_sum(tf.pack([tf.nn.l2_loss(w) for w in self.w])),
                          tf.reduce_sum(tf.pack([tf.nn.l2_loss(b) for b in self.b]))),self.tf_l2regMul)
        if self.prob_score:
            self.prob_loss=tf.reduce_mean(tf.mul(self.logits,self.y_))
            self.loss=tf.add(self.prob_loss,self.l2reg)
        else:
            self.cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits,self.y_))
            self.loss=tf.add(self.cross_entropy,self.l2reg)
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.error_rate=tf.sub(1.,self.accuracy)
        #Training step
        self.train_step = tf.train.MomentumOptimizer(self.tf_learnRate,self.tf_momentum).minimize(self.loss)
    """def train(self,n,newLearnRate=None):
        assert(self.ready)
        if newLearnRate:
            self.learnRate=newLearnRate
        for i in range(n):
            self.sess.run(self.train_step,feed_dict={self.tf_learnRate:self.learnRate})
        self.step+=n"""
    def train(self,n,newLearnRate=None,printN=None,evalN=10,saveN=None,saveDir=None,safe=True):
        assert(self.ready)
        self.ensureSess()
        if saveN!=None and safe:
            safeDir(saveDir)
        if newLearnRate:
            self.learnRate=newLearnRate
        for i in range(n):
            self.sess.run(self.train_step,feed_dict={self.tf_learnRate:self.learnRate,
                                                     self.tf_l2regMul:self.l2regMul,
                                                     self.tf_momentum:self.momentum})
            if printN!=None and (i+1)%printN==0:
                res=self.evaluate(n=evalN,print_=False)
                print("After %i steps of %i: %f, %f"%((i+1),n,res[0],res[1]))
            if saveN!=None and (i+1)%saveN==0:
                assert(saveDir!=None)
                print(self.saveVars(saveDir,False))
        self.step+=n
    """def trainMany(self,n,m,saveDir="",saveBaseName="model",newLearnRate=None,printN=None,evalN=10,saveN=None,safe=True):
        fixDir(saveDir)
        self.ensureSess()
        for i in range(m):
            tf.initialize_all_variables().run()
            self.batchQueue.assign(self.sess)
            fullDir=saveDir+"/"+saveBaseName+"_%i"%i
            if safe:
                safeDir(fullDir)
            self.train(n,newLearnRate=newLearnRate,printN=printN,evalN=evalN,saveN=saveN,saveDir=fullDir,safe=False)
            print(self.saveVars(fullDir,safe=False))
            self.saveAll(folder=fullDir)"""
    def evaluate(self,n=10,print_=True):
        assert(self.ready)
        e=np.array([self.sess.run([self.loss,self.error_rate],{self.tf_l2regMul:self.l2regMul}) for k in range(n)])
        loss=np.mean(e[:,0])
        error=np.mean(e[:,1])
        if print_:
            print((loss,error))
        return (loss,error)
    def confusionMatrix1(self,n=10,print_=True):
        a=np.array([self.sess.run([self.y_,self.y]) for i in range(n)]).argmax(3)
        c=np.zeros([self.numClasses]*2)
        [[d[:,i] for i in range(d.shape[1])]for d in b]
        for d in b:
            for i in range(d.shape[1]):
                c[d[0,i],d[1,i]]+=1
        return c
    def confusionMatrix(self,n=10,print_=True):
        a=np.concatenate([np.array(self.sess.run([self.y_,self.y])).argmax(2).transpose() for i in range(n)])
        b=np.zeros([self.numClasses]*2)
        for c in a:
            b[c[0],c[1]]+=1
        return b
    def startSess(self):
        Network.startSess(self)
        self.batchQueue.start(self.sess)
    def endSess(self):
        self.batchQueue.stop()
        Network.endSess(self)
    def save(self,dic={},folder=None):
        self.batchQueue.save(dic)
        Network.save(self.dic)
        dic["l2regMul"]=self.l2regMul
        dic["step"]=self.step
        dic["learnRate"]=self.learnRate
        if folder!=None:
            np.save(folder+"/graph.npy",dic)
        return dic
    @staticmethod
    def load(dic,finish=True):
        return TrainingNetwork(dic=dic)
    def checkInputProportions(self,n=10):
        a=[self.sess.run(self.y_) for i in range(n)]
        c=[[(b.argmax(1)==i).sum() for i in range(self.numClasses)] for b in a]
        return np.sum(c,0)'''
    
    
    
    
    
    
    

    
'''class TrainingNetwork:
    def __init__(self,batchQueue,networkDef):
        assert(networkDef[-1]["type"]=="linear" and networkDef[-1]["size"]==batchQueue.numClasses)
        self.batchQueue=batchQueue
        self.networkDef=networkDef
        self.numClasses=batchQueue.numClasses
        self.size=batchQueue.size
        self.x_=tf.placeholder_with_default(batchQueue.get()[0],[None,self.size,self.size,1],name='input')#batch_data
        self.y_=tf.placeholder_with_default(batchQueue.get()[1],[None,self.numClasses],name='labels')#batch_labels
        self.network=Network(self.x_,networkDef,finish=True,softmax=True)
        self.trainer=NetworkTrainer(self.network,self.y_)
    def startSess(self):
        self.coord = tf.train.Coordinator()
        self.sess=tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.threads = tf.train.start_queue_runners(coord=self.coord)
        self.trainer.setSess(self.sess)
    def endSess(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()'''
    
    
    
    
'''class NetworkTrainer:
    def __init__(self,network,labels,l2regMul=5e-8,step=0,learnRate=0.01):
        self.step=step
        self.network=network
        self.labels=labels
        #Loss and error rate
        self.cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network.logits,labels))
        self.l2regMul=l2regMul
        self.l2reg=tf.mul(tf.add(tf.reduce_sum(tf.pack([tf.nn.l2_loss(w) for w in network.w])),
                          tf.reduce_sum(tf.pack([tf.nn.l2_loss(b) for b in network.b]))),l2regMul)
        self.loss=tf.add(self.cross_entropy,self.l2reg)
        self.correct_prediction = tf.equal(tf.argmax(network.y, 1), tf.argmax(labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.error_rate=tf.sub(1.,self.accuracy)
        #Training step
        self.learnRate=learnRate
        self.tf_learnRate=tf.placeholder(tf.float32)
        self.train_step = tf.train.MomentumOptimizer(self.tf_learnRate,0.90).minimize(self.loss)
    def setSess(self,sess):
        self.sess=sess
    def train(self,n,newLearnRate=None):
        if newLearnRate:
            self.learnRate=newLearnRate
        for i in range(n):
            self.sess.run(self.train_step,feed_dict={self.tf_learnRate:self.learnRate})
    def evaluate(self,n=10,print_=True):
        e=np.array([self.sess.run([self.loss,self.error_rate]) for k in range(n)])
        loss=np.mean(e[:,0])
        error=np.mean(e[:,1])
        if print_:
            print((loss,error))
        return (loss,error)'''

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    