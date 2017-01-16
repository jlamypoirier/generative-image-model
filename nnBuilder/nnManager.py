import tensorflow as tf
import numpy as np
import os
#from dataProducer import *


class _LayerFactory:
    classes={}
    updated=False
    @staticmethod
    def _add_class(cl):
        assert(isinstance(cl,type) and issubclass(cl,Layer))
        if "type" in dir(cl):
            t=cl.type
        else:
            t=cl.__name__
        if t!="Abstract":
            _LayerFactory.classes[t]=cl
    @staticmethod
    def _update_classes(cl):
        _LayerFactory._add_class(cl)
        for scl in type.__subclasses__(cl):
            _LayerFactory._update_classes(scl)
    @staticmethod
    def update_classes():
        _LayerFactory._update_classes(Layer)
    @staticmethod
    def build(type,*args,**kwargs):
        if not _LayerFactory.updated:
            _LayerFactory.update_classes()
        assert(type in _LayerFactory.classes)
        return(_LayerFactory.classes[type](*args,**kwargs))
LayerFactory=_LayerFactory.build

###Basic layer
class Layer:
    type="Identity"
    def __init__(self,x,batch_norm=False,dropout=False,**kwargs):
        for a,b in enumerate(kwargs):
            print("Unknown argument: "+a)
        self.type=self.__class__.type
        self._x=x
        self.x=self._x
        self.batch_norm=batch_norm
        self.dropout=dropout
        if self.batch_norm:
            mean,std=tf.nn.moments(x,range(len(self.x.get_shape().as_list())))
            self.x=(self.x-mean)/std
        if self.dropout:
            self.tf_keep_rate=tf.placeholder_with_default(np.float32(1.0),[])
            self.x=tf.nn.dropout(self.x,self.tf_keep_rate)
        self.y=self.x
    def save(self,**kwargs):
        kwargs["type"]=self.type
        kwargs["dropout"]=self.dropout
        kwargs["batch_norm"]=self.batch_norm
        return kwargs
    def get(self):
        return self.y
    def getVars(self):
        return self.getWeights()+self.getBiases()
    def getWeights(self):
        return []
    def getBiases(self):
        return []
    def getDropout(self):
        if self.dropout:
            return [self.tf_keep_rate]
        return []
    def getIn(self):
        return self.x
    def start(self,sess):
        return
    def stop(self):
        return

###Input and random layers
class BasicInputLayer(Layer):
    type="Abstract"
    def __init__(self,x=None,shape=None,**kwargs):
        Layer.__init__(self,None,**kwargs)
        self._shape=shape                    #Shape of the tensor, or None to keep arbitrary, or -1 to deduce from x
        self.shape=shape                     #Can set individual components to None or -1
        self.var_dim=self.shape==None
        if self.shape==None:
            pass
        elif self.shape<0:
            assert(x!=None)
            self.shape=tf.shape(x)
        else:
            shape=self.shape
            self.var_dim=self.var_dim or None in self.shape
            for i,s in enumerate(self.shape):
                if s!=None and s<0:
                    assert(x!=None and tf.shape(x).ndims==len(shape))
                    shape[i]=tf.shape(x)[i]
            self.shape=tf.concat(0,shape)
    def save(self,**kwargs):
        kwargs["shape"]=self._shape
        return Layer.save(self,**kwargs)


class PlaceholderLayer(BasicInputLayer):     
    #Warning: TensorFlow doesn't always like these:
    #    Doesn't work with derivatives even if no input is fed
    #    Whole shape is forgotten if any component is set to None
    type="Placeholder"
    def __init__(self,x,ignore_shape=False,**kwargs):
        BasicInputLayer.__init__(self,x,**kwargs)
        self.y=tf.placeholder_with_default(self.y,self.shape)

class InputLayer(BasicInputLayer):
    type="Input"
    def __init__(self,x=None,**kwargs):
        BasicInputLayer.__init__(self,x,**kwargs)
        self.y=tf.placeholder(dtype=tf.float32,shape=self.shape)
    
class RandomLayer(BasicInputLayer):
    type="Random"
    def __init__(self,x=None,rand_type="normal",scale=1.,mean=0.,**kwargs):
        BasicInputLayer.__init__(self,x,**kwargs)
        self.rand_type=rand_type             #normal or uniform generator
        self.scale=scale                     #scale for the distribution (std or half-range)
        self.mean=mean                       #mean for the distribution
        assert(not self.var_dim)             #Shape tensor must be known
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
        return Layer.save(self,**kwargs)
    
###Linear and related layers
class LinearLayer(Layer):
    type="Linear"
    def __init__(self,x,size=None,rand_scale=0.1,**kwargs):
        Layer.__init__(self,x,**kwargs)
        self.size=size                       #Number of output channels
        self.rand_scale=rand_scale           #Scale for random initialization of weights and biases
        assert(size!=None)
        shape=self.y.get_shape()[1:]
        if len(shape)>1:
            self.y=tf.reshape(self.y,[-1,shape.num_elements()])
        self.shape=[shape.num_elements(),size]
        self.w=tf.Variable(tf.random_normal(self.shape, 0, self.rand_scale),name='Weights')
        self.b=tf.Variable(tf.random_normal([size], self.rand_scale, self.rand_scale),name='Biases')
        self.y=tf.matmul(self.y,self.w)+self.b
    def save(self,**kwargs):
        kwargs["size"]=self.size
        kwargs["rand_scale"]=self.rand_scale
        return Layer.save(self,**kwargs)
    def getWeights(self):
        return [self.w]
    def getBiases(self):
        return [self.b]
            

class SimpleRelu(Layer):
    type="Simple_Relu"
    def __init__(self,x,**kwargs):
        Layer.__init__(self,x,dic)
        self.y=tf.nn.relu(self.y)
        

class ReluLayer(LinearLayer):
    type="Relu"
    def __init__(self,x,**kwargs):
        LinearLayer.__init__(self,x,**kwargs)
        self.relu=SimpleRelu(self.y)
        self.y=self.relu.y
        
class SimpleSoftmax(Layer):
    type="Simple_Softmax"
    def __init__(self,x,**kwargs):
        Layer.__init__(self,x,**kwargs)
        self.y=tf.nn.softmax(self.y)
        
class SoftmaxLayer(LinearLayer):
    type="Softmax"
    def __init__(self,x,**kwargs):
        LinearLayer.__init__(self,x,**kwargs)
        self.softmax=SimpleSoftmax(self.y)
        self.y=self.softmax.y
        
class SimpleSigmoid(Layer):
    type="Simple_Sigmoid"
    def __init__(self,x,**kwargs):
        Layer.__init__(self,x,**kwargs)
        self.y=tf.nn.sigmoid(self.y)
        
class SigmoidLayer(LinearLayer):
    type="Sigmoid"
    def __init__(self,x,**kwargs):
        LinearLayer.__init__(self,x,**kwargs)
        self.sigmoid=SimpleSigmoid(self.y)
        self.y=self.sigmoid.y
        
class QuadLayer(LinearLayer):
    type="Quadratic"
    def __init__(self,x,**kwargs):
        LinearLayer.__init__(self,x,**kwargs)
        self.y=tf.mul(self.y,self.y)

###Convolution and pooling
class WindowLayer(Layer): #Common elements of convolution and pooling (abstract class)
    type="Abstract"
    def __init__(self,x,pad="VALID",window=3,stride=1,**kwargs):
        Layer.__init__(self,x,**kwargs)
        self.pad=pad                         #Type of padding used
        self.window=window                   #Size of the input window
        self.stride=stride                   #Stride distance of the input window
    def save(self,**kwargs):
        kwargs["pad"]=self.pad
        kwargs["window"]=self.window
        kwargs["stride"]=self.stride
        return Layer.save(self,**kwargs)
        
class ConvLayer(WindowLayer):
    type="Convolution"
    def __init__(self,x,size=None,relu=True,input_channels=None,rand_scale=0.1,**kwargs):
        WindowLayer.__init__(self,x,**kwargs)
        self.size=size                       #Number of output channels
        self.relu=relu                       #Optional relu on the output
        self._input_channels=input_channels  #(Optional) overrides the number of input channels, needed for variable size input
        self.rand_scale=rand_scale           #Scale for random initialization of weights and biases
        
        assert(self.size!=None)
        self.input_channels=self._input_channels or self.y.get_shape()[3].value
        #if self.input_channels==None:
        #    self.input_channels=self.y.get_shape()[3].value
        self.filter_shape=[self.window,self.window,self.input_channels,self.size]
        self.w=tf.Variable(tf.random_normal(self.window, 0, self.rand_scale),name='Weights')
        self.b=tf.Variable(tf.random_normal([self.size], rand_scale, self.rand_scale),name='Biases')
        self.y=tf.nn.bias_add(tf.nn.conv2d(self.y,self.w,[1,self.stride,self.stride,1], self.pad),self.b)
        if self.relu:
            self.y=tf.nn.relu(self.y)
    def getWeights(self):
        return [self.w]
    def getBiases(self):
        return [self.b]
    def save(self,**kwargs):
        kwargs["size"]=self.size
        kwargs["relu"]=self.relu
        kwargs["input_channels"]=self._input_channels
        kwargs["rand_scale"]=self.rand_scale
        return WindowLayer.save(self,**kwargs)
    
class PoolLayer(WindowLayer):
    type="Pool"
    def __init__(self,x,pool_type="max",stride=2,**kwargs):
        WindowLayer.__init__(self,x,stride=stride**kwargs)
        self.pool_type=pool_type             #avg or max pooling 
        self.ksize=[1, self.poolWindow, self.poolWindow, 1]
        self.strides=[1, self.poolStride, self.poolStride, 1]
        if self.pool_type=="max":
            self.y=tf.nn.max_pool(self.y, ksize=self.ksize, 
                      strides=self.strides,padding=self.pad)
        elif self.pool_type=="avg":
            self.y=tf.nn.avg_pool(self.y, ksize=self.ksize, 
                      strides=self.strides,padding=self.pad)
        else: 
            assert(False)
    def save(self,**kwargs):
        kwargs["pool_type"]=self.pool_type
        return WindowLayer.save(self,**kwargs)


    

###Layer combining and composition
class Composite(Layer): #Common elements of composite layers (abstract class)
    type="Abstract"
    def __init__(self,x,layers=[],**kwargs):
        Layer.__init__(self,x,**kwargs)
        self._layers=layers                  #Definition of the layers
        self.layers=[]
    def save(self,**kwargs):
        kwargs["layers"]=[layer.save() for layer in self.layers]
        #kwargs["layers"]=self._layers
        return Layer.save(self,**kwargs)
    def start(self,sess):
        for layer in self.layers: layer.start(sess) 
    def stop(self):
        for layer in self.layers: layer.stop() 
    def getWeights(self):
        return Layer.getWeights(self)+[weight for weight in layer.getWeights() for layer in self.layers]
    def getBiases(self):
        return Layer.getBiases(self)+[bias for bias in layer.getBiases() for layer in self.layers]
    def getDropout(self):
        return Layer.getDropout(self)+[rate for rate in layer.getDropout() for layer in self.layers]
        #rates=Layer.getDropout(self)
        #for layer in self.layers: rates+=layer.getDropout()
        #return rates

class SimpleCombine:#Auxiliary class to Combine
    def __init__(self,x,combine_op="combine"):
        self.x=x
        shapes=[_x.get_shape().as_list() for _x in self.x]
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
            
class CombineLayer(Composite): #Combines layers in a parallel way
    type="Combine"
    def __init__(self,x,combine_op="combine",**kwargs):
        CompositeLayer.__init__(self,x,**kwargs)
        self.combine_op=combine_op           #Type of combining: "combine" (on channel dimension), "combine_batch", "sub", "mult"
        for _layer in _layers:
            self.layers.append(LayerFactory(self.y,_layer))
        self.combine=SimpleCombine([layer.get() for layer in self.layers],combine_op=self.combine_op)
        self.y=self.combine.y
    def save(self,**kwargs):
        kwargs["combine_op"]=self.combine_op
        return Composite.save(self,**kwargs)

class Network(Composite):
    type="Network"
    def __init__(self,x,**kwargs):
        layers=[]
        Composite.__init__(self,x,**kwargs)
        for _layer in _layers:
            self.layers.append(LayerFactory(self.y,_layer))
            self.y=self.layers[-1].get()
    


###Complex Layers
class RandomCombineLayer(CombineLayer):
    type="Random_Combined"
    def __init__(self,x,channels=1,**kwargs):
        layers=[{"Type":"Identity"},kwargs.update({"Type":"Random",shape=[-1 for i in range(x.get_shape().ndims-1)]+[channels]})]
        CombineLayer.__init__(self,x,layers=layers)
    def save(self,**kwargs):
        kwargs=self.layers[1].save(kwargs)
        kwargs["type"]=self.type
        return kwargs
    
    
'''def RandomLayer(x,dic):
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
    return layerFactory(x,new_dic)'''


        

'''    
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
        BasicNetwork.__init__(self,x,dic)'''
        
'''class NetworkSaver:
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
'''   
    