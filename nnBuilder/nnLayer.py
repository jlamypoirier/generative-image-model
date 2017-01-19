import numpy as np
import tensorflow as tf
import warnings
#import os
#from dataProducer import *






class _LayerFactory:
    classes={}
    updated=False
    _notAClassError="Argument is not a class: %s"
    _subclassError="Argument is not a subclass of Layer: %s"
    _layerTypeError="Unknown Layer type: %s"
    @staticmethod
    def _add_class(cl):
        assert(isinstance(cl,type)),self._notAClassError%cl
        assert(issubclass(cl,Layer)), self._subclassError%cl
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
        assert(type in _LayerFactory.classes), self._layerTypeError%type
        return(_LayerFactory.classes[type](*args,**kwargs))
LayerFactory=_LayerFactory.build

###Basic layer
class Layer:
    type="Identity"
    _uncaughtArgumentWarning="Unknown argument: %s"
    _clonedTypeError='Trying to clone layers of different type: "%s" vs "%s"'
    _copyManagerError="Copying variable requires a manager with an active session"
    _copySessionError="Copying variable requires both networks to have the same active session"
    _copyNetworkError="Copying variable requires identical networks"
    def __init__(self,x,batch_norm=False,dropout=False,_cloned_layer=None,**kwargs):
        for kw in kwargs:
            warnings.warn(self._uncaughtArgumentWarning%kw)
        self.type=self.__class__.type        #The type of the layer (user-friendly class name)
        self._x=x                            #The input for the layer
        self.x=self._x
        self.batch_norm=batch_norm           #Optional batch normalization on the input
        self.dropout=dropout                 #Optional dropout on the input
        if self.batch_norm:
            mean,std=tf.nn.moments(x,range(len(self.x.get_shape().as_list())))
            self.x=(self.x-mean)/std
        if self.dropout:
            self.tf_keep_rate=tf.placeholder_with_default(np.float32(1.0),[])
            self.x=tf.nn.dropout(self.x,self.tf_keep_rate)
        if _cloned_layer!=None:              #Used by the clone function, not for direct use
            assert(self.type==_cloned_layer.type), self._clonedTypeError%(self.type,_cloned_layer.type)
        self.y=self.x
    def save(self,**kwargs):
        kwargs["type"]=self.type
        kwargs["dropout"]=self.dropout
        kwargs["batch_norm"]=self.batch_norm
        return kwargs
    def _getLabels(self): #Need label producing layers
        return None
    def getLabels(self):
        return self._getLabels() or self.getInputLabels()
    def getInputLabels(self):
        return (isinstance(self.x,Layer) and x.getLabels()) or None
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
        self.sess=sess
    def stop(self):
        self.sess=None                        
    def copy(self,                            #Makes a copy of the graph, with a new variable set
             x=None,                          #Feed a new input x or use the existing one (x=None)
             manager=None,                     #Specify a session manager for the new layer (Optional)
             copy_vars=False,                 #Copy the variable values (requires a session manager with an active session)
             **kwargs                         #Override some other arguments (risky if copy_vars=True)
            ):    
        new_layer=LayerFactory(x=x or self.x,**self.save().update(kwargs))
        if manager!=None:
            manager.add([new_layer])
        if copy_vars:
            assert(manager!=None and manager.running), self._copyManagerError
            self.copy_vars_to(layer)
    def copy_vars_to(self,layer):#Copies the variables to another layer. The two layers must be identical
        assert(self.sess!=None and self.sess==layer.sess), self._copySessionError
        self.sess.run(self.copy_vars_op(layer))
    def copy_vars_op(self,layer):#Generates the copying operation but doesn't run it
        old_vars=self.getVars()
        new_vars=new_layer.getVars(),
        assert(len(old_vars)==len(new_vars)),self._copyNetworkError
        for i in len(old_vars): #Check if the variables are compatible
            assert(old_vars[i].get_shape().as_list()==new_vars[i].get_shape().as_list()), self._copyNetworkError
        return [new_var.assign(old_vars[i]) for i,new_var in enumerate(new_vars)]
    def clone(self,                           #Makes a copy of the graph, with a new variable set
             x=None,                          #Feed a new input x or use the existing one (x=None)
             manager=None,                    #Specify a session manager for the new layer (Optional)
             **kwargs                         #Override some other arguments (risky)
            ):    
        new_layer=LayerFactory(x=x or self.x,_cloned_layer=self,**self.save().update(kwargs))
        if manager!=None:
            manager.add([new_layer])
            

###Input and random layers
class BasicInputLayer(Layer):
    type="Abstract"
    _noInputError="Can't deduce the shape from the input: no input tensor given"
    _dimensionError="The input tensor and the given shape have different dimension: %s vs %s"
    def __init__(self,x=None,shape=None,**kwargs):
        Layer.__init__(self,None,**kwargs)
        self._shape=shape                    #Shape of the tensor, or None to keep arbitrary, or -1 to deduce from x
        self.shape=shape                     #Can set individual components to None or -1
        self.var_dim=self.shape==None
        if self.shape==None:
            pass
        elif self.shape<0:
            assert(x!=None), _noInputError
            self.shape=tf.shape(x)
        else:
            shape=self.shape
            self.var_dim=self.var_dim or None in self.shape
            for i,s in enumerate(self.shape):
                if s!=None and s<0:
                    assert(x!=None), _noInputError
                    assert(tf.shape(x).ndims==len(shape)), _dimensionError%(tf.shape(x).ndims,len(shape))
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
    _shapeError="Random layer must have a fixed shape"
    def __init__(self,x=None,rand_type="normal",scale=1.,mean=0.,**kwargs):
        BasicInputLayer.__init__(self,x,**kwargs)
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
        return Layer.save(self,**kwargs)
    
###Linear and related layers
class LinearLayer(Layer):
    type="Linear"
    def __init__(self,x,size=1,rand_scale=0.1,**kwargs):
        Layer.__init__(self,x,**kwargs)
        self.size=size                       #Number of output channels
        self.rand_scale=rand_scale           #Scale for random initialization of weights and biases
        shape=self.y.get_shape()[1:]
        if len(shape)>1:
            self.y=tf.reshape(self.y,[-1,shape.num_elements()])
        self.shape=[shape.num_elements(),size]
        if "_cloned_layer" in kwargs:
            self.w=kwargs["_cloned_layer"].w
            self.b=kwargs["_cloned_layer"].b
        else:
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
    def __init__(self,x,size=1,relu=True,input_channels=None,rand_scale=0.1,**kwargs):
        WindowLayer.__init__(self,x,**kwargs)
        self.size=size                       #Number of output channels
        self.relu=relu                       #Optional relu on the output
        self._input_channels=input_channels  #(Optional) overrides the number of input channels, needed for variable size input
        self.rand_scale=rand_scale           #Scale for random initialization of weights and biases
        self.input_channels=self._input_channels or self.y.get_shape()[3].value
        #if self.input_channels==None:
        #    self.input_channels=self.y.get_shape()[3].value
        self.filter_shape=[self.window,self.window,self.input_channels,self.size]
        if "_cloned_layer" in kwargs:
            self.w=kwargs["_cloned_layer"].w
            self.b=kwargs["_cloned_layer"].b
        else:
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
    _poolTypeError="Invlid pooling type: %s"
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
            raise Exception(self._poolTypeError%self.pool_type)
    def save(self,**kwargs):
        kwargs["pool_type"]=self.pool_type
        return WindowLayer.save(self,**kwargs)


    

###Layer combining and composition
class Composite(Layer): #Common elements of composite layers (abstract class)
    type="Abstract"
    _clonedSublayerError="Trying to clone networks with a different number of sublayers: %s vs %s"
    def __init__(self,x,layers=[],**kwargs):
        Layer.__init__(self,x,**kwargs)
        self._layers=layers                  #Definition of the layers
        if "_cloned_layer" in kwargs:
            cloned=kwargs["_cloned_layer"].layers
            assert(len(_layers)==len(cloned)), self._clonedSublayerError%(len(_layers),len(cloned))
            for i,_layer in enumerate(_layers):
                _layer["_cloned_layer"]=cloned[i]
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
    _combineTypeError="Invlid combine type: %s"
    _combineLengthError="Combine operation %s takes exactly 2 arguments (%s given)"
    _combineShapeError="Shapes %s and %s are incompatible for the combine operation %s"
    def __init__(self,x,combine_op="combine"):
        self.x=x
        shapes=[_x.get_shape().as_list() for _x in self.x]
        assert(self.compatible(shapes,combine_op)), self._combineShapeError%(shapes[0],shape,combine_op)
        if combine_op=="combine":
            self.y=tf.concat(len(shapes[0])-1,self.x)
        elif combine_op=="combine_batch":
            self.y=tf.concat(0,self.x)
        elif combine_op=="sub":
            assert(len(x)==2), self._combineLengthError%(combine_op,len(x))
            self.y=tf.sub(self.x[0],self.x[1])
        elif combine_op=="mult":
            if len(x)==2:
                self.y=tf.mul(self.x[0],self.x[1])
            else:
                self.y=tf.reduce_prod(tf.pack(self.x),[0])
        else:
            raise Exception(self._combineTypeError%combine_op)
    @staticmethod
    def compatible(shapes,combine_op):
        for shape in shapes[1:]:
            if len(shapes[0])!=len(shape): 
                return False
            if combine_op=="combine": 
                if shapes[0][:-1]!=shape[:-1]: 
                    return False
            elif combine_op=="combine_batch": 
                if shapes[0][1:]!=shape[1:]: 
                    return False
            else: 
                if shapes[0]!=shape: 
                    return False
        return True
        
            
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
        layers=[{"Type":"Identity"},kwargs.update({"Type":"Random","shape":[-1 for i in range(x.get_shape().ndims-1)]+[channels]})]
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
        
                  