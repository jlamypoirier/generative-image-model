import numpy as np
import tensorflow as tf
import warnings
from functools import partial
#import os
#from dataProducer import *



#####To do list:

###Fixes
#Update nnUtils
#Rework _LayerVars
#Rework combine ops
#Update and add complex layers
#Finish CIFAR-10 example
#Ignore useless parameters on saving
#More tests

###Features:
#Input pipeline
#Implement Testing Phase
#Better input and batching
#Input preprocessing
#Implement "Soft Dropout"
#Reimplement DeepDream (update nnDreamer)
#Reimplement GAN (update nnGenerator)
#More layers

###Long term:
#Load large datasets
#ImageNet?
#Multi-GPU?


class LayerFactory:
    classes={}
    updated=False
    _notAClassError="Argument is not a class: %s"
    _subclassError="Argument is not a subclass of Layer: %s"
    _layerTypeError="Unknown Layer type: %s"
    @staticmethod
    def _add_class(cl):
        assert(isinstance(cl,type)),self._notAClassError%cl
        assert(issubclass(cl,SimpleLayer)), self._subclassError%cl
        if "type" in dir(cl):
            t=cl.type
        else:
            t=cl.__name__
        if t!="Abstract":
            LayerFactory.classes[t]=cl
    @staticmethod
    def _update_classes(cl):
        LayerFactory._add_class(cl)
        for scl in type.__subclasses__(cl):
            LayerFactory._update_classes(scl)
    @staticmethod
    def update_classes():
        LayerFactory._update_classes(SimpleLayer)
        LayerFactory.updated=True
    @staticmethod
    def build(type,*args,**kwargs):
        if not LayerFactory.updated:
            LayerFactory.update_classes()
        assert(type in LayerFactory.classes), LayerFactory._layerTypeError%type
        return(LayerFactory.classes[type](*args,**kwargs))
Layer=LayerFactory.build


#Functionality of SimpleLayer Spread into multiple classes
class _LayerRaw:#Most basic layer, has a type, an input and an output
    type="Abstract"
    _uncaughtArgumentWarning="Unknown argument: %s"
    _finishedError="Trying to set the output of a finished layer"
    _extra_args=[]
    def __init__(self,x=None,**kwargs):
        for kw in kwargs:
            if kw not in self._extra_args:
                warnings.warn(self._uncaughtArgumentWarning%kw)
        self.type=self.__class__.type        #The type of the layer (user-friendly class name)
        self._x=x                            #The input for the layer (can be a Layer, a Tensor, or a numpy array)
        self.x=self._x                       #The input, converted to tensor-like (can be None)
        self.y=self.x                        #The output of the layer during construction (final output obtained through get)
        if isinstance(self.y,_LayerRaw):
            self.y=self.y.get()
            self._x=self._x.get()
        self.finished=False
    def finish(self):
        if not self.finished:
            assert("_y" not in dir(self))    #Debug
            self._y=self.y                   #The final output
            self.finished=True
            del self.y
    def get_in(self):
        return self.x
    def get(self):                           #Garanteed to be the final output
        self.finish()
        return self._y
    def set(self,y):                         #Updates the output
        assert(not self.finished), self._finishedError
        self.y=y
    def save(self,**kwargs):
        kwargs["type"]=self.type
        return kwargs


class _LayerInstance(_LayerRaw):#Session management (use as base class for saver and trainer?)
    type="Abstract"
    _sessionError="No active session"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.sess=None
    def start(self,sess):
        self.finish()
        self.sess=sess       #Should be a SessManager
    def stop(self):
        self.sess=None
    def run(self):
        assert(self.sess!=None),self._sessionError
        return self.sess.run(self.y)
    
class _LayerCopy(_LayerInstance):#Copying and cloning
    type="Abstract"
    _copyTypeError='Trying to share variables between layers of different type: "%s" vs "%s"'
    _copyManagerError="Copying variable requires a manager with an active session"
    _copySessionError="Copying variable requires both networks to have the same active session"
    _copyNetworkError="Copying variable requires identical networks"
    def __init__(self,_cloned=None,**kwargs):
        super().__init__(**kwargs)
        self.cloned=_cloned         #The copied layer, if variables are shared (should be set through the copy function only)
        if self.cloned!=None:
            assert(self.type==self.cloned["type"]), self._clonedTypeError%(self.type,self.cloned.type)
    def copy(self,                            #Makes a copy of the graph, with or without variable sharing
             x=None,                          #Feed a new input x or use the existing one (x=None)
             sess=None,                       #Specify a session manager for the new layer, uses the copied layer's one if None
             share_vars=False,                #Enables variable sharing
             copy_vars=False,                 #Copy the variable values (requires a session manager with an active session)
             **kwargs                         #Override some other arguments (risky if copy_vars=True)
            ):
        if x==None:
            x=self.x
        cloned=share_vars and self or None
        layer=LayerFactory(x=x,**self.save().update(kwargs),_cloned=cloned)
        if sess!=None:
            sess.add([layer])
        elif self.sess!=None:
            self.sess.add([layer])
        if copy_vars and not share_vars:
            assert(layer.sess!=None and layer.sess.running), self._copyManagerError
            self.copy_vars_to(layer)
    def copy_vars_to(self,layer):#Copies the variables to another layer. The two layers must be identical
        assert(self.sess!=None and self.sess==layer.sess), self._copySessionError
        self.sess.run(self.copy_vars_op(layer))
    def copy_vars_op(self,layer):#Generates the copying operation but doesn't run it
        old_vars=self.getVars()  #Implemented in _LayerVars
        new_vars=layer.getVars(),
        assert(len(old_vars)==len(new_vars)),self._copyNetworkError
        for i in len(old_vars): #Check if the variables are compatible
            assert(old_vars[i].get_shape().as_list()==new_vars[i].get_shape().as_list()), self._copyNetworkError
        return [new_var.assign(old_vars[i]) for i,new_var in enumerate(new_vars)]

    
class _LayerTree(_LayerCopy):#Feature layers and sublayers, allows new sublayer types in derived classes
    type="Abstract"
    #A set of layers modifying the input / output
    SUBLAYER_INPUT=dict(kw="in_features",lst="sublayers_in",set_y=True,arg=True)
    SUBLAYER_OUTPUT=dict(kw="out_features",lst="sublayers_out",set_y=True,arg=True)#A set of layers modifying the output.
    #The layers of the network, if applicable
    SUBLAYER_PROPER=dict(kw="layers",lst="sublayers_proper",set_y=True,arg=True)
    #Layers added by derived classes, should be added before calling __init__
    SUBLAYER_INPUT_MANAGED=dict(kw="in_features_managed",lst="sublayers_in_managed",set_y=True,arg=False,managed=True,reverse=False)
    SUBLAYER_OUTPUT_MANAGED=dict(kw="out_features_managed",lst="sublayers_out_managed",set_y=True,arg=False,managed=True,reverse=True)
    SUBLAYER_PROPER_MANAGED=dict(kw="layers_managed",lst="sublayers_proper_managed",set_y=True,arg=False,managed=True,reverse=False)
    sublayer_types=[SUBLAYER_INPUT,SUBLAYER_OUTPUT,SUBLAYER_INPUT_MANAGED,SUBLAYER_OUTPUT_MANAGED,SUBLAYER_PROPER,SUBLAYER_PROPER_MANAGED]
    _extra_args=_LayerCopy._extra_args+[type["kw"] for type in sublayer_types if "arg" in type and type["arg"]]
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.sublayers=[]                        #All the sublayers of the layer
        for type in self.sublayer_types:         #The sublayers sorted by type: def at getattr(self,type["kw"]
            if "arg" in type and type["arg"] and type["kw"] in kwargs and kwargs[type["kw"]]!=None:
                setattr(self,type["kw"],kwargs[type["kw"]])
            elif type["kw"] not in dir(self):
                setattr(self,type["kw"],[])
            setattr(self,type["lst"],[])         #Instances at getattr(self,type["lst"]
        self.make_sublayers(**self.SUBLAYER_INPUT)
        self.make_sublayers(**self.SUBLAYER_INPUT_MANAGED)
        self.make_sublayers(**self.SUBLAYER_PROPER)
        self.make_sublayers(**self.SUBLAYER_PROPER_MANAGED)
    def save(self,**kwargs):
        for type in self.sublayer_types:
            if not ("managed" in type and type["managed"]):
                sublayers=getattr(self,type["lst"])
                if len(sublayers)>0:
                    kwargs[type["kw"]]=[layer.save() for layer in sublayers]
        return super().save(**kwargs)
    def make_sublayers(self,kw,lst,set_y,**kwargs):
        desc=getattr(self,kw)
        if self.cloned:
            cloned=getattr(self.cloned,getattr(self,lst))
            assert(len(desc)<=len(cloned)) #Allow removing some features, but only the last ones
        else:
            cloned=[None for _ in desc]
        for i,_desc in enumerate(desc):#Features implemented in the order they are given
            if type(_desc)==dict:
                layer=Layer(x=self.y,_cloned=cloned[i],**_desc)
            else:
                layer=Layer(x=self.y,_cloned=cloned[i],type=_desc)
            self.add_sublayer(layer,lst=lst,set_y=set_y)
    def add_sublayer_def(self,sublayer_type,inner=True,**kwargs):
        assert("x" not in dir(self)), "Sublayer definitions must be added before initializing the layer"
        assert("managed" in sublayer_type and sublayer_type["managed"]), "Adding a definition for a wrong type of layer"
        if sublayer_type["kw"] not in dir(self):
            setattr(self,sublayer_type["kw"],[])
        reverse="reverse" in sublayer_type and sublayer_type["reverse"] or False
        if inner==reverse:
            getattr(self,sublayer_type["kw"]).insert(0,kwargs)
        else:
            getattr(self,sublayer_type["kw"]).append(kwargs)
    def add_sublayer(self,layer,lst=None,set_y=False,finish=True,**kwargs):
        self.sublayers.append(layer)
        if finish or set_y:
            layer.finish()
        if lst!=None and lst in dir(self):
            getattr(self,lst).append(layer)
        if set_y:
            self.set(layer.get())
        return layer
    def start(self,sess):#Risky if the starting order matters?
        for sublayer in self.sublayers: sublayer.start(sess) 
        super().start(sess)
    def stop(self):
        super().stop()
        for sublayer in self.sublayers: sublayer.stop() 
    def finish(self):
        if not self.finished:
            self.make_sublayers(**self.SUBLAYER_OUTPUT_MANAGED)
            self.make_sublayers(**self.SUBLAYER_OUTPUT)
            for sublayer in self.sublayers:
                sublayer.finish()
            super().finish()
            
            
            

class _LayerVars(_LayerTree):#Gathering functions for variables, labels and some other parameters
    def __init__(self,**kwargs):#Needs reviewing
        super().__init__(**kwargs)
    def get_weights(self):
        return self.simple_gather_fun("get_weights")
    def get_biases(self):
        return self.simple_gather_fun("get_biases")
    def get_dropout(self):
        return self.simple_gather_fun("get_dropout")
    def get_vars(self):
        return self.get_weights()+self.get_biases()
    def simple_gather_fun(self,fun,**kwargs):#Abstract function for iterating across all sublayers
        return [param for sublayer in self.sublayers for param in sublayer.gather_self(fun,**kwargs)]
    def sub_gather_fun(self,fun,**kwargs):#Abstract function for iterating across all sublayers
        return [param for sublayer in self.sublayers for param in sublayer.gather_fun(fun,**kwargs)]
    def gather_fun(self,fun,**kwargs):#Also include the current layer
        return self.gather_self(fun,**kwargs)+self.sub_gather_fun(fun,**kwargs)
    def gather_self(self,fun,**kwargs):#Only the current layer
        return (fun in dir(self) and getattr(self,fun)(**kwargs) or [])
    def _get_labels(self):
        if "labels" in dir(self):
            if isinstance(self.labels,SimpleLayer):
                return self.labels.get()
            return self.labels
        return None
    def get_input_labels(self,recurse=True):
        if isinstance(self.x,SimpleLayer):
            return self.x.get_labels(recurse=recurse)
        return None
    def get_labels(self,recurse=True):
        labels=self._get_labels()
        if labels==None and recurse:
            return self.get_input_labels()
        return labels 

###Basic layer
class SimpleLayer(_LayerVars):
    type="Network"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
#Template for simple layer classes. Possible uses:
#Start from any existing layer type
#Modify the output using a function fun, must take and return a single Tensor, and possibly some extra (serializable, named) arguments
#Add input and/or output features to the layer
#Add or mofidy a default argument values for a layer
def make_layer(name,fun=None,args=None,BaseClass=SimpleLayer,in_features=None,out_features=None,**kwargs):
    assert(issubclass(BaseClass,SimpleLayer)),"Class %s is not a layer type"%BaseClass.__name__
    class LayerClass(BaseClass):
        type=name
        if args!=None:
            _extra_args=BaseClass._extra_args+args
        def __init__(self,**kwargs1):
            if in_features!=None:
                for feature in in_features:
                    if type(feature)==string:
                        feature={"type":feature}
                    self.add_sublayer_def(sublayer_type=self.SUBLAYER_INPUT_MANAGED,**feature)
            if out_features!=None:
                for feature in out_features:
                    if type(feature)==str:
                        feature={"type":feature}
                    self.add_sublayer_def(sublayer_type=self.SUBLAYER_OUTPUT_MANAGED,**feature)
            kwargs2=kwargs.copy()
            kwargs2.update(kwargs1)
            super().__init__(**kwargs2)
            arg_dic={}
            if args!=None:
                assert(fun!=None),"Argument list given but no function"
                for arg in args:
                    assert(arg in kwargs2),"Missing argument: %s"%arg
                    arg_dic[arg]=kwargs2[arg]
                    setattr(self,arg,kwargs2[arg])
            if fun!=None:
                self.y=fun(self.y,**arg_dic)
        if args!=None:
            def save(self,**kwargs3):
                for arg in args:
                    val=getattr(self,arg)
                    if val!=kwargs[arg]:
                        kwargs3[arg]=val
                return super().save(**kwargs3)
    LayerClass.__name__=name
    LayerFactory._add_class(LayerClass)
    return LayerClass

def batch_fun(fun,parallel_iterations=1024):
    def f(x,*args,**kwargs):
        return tf.map_fn(partial(fun,*args,**kwargs), x, parallel_iterations=parallel_iterations)
    return f
def make_batch_layer(name,fun,*args,make_both=True, parallel_iterations=1024,**kwargs):
    if make_both:
        return (make_layer(name=name,fun=fun,*args,make_both=False,**kwargs),
                make_batch_layer(name="Batch_"+name,fun=fun,*args,make_both=False,**kwargs))
    return make_layer(name=name,fun=batch_fun(fun,parallel_iterations),*args,**kwargs)


###Layer combining
class SimpleCombine:#Auxiliary class to Combine (needs reworking)
    _combineTypeError="Invalid combine type: %s"
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
        
class CombineLayer(SimpleLayer): #Combines layers in a parallel way (most stuff handled by _LayerTree class)
    type="Combine"
    SUBLAYER_COMBINE=dict(kw="combined_layers",lst="sublayers_combined",set_y=False)
    SUBLAYER_COMBINE_MANAGED=dict(kw="combined_layers_managed",lst="sublayers_combined_managed",set_y=False,managed=True,reverse=False)
    sublayer_types=_LayerTree.sublayer_types+[SUBLAYER_COMBINE,SUBLAYER_COMBINE_MANAGED]
    _combineManagedError="A combine layer should not combine both managed and unmanaged layers"
    def __init__(self,combine_op="combine",combined_layers=[],**kwargs):
        super().__init__(**kwargs)
        self.combine_op=combine_op           #Type of combining: "combine" (on channel dimension), "combine_batch", "sub", "mult"
        self.layers=layers                   #The combined layers
        self.make_sublayers(**self.SUBLAYER_COMBINE)
        self.make_sublayers(**self.SUBLAYER_COMBINE_MANAGED)
        assert(len(self.sublayers_combined)==0 or len(self.sublayers_combined_managed)==0),self._combineManagedError
        self.combine=SimpleCombine([layer.get() for layer in self.sublayers_combined+self.sublayers_combined_managed],
                                   combine_op=self.combine_op)
        self.y=self.combine.y
    def save(self,**kwargs):
        kwargs["combine_op"]=self.combine_op
        return super().save(**kwargs)



    



###Feature layers
class DropoutLayer(SimpleLayer):
    type="Dropout"
    def __init__(self,default_rate=1.0,**kwargs):
        super().__init__(**kwargs)
        self.default_rate=default_rate
        self.tf_keep_rate=tf.placeholder_with_default(np.float32(self.default_rate),[])
        self.y=tf.nn.dropout(self.y,self.tf_keep_rate)
    def get_dropout(self):
        return super().get_dropout()+[self.tf_keep_rate]
    def save(self,**kwargs):
        if self.default_rate!=1.0: kwargs["default_rate"]=self.default_rate
        return super().save(**kwargs)

def _batchNormFunction(x):
    mean,std=tf.nn.moments(x,range(len(x.get_shape().as_list())))
    return (x-mean)/std
BatchNormLayer=make_layer(name="Batch_Norm",fun=_batchNormFunction)
LocalResponseNormLayer=make_layer(name="Local_Response_Norm",fun=tf.nn.local_response_normalization)





###Linear and related layers
class LinearLayer(SimpleLayer):
    type="Linear"
    def __init__(self,size=1,rand_scale=0.1,**kwargs):
        super().__init__(**kwargs)
        self.size=size                       #Number of output channels
        self.rand_scale=rand_scale           #Scale for random initialization of weights and biases
        shape=self.y.get_shape()[1:]
        if len(shape)>1:
            self.y=tf.reshape(self.y,[-1,shape.num_elements()])
        self.shape=[shape.num_elements(),size]
        if self.cloned!=None:
            self.w=self.cloned.w
            self.b=self.cloned.b
        else:
            self.w=tf.Variable(tf.random_normal(self.shape, 0, self.rand_scale),name='Weights')
            self.b=tf.Variable(tf.random_normal([size], self.rand_scale, self.rand_scale),name='Biases')
        self.y=tf.matmul(self.y,self.w)+self.b
    def save(self,**kwargs):
        kwargs["size"]=self.size
        kwargs["rand_scale"]=self.rand_scale
        return super().save(**kwargs)
    def get_weights(self):
        return super().get_weights()+[self.w]
    def get_biases(self):
        return super().get_biases()+[self.b]


ReluFeature=make_layer(name="Relu_Feature",fun=tf.nn.relu,BaseClass=SimpleLayer)
SoftmaxFeature=make_layer(name="Softmax_Feature",fun=tf.nn.softmax,BaseClass=SimpleLayer)
SigmoidFeature=make_layer(name="Sigmoid_Feature",fun=tf.nn.sigmoid,BaseClass=SimpleLayer)


#Same as linear layer with feature on output (move to actual features?)

ReluLayer=make_layer(name="Relu",out_features=["Relu_Feature"],BaseClass=LinearLayer)
SoftmaxLayer=make_layer(name="Softmax",out_features=["Softmax_Feature"],BaseClass=LinearLayer)
SigmoidLayer=make_layer(name="Sigmoid",out_features=["Sigmoid_Feature"],BaseClass=LinearLayer)



###Convolution and pooling
class WindowLayer(SimpleLayer): #Common elements of convolution and pooling (abstract class)
    type="Abstract"
    def __init__(self,pad="VALID",window=3,stride=1,**kwargs):
        super().__init__(**kwargs)
        self.pad=pad                         #Type of padding used
        self.window=window                   #Size of the input window
        self.stride=stride                   #Stride distance of the input window
    def save(self,**kwargs):
        kwargs["pad"]=self.pad
        kwargs["window"]=self.window
        kwargs["stride"]=self.stride
        return super().save(**kwargs)
        
class ConvLayer(WindowLayer):
    type="Convolution"
    def __init__(self,size=1,relu=True,input_channels=None,rand_scale=0.1,**kwargs):
        if relu:
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_OUTPUT_MANAGED,type="Relu_Feature")
        super().__init__(**kwargs)
        self.size=size                       #Number of output channels
        self.relu=relu                       #Optional relu on the output
        self._input_channels=input_channels  #(Optional) overrides the number of input channels, needed for variable size input
        self.rand_scale=rand_scale           #Scale for random initialization of weights and biases
        self.input_channels=self._input_channels or self.y.get_shape()[3].value
        self.filter_shape=[self.window,self.window,self.input_channels,self.size]
        if self.cloned!=None:
            self.w=self.cloned.w
            self.b=self.cloned.b
        else:
            self.w=tf.Variable(tf.random_normal(self.filter_shape, 0, self.rand_scale),name='Weights')
            self.b=tf.Variable(tf.random_normal([self.size], rand_scale, self.rand_scale),name='Biases')
        self.set(tf.nn.bias_add(tf.nn.conv2d(self.y,self.w,[1,self.stride,self.stride,1], self.pad),self.b))
    def get_weights(self):
        return super().get_weights()+[self.w]
    def get_biases(self):
        return super().get_biases()+[self.b]
    def save(self,**kwargs):
        kwargs["size"]=self.size
        kwargs["relu"]=self.relu
        kwargs["input_channels"]=self._input_channels
        kwargs["rand_scale"]=self.rand_scale
        return super().save(**kwargs)
    
class PoolLayer(WindowLayer):
    type="Pool"
    _poolTypeError="Invlid pooling type: %s"
    def __init__(self,pool_type="max",stride=2,**kwargs):
        super().__init__(stride=stride,**kwargs)
        self.pool_type=pool_type             #avg or max pooling 
        self.ksize=[1, self.window, self.window, 1]
        self.strides=[1, self.stride, self.stride, 1]
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
        return super().save(**kwargs)


    


    




###Complex Layers
'''class RandomCombineLayer(CombineLayer):
    type="Random_Combined"
    def __init__(self,channels=1,**kwargs):
        layers=[{"Type":"Identity"},kwargs.update({"Type":"Random","shape":[-1 for i in range(x.get_shape().ndims-1)]+[channels]})]
        super().__init__(layers=layers,**kwargs)
    def save(self,**kwargs):
        kwargs.update(self.layers[1].save(kwargs))
        kwargs["type"]=self.type
        return kwargs'''
    
#Old

'''class BatchNormLayer(SimpleLayer): #Batch normalization
    type="Batch_Norm"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.mean,self.std=tf.nn.moments(self.y,range(len(self.y.get_shape().as_list())))
        self.y=(self.y-self.mean)/self.std


class LocalResponseNormLayer(SimpleLayer): #Local response normalization
    type="Local_Response_Norm"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        assert(self.y.get_shape().ndims==4), "Local response normalization requires a 4d tensor"
        self.y=tf.nn.local_response_normalization(self.y)'''

'''class ReluFeature(SimpleLayer):
    type="Relu_Feature"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.y=tf.nn.relu(self.y)
        
class SoftmaxFeature(SimpleLayer):
    type="Softmax_Feature"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.y=tf.nn.softmax(self.y)

class SigmoidFeature(SimpleLayer):
    type="Sigmoid_Feature"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.y=tf.nn.sigmoid(self.y)'''

'''class ReluLayer(LinearLayer):
    type="Relu"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.relu=self.add_sublayer(ReluFeature(x=self.y),set_y=True)
        
class SoftmaxLayer(LinearLayer):
    type="Softmax"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.softmax=self.add_sublayer(SoftmaxFeature(x=self.y),set_y=True)
        
class SigmoidLayer(LinearLayer):
    type="Sigmoid"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.sigmoid=self.add_sublayer(SigmoidFeature(x=self.y),set_y=True)
        
class QuadLayer(LinearLayer):
    type="Quadratic"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.y=tf.mul(self.y,self.y)'''

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

'''class Network(Composite):#Similar to SimpleLayer, but proper layers are distinguished from feature layers
    type="Network"
    def __init__(self,**kwargs):
        layers=[]
        Composite.__init__(self,**kwargs)
        for layer in self._layers:
            self.add_sublayer(Layer(x=self.y,**layer),_layer=True)
            self.y=self.layers[-1].get()'''  

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
        
                  