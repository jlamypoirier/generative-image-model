import numpy as np
import tensorflow as tf
import warnings
from functools import partial
from nnUtils import *
from nnHandler import *
#import os
#from dataProducer import *



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
        if type(t)==str:
            t=[t]
        for tp in t:
            if tp!="Abstract":
                LayerFactory.classes[tp]=cl
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
    def build(*args,**kwargs):
        if "type" in kwargs:#Standard case
            _type=kwargs.pop("type")
            return LayerFactory._build(_type,*args,**kwargs)
        elif type(args[0])==str:#Build from string
            return LayerFactory._build(*args,**kwargs)
        elif type(args[0])==lis:#Build network from list of layers
            assert(len(args)==1 and kwargs=={}),"Too many arguments"
            return LayerFactory._build(type="Network",layers=args[0])
    @staticmethod
    def _build(type,*args,**kwargs):
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
    def __init__(self,x=None,device=None,**kwargs):
        self.desc=itercopy(kwargs)
        self.desc["type"]=self.get_type()
        self.input=x                         #The input for the layer (can be a Layer, a Tensor, a numpy array, or None)
        self.x=x                             #The input, converted to tensor-like (or None)
        self.device=device                   #Allow manual device placement
        if isinstance(self.x,_LayerRaw):
            self.x=self.x.get()
        if self.device!=None:
            with tf.device(self.device):
                self.init(kwargs)
        else:
            self.init(kwargs)
        for kw in kwargs:
            #print(self.type)
            #assert(False)
            warnings.warn(self._uncaughtArgumentWarning%kw)
        self.y=self.x
        self.dropped=self.test and self.drop_on_test or (self.drop_on_test==None and "DROP_ON_TEST" in dir(self) and self.DROP_ON_TEST)
        if not self.dropped:
            if self.device!=None:
                with tf.device(self.device):
                    self.call()
                    self.finish()
            else:
                self.call()
                self.finish()
        self._y=self.y                       #The final output
        del self.y
        self.register()
    def init(self,kwargs):
        self.type=self.__class__.type        #The type of the layer (user-friendly class name)
        self.drop_on_test=kwargs.pop("drop_on_test",None) #Layer is ignored if test=True 
                                             #(defaults to False or layer's DROP_ON_TEST variable)
        self.test=kwargs.pop("test",False)   #Indicates test phase (propagates to sublayers if set)
    def get(self):                           #Garanteed to be the final output
        return self._y
    def get_type(self):
        if type(self.type)==str:
            return self.type
        return self.type[0]
    def save(self):
        return itercopy(self.desc)


class _LayerInstance(_LayerRaw):#Session management (use as base class for saver and trainer?)
    type="Abstract"
    _sessionError="No active session"
    def init(self,kwargs):
        super().init(kwargs)
        self.sess=None
    def start(self,sess):
        #if self.sess!=None:
        #    warnings.warn("Layer is already running")
        self.sess=sess       #Should be a SessManager
        self._start()
    def register(self):
        SessManager.register(self)
    def stop(self):
        self.sess=None
    def run(self,feed_dict=None):
        assert(self.sess!=None),self._sessionError
        return self.sess.run(self.get(),feed_dict=feed_dict)
    
class _LayerCopy(_LayerInstance):#Copying and cloning
    type="Abstract"
    _copyTypeError='Trying to share variables between layers of different type: "%s" vs "%s"'
    _copyManagerError="Copying variable requires a manager with an active session"
    _copySessionError="Copying variable requires both networks to have the same active session"
    _copyNetworkError="Copying variable requires identical networks"
    def init(self,kwargs):
        super().init(kwargs)
        self.cloned=kwargs.pop("_cloned",None)     #The copied layer, for variable sharing 
        #if self.cloned!=None:
        #    assert(self.type==self.cloned.type), self._clonedTypeError%(self.type,self.cloned.type)
    def export(self):#Attempt, export the definition with reference to copied layers
        sav=self.save()
        sav["_cloned"]=self
        return sav
    def copy(self,                            #Makes a copy of the graph, with or without variable sharing
             x=None,                          #Feed a new input x or use the existing one (x=None)
             share_vars=True,                #Enables variable sharing
             copy_vars=False,                 #Copy the variable values (requires a session manager with an active session)
             **kwargs                         #Override some other arguments (risky)
            ):                                #(ex. test=True)
        if x==None:
            x=self.x
        save=share_vars and self.export() or self.save()
        save.update(kwargs)
        layer=Layer(x=x,**save)
        if layer.sess==None and self.sess!=None:
            self.sess.add([layer])
        if copy_vars and not share_vars:
            self.copy_vars_to(layer)
        return layer
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
    #Layers modifying the input / output
    SUBLAYER_INPUT=dict(kw="in_features",attr="sublayers_in",set_y=True,arg=True)
    SUBLAYER_OUTPUT=dict(kw="out_features",attr="sublayers_out",set_y=True,arg=True)
    #The layers of the network, if applicable
    SUBLAYER_PROPER=dict(kw="layers",attr="sublayers_proper",set_y=True,arg=True)
    #Layers added by derived classes, should be added in init function
    SUBLAYER_INPUT_MANAGED=dict(kw="in_features_managed",attr="sublayers_in_managed",set_y=True,arg=False,managed=True,reverse=False)
    SUBLAYER_OUTPUT_MANAGED=dict(kw="out_features_managed",attr="sublayers_out_managed",set_y=True,arg=False,managed=True,reverse=True)
    SUBLAYER_PROPER_MANAGED=dict(kw="layers_managed",attr="sublayers_proper_managed",set_y=True,arg=False,managed=True,reverse=False)
    sublayer_types=[SUBLAYER_INPUT,SUBLAYER_OUTPUT,SUBLAYER_INPUT_MANAGED,SUBLAYER_OUTPUT_MANAGED,SUBLAYER_PROPER,SUBLAYER_PROPER_MANAGED]
    #_extra_args=_LayerCopy._extra_args+[type["kw"] for type in sublayer_types if "arg" in type and type["arg"]]
    def init(self,kwargs):
        super().init(kwargs)
        self.sublayers=[]                        #All the sublayers of the layer
        for type in self.sublayer_types:         #The sublayers sorted by type: def at getattr(self,type["kw"]
            if "arg" in type and type["arg"]:
                setattr(self,type["kw"],kwargs.pop(type["kw"],[]) or [])
            elif type["kw"] not in dir(self):
                setattr(self,type["kw"],[])
            setattr(self,type["attr"],[])        #Instances at getattr(self,type["lst"])
    def call(self):
        self.make_sublayers(**self.SUBLAYER_INPUT)
        self.make_sublayers(**self.SUBLAYER_INPUT_MANAGED)
        self.make_sublayers(**self.SUBLAYER_PROPER)
        self.make_sublayers(**self.SUBLAYER_PROPER_MANAGED)
    def _export(self,sav):#Append _cloned variable to all sublayer definitions
        sav["_cloned"]=self
        for _type in self.sublayer_types:
            if _type["kw"] in sav and sav[_type["kw"]]!=None:
                sublayers=getattr(self,_type["attr"])
                exp_sublayers=sav[_type["kw"]]
                for i,sublayer in enumerate(sublayers):
                    #exp_sublayers[i]["_cloned"]=sublayer
                    if type(exp_sublayers[i])==str:
                        exp_sublayers[i]=dict(type=exp_sublayers[i])
                    sublayer._export(exp_sublayers[i])
        return sav
    def export(self):
        return self._export(super().export())
    def make_sublayers(self,kw,attr,set_y,**kwargs):
        desc=getattr(self,kw)
        if self.cloned:
            cloned=getattr(self.cloned,attr)#getattr(self,attr))
            #assert(len(desc)<=len(cloned)) #Allow removing some features, but only the last ones
        else:
            cloned=[None for _ in desc]
        for i,_desc in enumerate(desc):#Features implemented in the order they are given
            if type(_desc)==dict:
                _desc=_desc.copy()
                if self.test!=None:
                    _desc["test"]=self.test
                _cloned=_desc.pop("_cloned",cloned[i])#Redundant
                x=_desc.pop("x",self.y)
                if type(x)==str:
                    x=getattr(self,x)
                layer_attr=_desc.pop("attr", None)
                set_y=_desc.pop("set_y", set_y)
                layer=Layer(x=x,_cloned=_cloned,**_desc)
                if layer_attr!=None:
                    setattr(self,layer_attr,layer)
            else:
                layer=Layer(x=self.y,_cloned=cloned[i],test=self.test,type=_desc)
            self.add_sublayer(layer,attr=attr,set_y=set_y)
    def add_sublayer_def(self,sublayer_type,inner=True,**kwargs):
        assert("y" not in dir(self) and "_y" not in dir(self)), "Sublayer definitions must be added before initializing the layer"
        assert("managed" in sublayer_type and sublayer_type["managed"]), "Adding a definition for a wrong type of layer"
        if sublayer_type["kw"] not in dir(self):
            setattr(self,sublayer_type["kw"],[])
        reverse="reverse" in sublayer_type and sublayer_type["reverse"] or False
        if inner==reverse:
            getattr(self,sublayer_type["kw"]).insert(0,kwargs)
        else:
            getattr(self,sublayer_type["kw"]).append(kwargs)
    def add_sublayer(self,layer,attr=None,set_y=False):
        self.sublayers.append(layer)
        if attr!=None:
            if attr not in dir(self):
                setattr(self,attr,[])
            getattr(self,attr).append(layer)
        if set_y:
            self.y=layer.get()
        return layer
    def _start(self):#Risky if the starting order matters?
        for sublayer in self.sublayers: sublayer.start(self.sess)
    def stop(self):
        for sublayer in self.sublayers: sublayer.stop()
        super().stop()
    def finish(self):
        self.make_sublayers(**self.SUBLAYER_OUTPUT_MANAGED)
        self.make_sublayers(**self.SUBLAYER_OUTPUT)
            
            
            

class _LayerVars(_LayerTree):#Gathering functions for variables, labels and some other parameters (Needs reviewing)
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
    def _get_labels_lst(self):
        labels=self._get_labels()
        return labels!=None and [labels] or []
    def _get_labels(self):
        #print(self.type)
        if "labels" in dir(self):
            if isinstance(self.labels,SimpleLayer):
                return self.labels.get()
            return self.labels
        sub_labels=self.simple_gather_fun("_get_labels_lst")
        if len(sub_labels)>0:
            return sub_labels[-1]#Picks last label, should also be last in the network
        return None
    def get_input_labels(self,recurse=True):
        if isinstance(self.input,SimpleLayer):
            return self.input.get_labels(recurse=recurse)
        return None
    def get_labels(self,recurse=True):
        labels=self._get_labels()
        if labels==None and recurse:
            return self.get_input_labels()
        return labels 

###Basic layer (Rename?)
class SimpleLayer(_LayerVars):
    type=["Network","Identity"]
        
#Template for simple layer classes. Possible uses:
#Start from any existing layer type
#Modify the output using a function fun, must take and return a single Tensor, and possibly some extra (serializable, named) arguments
#Add input and/or output features to the layer
#Add or mofidy a default argument values for a layer
def make_layer(name,fun=None,args=None,BaseClass=SimpleLayer,in_features=None,out_features=None,drop_on_test=None,**kwargs):
    assert(issubclass(BaseClass,SimpleLayer)),"Class %s is not a layer type"%BaseClass.__name__
    kwargs=kwargs.copy()
    class LayerClass(BaseClass):
        type=name
        if drop_on_test!=None:
            DROP_ON_TEST=drop_on_test
        if args!=None:
            assert(fun!=None),"Argument list given but no function"
        def init(self,kwargs1):
            if in_features!=None:
                for feature in in_features:
                    if type(feature)==str:
                        feature={"type":feature}
                    self.add_sublayer_def(sublayer_type=self.SUBLAYER_INPUT_MANAGED,**feature)
            if out_features!=None:
                for feature in out_features:
                    if type(feature)==str:
                        feature={"type":feature}
                    self.add_sublayer_def(sublayer_type=self.SUBLAYER_OUTPUT_MANAGED,**feature)
            #kwargs2=kwargs.copy()
            #kwargs2.update(kwargs1)
            for key in kwargs:
                if key not in kwargs1:
                    kwargs1[key]=kwargs[key]
            super().init(kwargs1)
            self.arg_dic={}
            if args!=None:
                for arg in args:
                    setattr(self,arg,kwargs1.pop(arg,None))
                    self.arg_dic[arg]=getattr(self,arg)
        if fun!=None:
            def call(self):
                super().call()
                self.y=fun(self.y,**self.arg_dic)
    LayerClass.__name__=type(name)==list and name[0] or name
    LayerFactory._add_class(LayerClass)
    return LayerClass

def batch_fun(fun,parallel_iterations=1024):
    def f(x,*args,**kwargs):
        return tf.map_fn(partial(fun,*args,**kwargs), x, parallel_iterations=parallel_iterations)
    return f
def make_batch_layer(name,fun,*args,make_both=True, parallel_iterations=1024,**kwargs):
    if make_both:
        return (make_layer(name=name,fun=fun,*args,**kwargs),
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
        elif combine_op=="add":
            self.y=tf.add_n(self.x)
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
    type=["Combine","Merge"]
    SUBLAYER_COMBINE=dict(kw="combined_layers",attr="sublayers_combined",set_y=False)
    SUBLAYER_COMBINE_MANAGED=dict(kw="combined_layers_managed",attr="sublayers_combined_managed",set_y=False,managed=True,reverse=False)
    sublayer_types=_LayerTree.sublayer_types+[SUBLAYER_COMBINE,SUBLAYER_COMBINE_MANAGED]
    _combineManagedError="A combine layer should not combine both managed and unmanaged layers"
    def init(self,kwargs):
        super().init(kwargs)
        self.combine_op=kwargs.pop("combine_op","combine")#Type of combining: "combine" (on channel dimension), 
                                                       #"combine_batch", "sub", "mult"
        self.combined_layers=kwargs.pop("combined_layers",[]) #The combined layers
    def call(self):
        super().call()
        self.make_sublayers(**self.SUBLAYER_COMBINE)
        self.make_sublayers(**self.SUBLAYER_COMBINE_MANAGED)
        assert(len(self.sublayers_combined)==0 or len(self.sublayers_combined_managed)==0),self._combineManagedError
        self.combine=SimpleCombine([layer.get() for layer in self.sublayers_combined+self.sublayers_combined_managed],
                                   combine_op=self.combine_op)
        self.y=self.combine.y



    



###Feature layers
class DropoutLayer(SimpleLayer):
    type="Dropout"
    DROP_ON_TEST=True
    def init(self,kwargs):
        super().init(kwargs)
        self.default_rate=kwargs.pop("default_rate",1.0)
        self.tf_keep_rate=tf.placeholder_with_default(np.float32(self.default_rate),[])
    def call(self):
        super().call()
        self.y=tf.nn.dropout(self.y,self.tf_keep_rate)
    def get_dropout(self):
        return super().get_dropout()+[self.tf_keep_rate]

def _batchNormFunction(x):
    mean,std=tf.nn.moments(x,range(len(x.get_shape().as_list())))
    return (x-mean)/std
BatchNormLayer=make_layer(name=["Batch_Norm","BatchNormalization"],fun=_batchNormFunction)
LocalResponseNormLayer=make_layer(name="Local_Response_Norm",fun=tf.nn.local_response_normalization)

'''def _reshape(x,shape):
    x_shape=x.get_shape().as_list()
    print(x_shape)
    print(shape)
    for i in range(len(shape)):
        if shape[i]==None:
            shape[i]=x_shape[i]
    print(shape)
    return tf.reshape(x,shape)'''
ReshapeLayer=make_layer(name="Reshape",fun=tf.reshape,args=["shape"])


###Linear and related layers
class LinearLayer(SimpleLayer):
    type=["Linear","Dense"]
    def init(self,kwargs):
        super().init(kwargs)
        self.size=kwargs.pop("size",1)              #Number of output channels
        self.rand_scale=kwargs.pop("rand_scale",0.1)#Scale for random initialization of weights and biases
    def call(self):
        super().call()
        shape=self.y.get_shape()[1:]
        if len(shape)>1:
            self.y=tf.reshape(self.y,[-1,shape.num_elements()])
        self.shape=[shape.num_elements(),self.size]
        if self.cloned!=None:
            assert(self.shape==self.cloned.shape)
            self.w=self.cloned.w
            self.b=self.cloned.b
        else:
            self.w=tf.Variable(tf.random_normal(self.shape, 0, self.rand_scale),name='Weights')
            self.b=tf.Variable(tf.random_normal([self.size], self.rand_scale, self.rand_scale),name='Biases')
        self.y=tf.matmul(self.y,self.w)+self.b
    def get_weights(self):
        return super().get_weights()+[self.w]
    def get_biases(self):
        return super().get_biases()+[self.b]


ReluFeature=make_layer(name=["Relu_Feature","Relu_Activation"],fun=tf.nn.relu,BaseClass=SimpleLayer)
SoftmaxFeature=make_layer(name=["Softmax_Feature","Softmax_Activation"],fun=tf.nn.softmax,BaseClass=SimpleLayer)
SigmoidFeature=make_layer(name=["Sigmoid_Feature","Sigmoid_Activation"],fun=tf.nn.sigmoid,BaseClass=SimpleLayer)


#Same as linear layer with feature on output (move to actual features?)

ReluLayer=make_layer(name="Relu",out_features=["Relu_Feature"],BaseClass=LinearLayer)
SoftmaxLayer=make_layer(name="Softmax",out_features=["Softmax_Feature"],BaseClass=LinearLayer)
SigmoidLayer=make_layer(name="Sigmoid",out_features=["Sigmoid_Feature"],BaseClass=LinearLayer)



###Convolution and pooling (Only supports 2d)
class WindowLayer(SimpleLayer): #Common elements of convolution and pooling (abstract class)
    type="Abstract"
    def init(self,kwargs):
        super().init(kwargs)
        self.pad=kwargs.pop("pad","VALID")            #Type of padding used
        self.window=kwargs.pop("window",3)            #Size of the input window
        self.stride=kwargs.pop("stride",1)            #Stride distance of the input window
        self.input_stride=kwargs.pop("input_stride",1)#Input stride
        assert(self.stride==1 or self.input_stride==1)
        
class ConvLayer(WindowLayer):
    type=["Convolution","Convolution2D"]#"AtrousConvolution2D"
    def init(self,kwargs):
        super().init(kwargs)
        self.size=kwargs.pop("size",1)       #Number of output channels
        self.rand_scale=kwargs.pop("rand_scale",0.1)#Scale for random initialization of weights and biases
        self.relu=kwargs.pop("relu",True)    #Optional relu on the output
        self.input_channels=kwargs.pop("input_channels",None) #(Optional) overrides the number of input channels, 
                                                               #needed for variable size input
        if self.relu:
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_OUTPUT_MANAGED,type="Relu_Feature")
    def call(self):
        super().call()
        self.input_channels=self.input_channels or self.y.get_shape()[3].value
        self.filter_shape=[self.window,self.window,self.input_channels,self.size]
        if self.cloned!=None:
            self.w=self.cloned.w
            self.b=self.cloned.b
        else:
            self.w=tf.Variable(tf.random_normal(self.filter_shape, 0, self.rand_scale),name='Weights')
            self.b=tf.Variable(tf.random_normal([self.size], self.rand_scale, self.rand_scale),name='Biases')
        if self.input_stride!=1:
            self.y=tf.nn.bias_add(tf.nn.atrous_conv2d(self.y,filters=self.w,rate=self.input_stride,padding=self.pad),self.b)
        else:
            strides=[1,self.stride,self.stride,1]
            self.y=tf.nn.bias_add(tf.nn.conv2d(self.y,filter=self.w,strides=strides, padding=self.pad),self.b)
    def get_weights(self):
        return super().get_weights()+[self.w]
    def get_biases(self):
        return super().get_biases()+[self.b]
    
class ConvTransposeLayer(WindowLayer):
    type=["Convolution_Transpose","Deconvolution2D"]
    def init(self,kwargs):
        super().init(kwargs)
        self.size=kwargs.pop("size",1)       #Number of output channels
        self.rand_scale=kwargs.pop("rand_scale",0.1)#Scale for random initialization of weights and biases
        self.relu=kwargs.pop("relu",True)    #Optional relu on the output
        self._input_channels=kwargs.pop("input_channels",None) #(Optional) overrides the number of input channels, 
                                                               #needed for variable size input
        if self.relu:
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_OUTPUT_MANAGED,type="Relu_Feature")
    def call(self):
        super().call()
        self.input_channels=self._input_channels or self.y.get_shape()[3].value
        self.filter_shape=[self.window,self.window,self.size,self.input_channels]
        if self.cloned!=None:
            self.w=self.cloned.w
            self.b=self.cloned.b
        else:
            self.w=tf.Variable(tf.random_normal(self.filter_shape, 0, self.rand_scale),name='Weights')
            self.b=tf.Variable(tf.random_normal([self.size], self.rand_scale, self.rand_scale),name='Biases')
        if None in self.y.get_shape().as_list():
            output_0=tf.shape(self.y)[0]
            output_1=tf.shape(self.y)[1]*self.stride+(self.pad=="VALID" and self.window-1 or 0)
            output_2=tf.shape(self.y)[2]*self.stride+(self.pad=="VALID" and self.window-1 or 0)
        else:
            output_0=self.y.get_shape()[0].value
            output_1=self.y.get_shape()[1].value*self.stride+(self.pad=="VALID" and self.window-1 or 0)
            output_2=self.y.get_shape()[2].value*self.stride+(self.pad=="VALID" and self.window-1 or 0)
        output_shape=[output_0,output_1,output_2,self.size]
        strides=[1,self.stride,self.stride,1]
        self.y=tf.nn.bias_add(tf.nn.conv2d_transpose(self.y,self.w, output_shape,strides,self.pad),self.b)
    def get_weights(self):
        return super().get_weights()+[self.w]
    def get_biases(self):
        return super().get_biases()+[self.b]
    
    
class PoolLayer(WindowLayer):
    type="Pool"#["Pool","MaxPooling2D","AveragePooling2D"]
    _poolTypeError="Invlid pooling type: %s"
    def init(self,kwargs):
        if "stride" not in kwargs:
            kwargs["stride"]=2
        super().init(kwargs)
        self.pool_type=kwargs.pop("pool_type","max")  #avg or max pooling 
        self.ksize=[1, self.window, self.window, 1]
        self.strides=[1, self.stride, self.stride, 1]
    def call(self):
        super().call()
        if self.pool_type=="max":
            self.y=tf.nn.max_pool(self.y, ksize=self.ksize, 
                      strides=self.strides,padding=self.pad)
        elif self.pool_type=="avg":
            self.y=tf.nn.avg_pool(self.y, ksize=self.ksize, 
                      strides=self.strides,padding=self.pad)
        else: 
            raise Exception(self._poolTypeError%self.pool_type)




    




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
        
                  