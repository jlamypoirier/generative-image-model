import numpy as np
import tensorflow as tf
import warnings
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




###Basic layer
class SimpleLayer:
    type="Identity"
    _uncaughtArgumentWarning="Unknown argument: %s"
    _clonedTypeError='Trying to clone layers of different type: "%s" vs "%s"'
    _copyManagerError="Copying variable requires a manager with an active session"
    _copySessionError="Copying variable requires both networks to have the same active session"
    _copyNetworkError="Copying variable requires identical networks"
    def __init__(self,x=None,features=[],_cloned_layer=None,**kwargs):
        for kw in kwargs:
            warnings.warn(self._uncaughtArgumentWarning%kw)
        self.type=self.__class__.type        #The type of the layer (user-friendly class name)
        self._x=x                            #The input for the layer (can be a Layer, a Tensor, or a numpy array)
        self.x=self._x                       #The input, guaranteed to be a tensor-like or None
        self.features=features               #A set of layers modifying the input. Can be any layer, but meant for feature layers
        self.cloned=_cloned_layer            #The cloned layer, if applicable (should be set through the cloner function only)
        self.y=self.x                        #The output of the layer
        self.sublayers=[]                    #All the sublayers of the layer
        self.feature_sublayers=[]            #The feature sublayers
        if self.cloned!=None:
            assert(self.type==_cloned_layer.type), self._clonedTypeError%(self.type,_cloned_layer.type)
        if isinstance(self.y,SimpleLayer):
            self.y=self.y.get()
            self._x=self._x.get()
        if self.cloned:
            cloned_features=self.cloned.features
            assert(len(self.features)<=len(cloned_features)) #Allow removing some features, but only the last ones
        else:
            cloned_features=[None]*len(self.features)
        for i,feature in enumerate(self.features):#Features implemented in the order they are given
            if type(feature)==dict:
                layer=Layer(x=self.y,_cloned_layer=cloned_features[i],**feature)
            else:
                layer=Layer(x=self.y,type=feature,_cloned_layer=cloned_features[i])
            self.add_sublayer(layer,_feature=True)
            self.y=layer.get()
    def save(self,**kwargs):
        kwargs["type"]=self.type
        if len(self.feature_sublayers)>0:kwargs["features"]=[feature.save() for feature in self.feature_sublayers]
        return kwargs
    def _getLabels(self): #Need label producing layers
        if "labels" in dir(self):
            if isinstance(self.labels,SimpleLayer):
                return self.labels.get()
            return self.labels
        return None
    def getLabels(self):
        labels=self._getLabels()
        if labels==None:
            return self.getInputLabels()
        return labels 
    def getInputLabels(self):
        if isinstance(self.x,SimpleLayer):
            return self.x.getLabels()
        return None
    def add_sublayer(self,sublayer,_feature=False):
        self.sublayers.append(sublayer)
        if _feature:
            self.feature_sublayers.append(sublayer)
        return sublayer
    def get(self):
        return self.y
    def getVars(self):
        return self.getWeights()+self.getBiases()
    def getWeights(self):
        return [weight for sublayer in self.sublayers for weight in sublayer.getWeights()]
    def getBiases(self):
        return [bias for sublayer in self.sublayers for bias in sublayer.getBiases()]
    def getDropout(self):
        return [rate for sublayer in self.sublayers for rate in sublayer.getDropout()]
    def getIn(self):
        return self.x
    def start(self,sess):
        self.sess=sess
        for sublayer in self.sublayers: sublayer.start(sess) 
    def stop(self):
        self.sess=None
        for sublayer in self.sublayers: sublayer.stop() 
    def run(self):
        assert(self.sess!=None),"No active session"
        return self.sess.run(self.y)
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
            


#Feature layers
class DropoutLayer(SimpleLayer):
    type="Dropout"
    def __init__(self,default_rate=1.0,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        self.default_rate=default_rate
        self.tf_keep_rate=tf.placeholder_with_default(np.float32(self.default_rate),[])
        self.y=tf.nn.dropout(self.y,self.tf_keep_rate)
    def getDropout(self):
        return [self.tf_keep_rate]
    def save(self,**kwargs):
        if self.default_rate!=1.0: kwargs["default_rate"]=self.default_rate
        return SimpleLayer.save(self,**kwargs)

class BatchNormLayer(SimpleLayer): #Batch normalization
    type="Batch_Norm"
    def __init__(self,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        self.mean,self.std=tf.nn.moments(self.y,range(len(self.y.get_shape().as_list())))
        self.y=(self.y-mean)/std


class LocalResponseNormLayer(SimpleLayer): #Local response normalization
    type="Local_Response_Norm"
    def __init__(self,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        assert(self.y.get_shape().ndims==4), "Local response normalization requires a 4d tensor"
        self.y=tf.nn.local_response_normalization(self.y)




###Linear and related layers
class LinearLayer(SimpleLayer):
    type="Linear"
    def __init__(self,size=1,rand_scale=0.1,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
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
        return SimpleLayer.save(self,**kwargs)
    def getWeights(self):
        return [self.w]
    def getBiases(self):
        return [self.b]
            

class SimpleRelu(SimpleLayer):
    type="Simple_Relu"
    def __init__(self,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        self.y=tf.nn.relu(self.y)
        

class ReluLayer(LinearLayer):
    type="Relu"
    def __init__(self,**kwargs):
        LinearLayer.__init__(self,**kwargs)
        self.relu=self.add_sublayer(SimpleRelu(x=self.y))
        self.y=self.relu.y
        
class SimpleSoftmax(SimpleLayer):
    type="Simple_Softmax"
    def __init__(self,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        self.y=tf.nn.softmax(self.y)
        
class SoftmaxLayer(LinearLayer):
    type="Softmax"
    def __init__(self,**kwargs):
        LinearLayer.__init__(self,**kwargs)
        self.softmax=self.add_sublayer(SimpleSoftmax(x=self.y))
        self.y=self.softmax.y
        
class SimpleSigmoid(SimpleLayer):
    type="Simple_Sigmoid"
    def __init__(self,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        self.y=tf.nn.sigmoid(self.y)
        
class SigmoidLayer(LinearLayer):
    type="Sigmoid"
    def __init__(self,**kwargs):
        LinearLayer.__init__(self,**kwargs)
        self.sigmoid=self.add_sublayer(SimpleSigmoid(x=self.y))
        self.y=self.sigmoid.y
        
class QuadLayer(LinearLayer):
    type="Quadratic"
    def __init__(self,**kwargs):
        LinearLayer.__init__(self,**kwargs)
        self.y=tf.mul(self.y,self.y)

###Convolution and pooling
class WindowLayer(SimpleLayer): #Common elements of convolution and pooling (abstract class)
    type="Abstract"
    def __init__(self,pad="VALID",window=3,stride=1,**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        self.pad=pad                         #Type of padding used
        self.window=window                   #Size of the input window
        self.stride=stride                   #Stride distance of the input window
    def save(self,**kwargs):
        kwargs["pad"]=self.pad
        kwargs["window"]=self.window
        kwargs["stride"]=self.stride
        return SimpleLayer.save(self,**kwargs)
        
class ConvLayer(WindowLayer):
    type="Convolution"
    def __init__(self,size=1,relu=True,input_channels=None,rand_scale=0.1,**kwargs):
        WindowLayer.__init__(self,**kwargs)
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
            self.w=tf.Variable(tf.random_normal(self.filter_shape, 0, self.rand_scale),name='Weights')
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
    def __init__(self,pool_type="max",stride=2,**kwargs):
        WindowLayer.__init__(self,stride=stride,**kwargs)
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
        return WindowLayer.save(self,**kwargs)


    

###Layer combining and composition
class Composite(SimpleLayer): #Common elements of composite layers (abstract class)
    type="Abstract"
    _clonedSublayerError="Trying to clone networks with a different number of sublayers: %s vs %s"
    def __init__(self,layers=[],**kwargs):
        SimpleLayer.__init__(self,**kwargs)
        self._layers=layers                  #Definition of the layers
        if "_cloned_layer" in kwargs and kwargs["_cloned_layer"]!=None:
            cloned=kwargs["_cloned_layer"].layers
            assert(len(_layers)==len(cloned)), self._clonedSublayerError%(len(_layers),len(cloned))
            for i,_layer in enumerate(_layers):
                _layer["_cloned_layer"]=cloned[i]
        self.layers=[]
    def save(self,**kwargs):
        kwargs["layers"]=[layer.save() for layer in self.layers]
        return SimpleLayer.save(self,**kwargs)
    def add_sublayer(self,sublayer,_feature=False,_layer=False):
        SimpleLayer.add_sublayer(self,sublayer,_feature)
        if _layer:
            self.layers.append(sublayer)
        return sublayer
    

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
    def __init__(self,combine_op="combine",**kwargs):
        CompositeLayer.__init__(self,**kwargs)
        self.combine_op=combine_op           #Type of combining: "combine" (on channel dimension), "combine_batch", "sub", "mult"
        for layer in self._layers:
            self.add_sublayer(Layer(x=self.y,**layer),_layer=True)
        self.combine=SimpleCombine([layer.get() for layer in self.layers],combine_op=self.combine_op)
        self.y=self.combine.y
    def save(self,**kwargs):
        kwargs["combine_op"]=self.combine_op
        return Composite.save(self,**kwargs)

class Network(Composite):#Similar to SimpleLayer, but proper layers are distinguished from feature layers
    type="Network"
    def __init__(self,**kwargs):
        layers=[]
        Composite.__init__(self,**kwargs)
        for layer in self._layers:
            self.add_sublayer(Layer(x=self.y,**layer),_layer=True)
            self.y=self.layers[-1].get()


###Complex Layers
class RandomCombineLayer(CombineLayer):
    type="Random_Combined"
    def __init__(self,channels=1,**kwargs):
        layers=[{"Type":"Identity"},kwargs.update({"Type":"Random","shape":[-1 for i in range(x.get_shape().ndims-1)]+[channels]})]
        CombineLayer.__init__(self,layers=layers)
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
        
                  