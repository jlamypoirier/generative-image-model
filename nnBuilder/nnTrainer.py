import numpy as np
import tensorflow as tf
import os, warnings
from nnLayer import *


#The basic trainer class, with a training operation and a set of training parameters
class Trainer:
    type="Abstract"
    _trainOpError="Training operation not defined"
    _testError="Can't train on a tester"
    _paramError="Invalid parameter: %s"
    _sessError="No active session"
    def __init__(self,test=False):
        self.test=test           #Dummy trainer to evaluate on test data
        self.params={}           #Maps the parameter tensors to their value
        self.param_dic={}        #Holds the tensor associated to each named parameter
        self.sess=None
    def finish(self):
        self._finish()
        SessManager.register(self)
    def _finish(self):
        return
    def start(self,sess):
        assert("train_op" in dir(self) or self.test), self._trainOpError
        self.sess=sess
    def stop(self):
        self.sess=None
    def ensure_running(self):
        assert(self.sess!=None), self._sessError
    def update_params(self,**kwargs):
        for kw, arg in kwargs.items():
            assert(kw in self.param_dic), self._paramError%kw
            params=self.param_dic[kw]
            if type(params)!=list:
                params=[params]
            for param in params:
                self.params[param]=arg
    def train(self,n=1,**kwargs):
        assert(not self.test)
        self.ensure_running()
        self.update_params(**kwargs)
        for i in range(n):
            self.sess.run(self.train_op,feed_dict=self.params)

class _Optimizer_SGD:
    def __init__(self):
        self.learn_rate=tf.placeholder(tf.float32,shape=[])
    def get_params(self):
        return {self.learn_rate:0.001}
    def get_param_dic(self):
        return {"learn rate":self.learn_rate,"learn_rate":self.learn_rate}
    def build(self,vars,full_loss):
        return tf.train.GradientDescentOptimizer(self.learn_rate).minimize(full_loss,var_list=vars)
class _Optimizer_Momentum(_Optimizer_SGD):
    def __init__(self):
        _Optimizer_SGD.__init__(self)
        self.momentum=tf.placeholder_with_default(np.float32(0.9),shape=[])
        #tf.placeholder(tf.float32,shape=[])
    def get_params(self):
        params=_Optimizer_SGD.get_params(self)
        params.update({self.momentum:0.9})
        return params
    def get_param_dic(self):
        param_dic=_Optimizer_SGD.get_param_dic(self)
        param_dic.update({"momentum":self.momentum})
        return param_dic
    def build(self,vars,full_loss):
        return tf.train.MomentumOptimizer(self.learn_rate,self.momentum).minimize(full_loss,var_list=vars)        
class _Optimizer_Adam(_Optimizer_Momentum):
    def __init__(self):
        _Optimizer_Momentum.__init__(self)
        self.momentum2=tf.placeholder_with_default(np.float32(0.999),shape=[])
        #self.momentum2=tf.placeholder(tf.float32,shape=[])
    def get_params(self):
        params=_Optimizer_Momentum.get_params(self)
        params.update({self.momentum2:0.999})
        return params
    def get_param_dic(self):
        param_dic=_Optimizer_Momentum.get_param_dic(self)
        param_dic.update({"beta1":self.momentum,
                "momentum2":self.momentum2,"beta2":self.momentum2})
        return param_dic
    def build(self,vars,full_loss):
        return tf.train.AdamOptimizer(self.learn_rate,self.momentum,self.momentum2).minimize(full_loss,var_list=vars)      
    
class Optimizer(Trainer):
    type="Abstract"
    _optimizerTypeError="Invalid optimizer type: %s"
    _noLossError="Can't finish the trainer: no loss function given"
    _noNetworkError="Can't finish the trainer: no network given"
    def __init__(self,network=None,loss=None,optimizer="adam",finish=None,regularize=True,dropout=True,test=False,**kwargs):
        Trainer.__init__(self,test=test)
        self.network=network         #The trained network
        self.dropout=dropout         #Enables dropout (requires layers with dropout)
        self.regularize=regularize   #Enable or disable l2 regularization
        self._loss=loss              #The loss tensor, before regularization
        self.loss=loss
        if not self.test:
            if optimizer=="sgd":
                self.optimizer=_Optimizer_SGD()
            elif optimizer=="momentum":
                self.optimizer=_Optimizer_Momentum()
            elif optimizer=="adam":
                self.optimizer=_Optimizer_Adam()
            else:
                raise Exception(self._optimizerTypeError%optimizer)
            self.params.update(self.optimizer.get_params())
            self.param_dic.update(self.optimizer.get_param_dic())
            if regularize:
                self.l2reg=tf.placeholder(tf.float32)
                self.params[self.l2reg]=1e-8
                self.param_dic["l2reg"]=self.l2reg
            self.update_params(**kwargs)
        self.sess=None
        if finish or (finish==None and self.network!=None and self.loss!=None):
            self.finish()
    def _finish(self,network=None,loss=None):
        if loss!=None:             #Use the parameters from the constructor if none given
            self._loss=loss
            self.loss=loss
        if self.loss==None:        #Explodes if the constructor also had none given
            raise Exception(self._noLossError)
        if network!=None:
            self.network=network
        if self.network==None: 
            raise Exception(self._noNetworkError)
        if self.test:
            return
        self.vars=self.network.get_vars()
        if self.dropout:
            self.keep_rates=self.network.get_dropout()
            self.params.update({rate:1. for rate in self.keep_rates})
            self.param_dic.update({"keep_rate":self.keep_rates,"keep rate":self.keep_rates})
            self.param_dic.update({"keep_rate%i":rate for i, rate in enumerate(self.keep_rates)})
            self.param_dic.update({"keep rate%i":rate for i, rate in enumerate(self.keep_rates)})
        if self.regularize:
            self.tf_l2reg=tf.mul(tf.reduce_sum(tf.pack([tf.nn.l2_loss(var) for var in self.vars])),self.l2reg)
            self.loss=tf.add(self.loss,self.l2reg)
        self.train_op=self.optimizer.build(self.vars,self.loss)
    def eval_loss(self,n=10,show=True):
        self.ensure_running()
        loss=np.mean([self.sess.run(self._loss,feed_dict=self.params) for k in range(n)]) #Use loss before l2 regularization
        if show:
            print(loss)
        return (loss)

class LabeledTrainer(Optimizer):
    type="Labeled"
    def __init__(self,labels=None,logits=None,loss="cross_entropy",array=False,**kwargs):
        self.labels=labels
        self.logits=logits
        self._loss_type=loss
        self.loss_type=loss
        self.array=array
        if "finish" not in kwargs or kwargs["finish"]==None:
            if ("network" in kwargs and kwargs["network"]!=None) or (self.labels!=None and self.logits!=None):
                kwargs["finish"]=True
        super().__init__(**kwargs)
    def _finish(self,network=None,loss=None,labels=None,logits=None):
        assert(not(loss or self.loss)), self._lossError
        if network!=None:
            self.network=network
        if self.network==None: 
            raise Exception(self._noNetworkError)
        if labels!=None:
            self.labels=labels
        if self.labels==None:
            self.labels=self.network.get_labels()
        if self.labels==None: 
            raise Exception("Can't find labels")
        if isinstance(self.labels,SimpleLayer):
            self.labels=self.labels.get()
        if logits!=None:
            self.logits=logits
        if self.logits==None:
            self.logits=self.network.get()
        if isinstance(self.logits,SimpleLayer):
            self.logits=self.logits.get()
        if self.array: #Fixes the dimension of the logits when the classifier results in an array (Equiv to avg pooling layer)
            array_dim= self.logits.get_shape().ndims-2  #logits is 2d tensor
            if array_dim>0:
                self.logits=tf.reduce_mean(self.logits,axis=np.array(range(1,array_dim+1)))
        if self.loss_type=="cross_entropy":
            self.loss_type=self.logits.get_shape()[-1].value==1 and "sigmoid" or "softmax"
        if self.loss_type=="sigmoid":
            loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits,self.labels))
        elif self.loss_type=="softmax":
            loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits,self.labels))
        elif self.loss_type=="mean_squared_error":
            loss=tf.reduce_mean(tf.square(self.logits-self.labels))
        elif self.loss_type=="network":
            assert("loss" in dir(self.network))
            loss=self.network.loss
        else:
            raise Exception("Unknown loss type: %s"%self.loss_type)
        super()._finish(loss=loss)
    

class ClassifierTrainer(LabeledTrainer):
    type="Classifier"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def _finish(self,**kwargs):
        super()._finish(**kwargs)
        if self.loss_type=="sigmoid":
            self.sigmoidLayer=SigmoidFeature(x=self.logits)
            self.y=self.sigmoidLayer.get()
            self.wrong_prediction = tf.not_equal(tf.greater(self.y, 0.5), tf.greater(self.labels, 0.5))
        elif self.loss_type=="softmax":
            self.softmaxLayer=SoftmaxFeature(x=self.logits)
            self.y=self.softmaxLayer.get()
            self.wrong_prediction = tf.not_equal(tf.argmax(self.y, 1), tf.argmax(self.labels, 1))
        else:
            raise Exception("Wrong loss type for classifier: %s"%self.loss_type)
        self.error_rate = tf.reduce_mean(tf.cast(self.wrong_prediction, tf.float32))
    def eval_error(self,n=None,show=True,show_loss=False, info=None):
        self.ensure_running()
        if n==None:
            n=self.test and 1 or 10
        e=np.array([self.sess.run([self.loss,self.error_rate],feed_dict=self.params) for k in range(n)])
        loss=np.mean(e[:,0])
        error=np.mean(e[:,1])
        if info==None:
            info=self.test and "test" or "train"
        if show_loss:
            print("Loss (%s): %s"%(info,loss))
        if show:
            print("Error rate (%s): %s"%(info,error))
        return (loss,error)
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    