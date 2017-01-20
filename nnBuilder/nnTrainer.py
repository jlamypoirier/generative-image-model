import numpy as np
import tensorflow as tf
import os, warnings
from nnLayer import *


#The basic trainer class, with a training operation and a set of training parameters
class Trainer:
    type="Abstract"
    _trainOpError="Training operation not defined"
    _paramError="Invalid parameter: %s"
    _sessError="No active session"
    def __init__(self):
        self.sess=None
        self.params={}           #Maps the parameter tensors to their value
        self.param_dic={}        #Holds the tensor associated to each named parameter
    def start(self,sess):
        assert("train_op" in dir(self)), self._trainOpError
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
    def __init__(self,network=None,loss=None,optimizer="adam",finish=None,regularize=True,dropout=True,**kwargs):
        Trainer.__init__(self)
        self.network=network         #The trained network
        self.dropout=dropout         #Enables dropout (requires layers with dropout)
        self.regularize=regularize   #Enable or disable l2 regularization
        self._loss=loss              #The loss tensor
        self.loss=loss
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
    def finish(self,network=None,loss=None):
        if loss!=None:             #Use the parameters from the constructor if none given
            self.loss=loss
        if self.loss==None:        #Explodes if the constructor also had none given
            raise Exception(self._noLossError)
        if network!=None:
            self.network=network
        if self.network==None: 
            raise Exception(self._noNetworkError)
        self.vars=self.network.getVars()
        if self.dropout:
            self.keep_rates=self.network.getDropout()
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

class ClassifierTrainer(Optimizer):
    type="Classifier"
    _lossError="Loss function fed to classifier, must feed labels"
    _noLabelsError="Can't finish the trainer: no labels given"
    def __init__(self,labels=None,logits=None,sigmoid=None,array=False,**kwargs):
        self.labels=labels
        self.logits=logits
        self.sigmoid=sigmoid
        self.array=array
        Optimizer.__init__(self,**kwargs)
    def finish(self,network=None,loss=None,labels=None,logits=None):
        assert(not(loss or self.loss)), self._lossError
        if network!=None:
            self.network=network
        if self.network==None: 
            raise Exception(self._noNetworkError)
        if labels!=None:
            self.labels=labels
        if self.labels==None:
            self.labels=self.network.getInputLabels()
        if self.labels==None: 
            raise Exception(self._noLabelsError)
        if logits!=None:
            self.logits=logits
        if self.logits==None:
            self.logits=self.network.get()
        if self.array: #Fixes the dimension of the logits when the classifier results in an array (Equiv to avg pooling layer)
            array_dim= self.logits.get_shape().ndims-2  #logits is 2d tensor
            if array_dim>0:
                self.logits=tf.reduce_mean(self.logits,axis=range(1,array_dim+1))
        if self.sigmoid==None:
            self.sigmoid=self.logits.get_shape()[-1].value==1
        if self.sigmoid:
            self.cross_entropy=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits,self.labels))
        else:
            self.cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits,self.labels))
        if self.sigmoid:
            self.sigmoidLayer=SimpleSigmoid(self.logits)
            self.y=self.sigmoidLayer.get()
            self.wrong_prediction = tf.not_equal(tf.greater(self.y, 0.5), tf.greater(self.labels, 0.5))
        else:
            self.softmaxLayer=SimpleSoftmax(x=self.logits)
            self.y=self.softmaxLayer.get()
            self.wrong_prediction = tf.not_equal(tf.argmax(self.y, 1), tf.argmax(self.labels, 1))
        self.error_rate = tf.reduce_mean(tf.cast(self.wrong_prediction, tf.float32))
        Optimizer.finish(self,loss=self.cross_entropy)
    def eval_error(self,n=10,print_=True,print_loss=False):
        self.ensure_running()
        e=np.array([self.sess.run([self.loss,self.error_rate],feed_dict=self.params) for k in range(n)])
        loss=np.mean(e[:,0])
        error=np.mean(e[:,1])
        if print_loss:
            print("Loss: %s"%loss)
        if print_:
            print("Error rate: %s"%error)
        return (loss,error)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    