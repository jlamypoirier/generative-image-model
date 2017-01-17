import numpy as np
import tensorflow as tf
import os, gc
from nnLayer import *


#The basic trainer class, with a training operation and a set of training parameters
class Trainer:
    type="Abstract"
    def __init__(self):
        self.sess=None
        self.params={}           #Maps the parameter tensors to their value
        self.param_dic={}        #Holds the tensor associated to each named parameter
    def start(self,sess):
        assert("train_op" in dir(self))
        self.sess=sess
    def stop(self):
        self.sess=None
    def update_params(**kwargs):
        for kw, arg in kwargs.items():
            assert(kw in self.param_dic)
            self.params[self.param_dic[kw]]=arg
    def train(self,n=1,**kwargs):
        assert(self.sess!=None)
        self.update_params(**kwargs)
        for i in range(n):
            self.sess.run(self.train_op,feed_dict=self.params)

class _Optimizer_SGD:
    def __init__(self):
        self.learn_rate=tf.placeholder(tf.float32)
    def get_params(self):
        return {self.learn_rate:0.001}
    def get_param_dic(self):
        return {"learn rate":self.learn_rate,"learn_rate":self.learn_rate}
    def build(self,vars,full_loss):
        return tf.train.GradientDescentOptimizer(self.learn_rate).minimize(full_loss,var_list=vars)
class _Optimizer_Momentum(_Optimizer_SGD):
    def __init__(self):
        _Optimizer_SGD.__init__(self)
        self.momentum=tf.placeholder(tf.float32)
    def get_params(self):
        return _Optimizer_SGD.get_params(self).update({self.momentum:0.9})
    def get_param_dic(self):
        return _Optimizer_SGD.get_param_dic(self).update({"momentum":self.momentum})
    def build(self,vars,full_loss):
        return tf.train.MomentumOptimizer(self.learn_rate,self.momentum).minimize(full_loss,var_list=vars)        
class _Optimizer_Adam(_Optimizer_Momentum):
    def __init__(self):
        _Optimizer_Momentum.__init__(self)
        self.momentum2=tf.placeholder(tf.float32)
    def get_params(self):
        return _Optimizer_Momentum.get_params(self).update({self.momentum2:0.999})
    def get_param_dic(self):
        return _Optimizer_Momentum.get_param_dic(self).update({"beta1":self.momentum,
                "momentum2":self.momentum2,"beta2":self.momentum2})
    def build(self,vars,full_loss):
        return tf.train.AdamOptimizer(self.learn_rate,self.momentum,self.momentum2).minimize(full_loss,var_list=vars)      
    
class Optimizer(Trainer):
    type="Abstract"
    def __init__(self,network=None,loss=None,optimizer="",finish=None,regularize=True,dropout=True,**kwargs):
        Trainer.__init__(self)
        self.network=network         #The trained network
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
            assert(False)
        self.params.update(optimizer.get_params())
        self.param_dic.update(optimizer.get_param_dic())
        if dropout:
            self.keep_rates=network.getDropout()
            self.params.update({rate:1. for rate in self.keep_rates})
            self.param_dic.update({"keep_rate":self.keep_rates,"keep rate":self.keep_rates})
            self.param_dic.update({"keep_rate%i":rate for i, rate in enumerate(self.keep_rates)})
            self.param_dic.update({"keep rate%i":rate for i, rate in enumerate(self.keep_rates)})
        if regularize:
            self.l2reg=tf.placeholder(tf.float32)
            self.params[self.l2reg]=1e-8
            self.param_dic["l2reg"]=self.l2reg
        self.update_params(**kwargs)
        self.sess=None
        if finish or (finish==None and self.network!=None and self.loss!=None):
            self.finish()
    def finish(self,network=None,loss=None):
        self.loss=loss or self.loss or assert(False)           #Use the parameters from the constructor if none given
        self.network=network or self.network or assert(False)  #Explodes if the constructor also had none given
        self.vars=network.getVars()
        if self.regularize:
            self.tf_l2reg=tf.mul(tf.reduce_sum(tf.pack([tf.nn.l2_loss(var) for var in self.vars])),self.l2reg)
            self.loss=tf.add(self.loss,self.l2reg)
        self.train_op=optimizer.build(self.vars,self.loss)
    def eval_loss(self,n=10,show=True):
        assert(self.sess!=None)
        loss=np.mean([self.sess.run(self._loss) for k in range(n)]) #Use loss before l2 regularization
        if show:
            print(loss)
        return (loss)


        
'''class NetworkTrainer:
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
    