from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
from dataProducer import *
from nnBuilder import *


class NetworkTrainer:
    def __init__(self,network,loss,dic):
        assert(network.ready)
        self.network=network
        self.loss=loss
        self.learnRate=dic["learnRate"]
        self.l2regMul=dic["l2regMul"]
        self.momentum=dic["momentum"]
        self.keep_rate=defaultinDict(dic,"keep_rate",1.0)
        self.tf_l2regMul=tf.placeholder(tf.float32)
        self.tf_learnRate=tf.placeholder(tf.float32)
        self.tf_momentum=tf.placeholder(tf.float32)
        self.l2reg=tf.mul(tf.add(tf.reduce_sum(tf.pack([tf.nn.l2_loss(w) for w in network.w])),
                          tf.reduce_sum(tf.pack([tf.nn.l2_loss(b) for b in network.b]))),self.tf_l2regMul)
        self.loss_full=tf.add(self.loss,self.l2reg)
        self.var_list=network.w+network.b
        self.train_step = tf.train.MomentumOptimizer(self.tf_learnRate,self.tf_momentum).minimize(
            self.loss_full,var_list=self.var_list)
    def train(self,n,sess):
        for i in range(n):
            sess.run(self.train_step,feed_dict={self.tf_learnRate:self.learnRate,self.tf_l2regMul:self.l2regMul,
                                                self.tf_momentum:self.momentum,self.network.tf_keep_rate:self.keep_rate})
        
        
    

class GenNetwork:
    def __init__(self,dic):
        self.rand_input=RandLayer(shape=[dic['batch']]+dic["generator"]["start_shape"])
        self.genNetwork=Network(x=self.rand_input.y,softmax=False,dic=dic["generator"])
        
        self.batch=dic['batch']
        self.size=self.genNetwork.y.get_shape()[1].value
        self.sampler=SamplerData(self.size,dic=dic["sampler"])
        self.data_input=tf.train.batch([self.sampler.data], batch_size=self.batch, num_threads=16, capacity=4096)
        self.class_input=tf.concat(0,[self.genNetwork.y,self.data_input])
        
        assert(dic["classifier"]["layers"][-1]["size"]==1)
        #self.classNetwork=Network(x=self.class_input,dic=dic["classifier"])
        #self.labels=tf.constant([[0,1]]*self.batch+[[1,0]]*self.batch,dtype=tf.float32)
        self.classNetwork=Network(x=self.class_input,dic=dic["classifier"],softmax=False)
        print(self.genNetwork.y.get_shape())
        print(self.classNetwork.y.get_shape())
        #self.cropped=tf.slice(self.classNetwork.y,[0,4,4,0],np.array(self.classNetwork.y.get_shape().as_list())-[0,8,8,0], name=None)
        #print(self.cropped.get_shape())
        self.logits=tf.reshape(self.classNetwork.y,[2*self.batch,-1])
        self.y=tf.sigmoid(self.logits)
        print(self.logits.get_shape(),self.logits.get_shape().num_elements())
        n_logits=self.logits.get_shape()[1].value
        self.labels=tf.constant([[0]*n_logits]*self.batch+[[1]*n_logits]*self.batch,dtype=tf.float32)
        self.cross_entropy=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits,self.labels))
        self.cross_entropy_2=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits,1-self.labels))
        self.cross_entropy_0=-tf.reduce_mean(tf.mul(self.logits,2*self.labels-1))
        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits,1-self.labels))
        #self.cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.classNetwork.logits,self.labels))
        self.correct_prediction = tf.equal(tf.greater(self.y, 0.5), tf.greater(self.labels, 0.5))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.error_rate=tf.sub(1.,self.accuracy)
        
        self.genTrainer=NetworkTrainer(self.genNetwork,self.cross_entropy_2,dic["generator"])
        self.classTrainer=NetworkTrainer(self.classNetwork,self.cross_entropy,dic["classifier"])
        self.training_ratio=dic["training_ratio"]
        self.running=False
        
        
        self.evalNetwork=Network(x=self.class_input,dic=dic["evaluator"],softmax=False)
        self.eval_logits=tf.reshape(self.evalNetwork.y,[2*self.batch,-1])
        self.eval_y=tf.sigmoid(self.eval_logits)
        print(self.eval_logits.get_shape(),self.eval_logits.get_shape().num_elements())
        eval_n_logits=self.eval_logits.get_shape()[1].value
        self.eval_labels=tf.constant([[0]*eval_n_logits]*self.batch+[[1]*eval_n_logits]*self.batch,dtype=tf.float32)
        self.eval_cross_entropy=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.eval_logits,self.eval_labels))
        self.eval_correct_prediction = tf.equal(tf.greater(self.eval_y, 0.5), tf.greater(self.eval_labels, 0.5))
        self.eval_accuracy = tf.reduce_mean(tf.cast(self.eval_correct_prediction, tf.float32))
        self.eval_error_rate=tf.sub(1.,self.eval_accuracy)
        self.evalTrainer=NetworkTrainer(self.evalNetwork,self.eval_cross_entropy,dic["evaluator"])
        
    def train(self,n):
        self.ensureSess()
        for i in range(n):
            self.genTrainer.train(self.training_ratio,self.sess)
            self.classTrainer.train(1,self.sess)
    def eval_train(self,n):
        self.ensureSess()
        self.evalTrainer.train(n,self.sess)
    def startSess(self):
        assert(not self.running)
        print("Starting new session")
        self.coord = tf.train.Coordinator()
        self.sess=tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.sampler.start(self.sess)
        self.threads = tf.train.start_queue_runners(coord=self.coord, start=True)
        self.genNetwork.sess=self.sess
        self.classNetwork.sess=self.sess
        self.evalNetwork.sess=self.sess
        self.genNetwork.running=True
        self.classNetwork.running=True
        self.evalNetwork.running=True
        self.running=True
    def endSess(self):
        assert(self.running)
        print("Ending session")
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()
        self.running=False
    def ensureSess(self):
        if not self.running:
            self.startSess()
    def ensureNoSess(self):
        if self.running:
            self.endSess()
    def evaluate(self,n=10,print_=True):
        assert(self.running)
        e=np.array([self.sess.run([self.cross_entropy,self.cross_entropy_2,self.error_rate]) for k in range(n)])
        loss=np.mean(e[:,0])
        loss_2=np.mean(e[:,1])
        error=np.mean(e[:,2])
        if print_:
            print((loss,loss_2,error))
        return (loss,loss_2,error)
    def eval_evaluate(self,n=10,print_=True):
        assert(self.running)
        e=np.array([self.sess.run([self.eval_cross_entropy,self.eval_error_rate]) for k in range(n)])
        loss=np.mean(e[:,0])
        #loss_2=np.mean(e[:,1])
        error=np.mean(e[:,1])
        if print_:
            print((loss,error))
        return (loss,error)
    def generate_full(self):
        return self.sess.run(self.genNetwork.y)
    def generate(self):
        return self.generate_full()[0,:,:,0]
        
        































