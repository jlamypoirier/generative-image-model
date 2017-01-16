import numpy as np
import tensorflow as tf
import os, gc
from nnLayer import *

class NetworkSaver:
    def __init__(self,network):
        self.network=network                 #Any Layer
        self.saveDict={}
        for i,var in enumerate(network.getVars()):
            self.saveDict["var_%i"%i]=var
        self.saver = tf.train.Saver(self.saveDict,max_to_keep=10000)
        self.reset_op=tf.variables_initializer(network.getVars())
        self.sess=None
    def save(self,folder="",file=None,safe=True):
        assert(self.sess!=None)
        if folder!="":
            fixDir(folder)
        File=os.path.join(folder,file or "vars.ckpt")
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
        self.networks=networks               #The managed networks, a set of layers or anything with start and stop functions
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
            network.start(self,self.sess)
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
    def maybe_stop(self):
        if self.running:
            self.stop()
    def maybe_start(self):
        if not self.running:
            self.start()
    def get(self):
        self.maybe_start()
        return self.sess
    def clean(self):#Destroy everything, managed or not
        global resize
        self.stop()
        tf.reset_default_graph()
        gc.collect()
        
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
    