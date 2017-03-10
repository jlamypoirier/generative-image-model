import numpy as np
import tensorflow as tf
import os, gc
import warnings
#from nnLayer import *
import nnLayer


#Handles the session stuff
class SessManager:
    reg=[]
    default=None
    @staticmethod
    def register(networks):
        if type(networks)==list:
            SessManager.reg=SessManager.reg+networks
        else:
            SessManager.reg.append(networks)
        if SessManager.default!=None:
            SessManager.default.add(networks)
    def __init__(self,default=True,start=True,*args):
        self.running=False
        self.networks=[]
        if default:
            if SessManager.default!=None:
                warnings.warn("Overwriting default session manager")
            SessManager.default=self
            self.add(SessManager.reg)
        self.add(*args)  #The managed networks or lists of networks, a set of layers or anything with start and stop functions
        self.coord = tf.train.Coordinator()
        if start:
            self.start()
    def add(self,*args):
        #assert(not self.running)
        for networks in args:
            if type(networks)!=list:
                networks=[networks]
            self.networks+=networks
            if self.running:
                for network in networks:
                    network.start(self)
    def start(self):
        assert(not self.running)
        #print("Starting new session")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        #config.gpu_options.per_process_gpu_memory_fraction=0.
        self.sess=tf.InteractiveSession(config=config)
        tf.global_variables_initializer().run()
        self.threads = tf.train.start_queue_runners(sess=self.sess,coord=self.coord, start=True)
        self.running=True
        for network in self.networks:
            network.start(self)
        return self.sess
    def run(self,fetches,*args,**kwargs):#Can take layers instead of tensors (does not support arbitrary nesting)
        self.maybe_start()
        try:
            for i,fetch in enumerate(fetches):
                if isinstance(fetch,SimpleLayer):
                    fetches[i]=fetch.get()
        except:
            if isinstance(fetches,nnLayer.SimpleLayer):
                fetches=fetches.get()
        return self.sess.run(fetches,*args,**kwargs)
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
    def clean(self):#Destroys everything, managed or not
        global resize
        self.stop()
        tf.reset_default_graph()
        gc.collect()



#Handles the saving and loading the variables
class NetworkSaver:
    def __init__(self,network):
        self.network=network                 #Any Layer
        self.save_dict={}
        self.vars=network.get_vars()
        for i,var in enumerate(self.vars):
            self.save_dict["var_%i"%i]=var
        self.saver = tf.train.Saver(self.save_dict,max_to_keep=10000)
        self.reset_op=tf.variables_initializer(self.vars)
        self.sess=None
        SessManager.register(self)
    def save(self,folder="",file=None,safe=True):
        assert(self.sess!=None)
        if folder!="":
            fixDir(folder)
        File=os.path.join(folder,file or "vars.ckpt")
        if safe:
            assert(not os.path.isfile(File))
        print(File)
        return self.saver.save(self.sess,os.path.abspath(File),write_meta_graph=False)
    def start(self,sess):
        self.sess=sess
    def stop(self):
        self.sess=None
    def load(self,folder):
        assert(self.sess!=None)
        return self.saver.restore(self.sess,folder+"/vars.ckpt")
    def init(self):
        self.sess.run(self.reset_op)

