from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os,time
from nnBuilder import *
from nnDreamer import *
from dataProducer import *

def composite(imgs):
    return np.concatenate([np.concatenate([img for img in _imgs],axis=1) for _imgs in imgs],axis=0)

class LapGAN:
    def __init__(self,x,sess,dic):
        self.setup(sess,dic)
        self.x=x                 
        self.rand_layer=RandomLayer(self.x,{"type":"random_layer","shape":"x_shape","channels":self.channels})
        self.generator=ConvNet(self.rand_layer.y,self.gen_def)
        
        self.diff=self.generator.y-tf.nn.conv2d(self.generator.y, k5x5, [1,1,1,1], 'SAME')
        self.y=tf.add(self.x[:,self.size_diff//2:-self.size_diff//2,self.size_diff//2:-self.size_diff//2,:],self.diff)
        
        self.class_mixer=CombineLayer(self.y,{"combine_op":"combine_batch",
                "layers":[{"type":"identity"},{"type":"batchProducer","size":self.size,"batch":self.batch//2,
                "producer":{'type': 'sampler_data','file': self.img_file,"dataShape":self.img_shape}}]})
        
        self.classifier=ConvNet(self.class_mixer.y,self.class_def)
        
        self.logits=self.classifier.y
        self.labels=np.array([[0]]*(self.batch//2)+[[1]]*(self.batch//2),dtype=np.float32)
        
        self.class_trainer=ClassifierArrayTrainer(self.classifier,self.labels,self.logits,sigmoid=True)
        self.gen_trainer=ClassifierArrayTrainer(self.generator,(1-self.labels),self.logits,sigmoid=True)
        
        self.gen_saver=NetworkSaver(self.generator)
        self.class_saver=NetworkSaver(self.classifier)
        
        sess.add([self.rand_layer,self.generator,self.class_mixer,self.classifier,self.class_trainer,self.gen_trainer,
                 self.gen_saver,self.class_saver])
        
    def setup(self,sess,dic):
        self.sess=sess                          #The blurred images, input for the generator
        self.img_file=dic["img_file"]                      #The file containing the unblurred image
        self.img_shape=dic["img_shape"]                    #The shape of the unblurred image
        self.blur_file=dic["blur_file"]                    #The file containing the blurred image (optional?)
        self.blur_shape=dic["blur_shape"]                  #The shape of the blurred image (optional?)
        self.noise=defaultinDict(dic,"noise",None)         #Noise added to the blurred image
        self.size=dic["size"]                              #Generated image size for training
        self.batch=dic["batch"]                            #Batch size (assumes x has batch=self.batch//2)
        self.channels=defaultinDict(dic,"channels",1)      #Number of random noise channels fed to the generator
        self.gen_def=dic["gen_def"]                        #Definition for the generator ConvNet
        self.class_def=dic["class_def"]                    #Definition for the classifier ConvNet
        self.size_diff=(np.array(dic["gen_def"]["windows"])-1).sum() #Extra size needed for the input
        
        self.gen_def["input_channels"]=self.img_shape[2]+self.channels
        
    def train(self,n,k=5,dt=.01):
        for i in range(n):
            self.class_trainer.train(k)
            time.sleep(dt)
            self.gen_trainer.train(1)
            time.sleep(dt)
    def train_class(self,n):
        self.class_trainer.train(n)
    def evaluate(self,n=10):
        return self.class_trainer.evaluate(n)
    def save(self,folder):
        fixDir(folder)
        self.class_saver.save(folder+"classifier",safe=False)
        self.gen_saver.save(folder+"generator",safe=False)
    def load(self,folder):
        self.class_saver.load(folder+"classifier")
        self.gen_saver.load(folder+"generator")
    def reset(self):
        self.class_saver.init()
        self.gen_saver.init()
    def compare(self):
        img=np.load(self.blur_file+".npy")[:500,:500]
        img2=np.load(self.img_file+".npy")[self.size_diff//2:500-self.size_diff//2,self.size_diff//2:500-self.size_diff//2]
        y=self.sess.sess.run(self.y,{self.x:img[None,:,:,None]})[0,:,:,0]
        print((img2-y).std())
        showarray(composite([[img2,y]]))
    def sample(self):
        img=np.load(self.blur_file+".npy")[:500,:500]
        diff,y=self.sess.sess.run([self.diff,self.y],{self.x:img[None,:,:,None]})
        showarray(composite([[diff[0,:,:,0],y[0,:,:,0]]]))
        return diff,y
        
class LapGAN_top(LapGAN):
    def __init__(self,sess,dic):
        self.setup(sess,dic)
        self.gen_input=BatchProducerLayer(None,{"size":self.size+self.size_diff,"batch":self.batch//2,
                  "producer":{'type': 'sampler_data','file': self.blur_file,"dataShape":self.blur_shape,"noise":self.noise}})
        
        sess.add([self.gen_input])
        LapGAN.__init__(self,self.gen_input.y,sess,dic)
    
    
        

