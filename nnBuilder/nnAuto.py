import numpy as np
import tensorflow as tf
import warnings,numbers,os,sys
from nnLayer import *
from nnInput import *
from nnUtils import *




class NormalLearned(SimpleLayer):
    type="Normal_Learned"
    def init(self,kwargs):
        super().init(kwargs)
        self.mean_def=kwargs.pop("mean")                        #The mean layer
        self.log_std_def=kwargs.pop("log_std",self.mean_def)    #Layer or number
        self.output=kwargs.pop("output","Sample")               #"Sample" or "Mean"
        if not isinstance(self.log_std_def,numbers.Number):
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="log_std_layer",set_y=False,**self.log_std_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="mean_layer",set_y=False,**self.mean_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,type="Random",set_y=True,x="mean_layer",attr="rand_layer")
    def call(self):
        super().call()
        self.mean=self.mean_layer.get()
        self.log_std=self.log_std_layer.get() if "log_std_layer" in dir(self) else self.log_std_def
        #self.log_std="log_std_layer" in dir(self) and self.log_std_layer.get() or self.log_std_def
        if self.output=="Sample":
            self.y=self.mean+self.y*tf.exp(self.log_std)
        elif self.output=="Mean":
            self.y=self.mean
        else:
            assert(False)
        self.shape=self.y.get_shape().as_list()
    def loss_kl(self,reduce=True):#KL loss with respect to a normal distribution of mean 0 and std 1
        axes=list(range(1,len(self.shape)))
        loss=0.5*tf.reduce_sum(tf.square(self.mean) + tf.exp(self.log_std) - self.log_std - 1,axis=axes)
        if reduce:
            loss=tf.reduce_mean(loss)
        return loss
    def loss_reconstruction(self,x,reduce=True,log_std=None):
        axes=list(range(1,len(self.shape)))
        reg=1e-8
        log_std=log_std or self.log_std #Overwrite std for learning
        #norm_x=tf.square((x-self.mean)/(tf.exp(log_std)))/2
        #log_det=np.log(np.sqrt(np.pi*2))+log_std
        norm_x=tf.square((x-self.mean)/(tf.exp(log_std)+reg))/2
        log_det=np.log(np.sqrt(np.pi*2))+log_std
        loss=tf.reduce_sum(norm_x+log_det,axis=axes)
        if reduce:
            loss=tf.reduce_mean(loss)
        return loss
    
class BinaryLearned(SimpleLayer):#Not fully implemented
    type="Binary_Learned"
    def init(self,kwargs):
        super().init(kwargs)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="prob_layer",type="Sigmoid_Feature")
        self.output=kwargs.pop("output","Mean") #"Sample" or "Mean"
    def call(self):
        super().call()
        self.prob=self.prob_layer.get()
        if self.output=="Sample":
            assert(False),"Not implemented"
        elif self.output=="Mean":
            self.y=self.prob
        else:
            assert(False)
        self.shape=self.y.get_shape().as_list()
    def loss_kl(self,reduce=True):
        assert(False),"Not implemented"
    def loss_reconstruction(self,x,reduce=True):
        axes=list(range(1,len(self.shape)))
        loss=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.prob_layer.x,x),axis=axes)
        if reduce:
            loss=tf.reduce_mean(loss)
        return loss

def _lst_to_net(x):
    if type(x)==list:
        x=dict(type="Network",layers=x)
    return x
    

class Autoencoder(SimpleLayer):
    type="Autoencoder"
    def init(self,kwargs):
        super().init(kwargs)
        self.encoder_def=_lst_to_net(kwargs.pop("encoder"))    #Encoder definition
        self.decoder_def=_lst_to_net(kwargs.pop("decoder"))    #Decoder definition
        self.variational=kwargs.pop("variational",False)    #Variational autoencoder (needs learned encoder/decoder)
        
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="encoder",**self.encoder_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="decoder",**self.decoder_def)
    def call(self):
        super().call()
        self.in_shape=self.encoder.x.get_shape().as_list()
        #assert(self.in_shape==self.decoder.get().get_shape().as_list())
        self.in_shape[0]=None
        self.out_shape=self.encoder.get().get_shape().as_list()
        self.out_shape[0]=None
        self.y=self.encoder.get()
    def loss(self,reduce=True,**kwargs):
        if "loss_reconstruction" in dir(self.decoder):
            loss=self.decoder.loss_reconstruction(x=self.encoder.x,reduce=False,**kwargs)
        else:
            axes=list(range(1,len(self.in_shape)))
            loss=0.5*tf.reduce_sum(tf.square(self.decoder.get()-self.encoder.x),axis=axes)
        if self.variational:
            assert("loss_kl" in dir(self.encoder))
            loss=loss+self.encoder.loss_kl(False)
        if reduce:
            loss=tf.reduce_mean(loss)
        return loss
    def make_encoder(self):
        if "encoder_e" not in dir(self):
            self.encoder_input=Layer(type="Input",shape=self.in_shape)
            self.encoder_e=self.encoder.copy(x=self.encoder_input)
    def make_decoder(self):
        if "decoder_d" not in dir(self):
            self.decoder_input=Layer(type="Input",shape=self.out_shape)
            self.decoder_d=self.decoder.copy(x=self.decoder_input)
    def make_decoder_full(self):
        self.make_encoder()
        if "decoder_f" not in dir(self):
            self.decoder_f=self.decoder.copy(x=self.encoder_e)
    def encode(self,x):
        assert(self.sess!=None)
        self.make_encoder()
        return self.sess.run(self.encoder_e,feed_dict={self.encoder_input.get():x})
    def decode(self,x):
        assert(self.sess!=None)
        self.make_decoder()
        return self.sess.run(self.decoder_d,feed_dict={self.decoder_input.get():x})
    def reconstruct(self,x):
        assert(self.sess!=None)
        self.make_decoder_full()
        return self.sess.run(self.decoder_f,feed_dict={self.encoder_input.get():x})
    def generate(self,n=1):
        self.make_decoder()
        x=np.random.normal(size=[n]+self.out_shape[1:])
        return self.sess.run(self.decoder_d,feed_dict={self.decoder_input.get():x})






























        