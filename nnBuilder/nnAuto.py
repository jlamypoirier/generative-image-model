import numpy as np
import tensorflow as tf
import warnings
from nnLayer import *
from nnInput import *
from _nnUtils import *
import os
import sys








class Autoencoder(SimpleLayer):
    type="Autoencoder"
    def init(self,kwargs):
        super().init(kwargs)
        self.encoder_def=kwargs.pop("encoder")
        self.decoder_def=kwargs.pop("decoder")
        if type(self.encoder_def)==list:
            self.encoder_def=dict(type="Network",layers=self.encoder_def)
        if type(self.decoder_def)==list:
            self.decoder_def=dict(type="Network",layers=self.decoder_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,type="Identity_Label")
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="encoder",**self.encoder_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="decoder",**self.decoder_def)
    def call(self):
        super().call()
        self.in_shape=self.encoder.x.get_shape().as_list()
        self.in_shape[0]=None
        self.shape=self.encoder.get().get_shape().as_list()
        self.shape[0]=None
    def make_encoder(self):
        if "encoder_encode" not in dir(self):
            self.encoder_input=Layer(type="Input",shape=self.in_shape)
            self.encoder_encode=self.encoder.copy(x=self.encoder_input)
    def make_decoder(self):
        if "decoder_decode" not in dir(self):
            self.decoder_input=Layer(type="Input",shape=self.shape)
            self.decoder_decode=self.decoder.copy(x=self.decoder_input)
    def make_decoder_full(self):
        self.make_encoder()
        if "decoder_full" not in dir(self):
            self.decoder_full=self.decoder.copy(x=self.encoder_encode)
    def encode(self,x):
        assert(self.sess!=None)
        self.make_encoder()
        return self.sess.run(self.encoder_encode,feed_dict={self.encoder_input:x})
    def decode(self,x):
        assert(self.sess!=None)
        self.make_decoder()
        return self.sess.run(self.decoder_decode,feed_dict={self.decoder_input:x})
    def run_x(self,x):
        assert(self.sess!=None)
        self.make_decoder_full()
        return self.sess.run(self.decoder_full,feed_dict={self.encoder_input:x})

class _VariationalAutoencoderInput(SimpleLayer):
    type="Variational_Autoencoder_Input"
    def init(self,kwargs):
        super().init(kwargs)
        self.mean_def=kwargs.pop("mean")
        self.log_std_def=kwargs.pop("log_std")
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="log_std",set_y=False,**self.log_std_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="mean",set_y=False,**self.mean_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,type="Random",set_y=True,x="mean")
    def call(self):
        super().call()
        self.y=self.mean.get()+self.y*tf.exp(self.log_std.get())
    
    


class VariationalAutoencoder(SimpleLayer):
    type="Variational_Autoencoder"
    def init(self,kwargs):
        super().init(kwargs)
        self.encoder_def=kwargs.pop("encoder")
        self.decoder_def=kwargs.pop("decoder")
        if type(self.encoder_def)==list:
            self.encoder_def=dict(type="Network",layers=self.encoder_def)
        if type(self.decoder_def)==list:
            self.decoder_def=dict(type="Network",layers=self.decoder_def)
        self.mean_def=kwargs.pop("mean")
        self.log_std_def=kwargs.pop("log_std",self.mean_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,type="Identity_Label")
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="encoder",**self.encoder_def)
        #self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="std",set_y=False,**self.std_def)
        #self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="mean",set_y=False,**self.mean_def)
        #self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,type="Random",x="mean")
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,type="Variational_Autoencoder_Input",attr="z",
                              mean=self.mean_def,log_std=self.log_std_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="decoder",**self.decoder_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="sigmoid",type="Sigmoid_Feature")
    def call(self):
        super().call()
        self.in_shape=self.encoder.x.get_shape().as_list()
        self.in_shape[0]=None
        self.shape=self.z.get().get_shape().as_list()
        self.shape[0]=None
        #self.loss_reconstruction=tf.reduce_mean(tf.square(self.y-self._x))
        self.loss_reconstruction=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.decoder.get(),self.x))
        self.loss_kl=0.5*tf.reduce_sum(tf.square(self.z.mean.get()) + tf.exp(self.z.log_std.get()) - self.z.log_std.get() - 1)
        self.loss=(self.loss_reconstruction+self.loss_kl)/tf.cast(tf.reduce_prod(tf.shape(self.x)),tf.float32)
        #self.loss_reconstruction=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.decoder.get(),self._x))
        #self.loss_kl=0.5*tf.reduce_mean(tf.square(self.z.mean.get()) + tf.exp(self.z.log_std.get()) - self.z.log_std.get() - 1)
        #self.loss=self.loss_reconstruction
    def make_encoder(self):
        if "z_encode" not in dir(self):
            self.encoder_input=Layer(type="Input",shape=self.in_shape)#Bad
            self.encoder_encode=self.encoder.copy(x=self.encoder_input)
            self.z_encode=self.z.copy(x=self.encoder_encode)
    def make_decoder(self):
        if "decoder_decode" not in dir(self):
            self.decoder_input=Layer(type="Input",shape=self.shape)
            self.decoder_decode=self.decoder.copy(x=self.decoder_input)
            self.y_decoder=self.sigmoid.copy(x=self.decoder_decode)
    def make_decoder_full(self):
        self.make_encoder()
        if "decoder_full" not in dir(self):
            self.decoder_full=self.decoder.copy(x=self.z_encode)
            self.y_decoder_full=self.sigmoid.copy(x=self.decoder_full)
    def make_generator(self):
        if "y_generator" not in dir(self):
            self.generator_input=Layer(type="Variational_Autoencoder_Input",#Use numpy random instead?
                                       mean=dict(type="Input",shape=self.shape),
                                       log_std=dict(type="Input",shape=self.shape))
            self.decoder_generator=self.decoder.copy(x=self.generator_input)
            self.y_generator=self.sigmoid.copy(x=self.decoder_generator)
    def encode(self,x):
        assert(self.sess!=None)
        self.make_encoder()
        return self.sess.run([self.z.mean,self.z.log_std],feed_dict={self.encoder_input.get():x})
    def decode(self,x):
        assert(self.sess!=None)
        self.make_decoder()
        return self.sess.run(self.y_decoder,feed_dict={self.decoder_input.get():x})
    def generate(self,mean=None,log_std=None,n=1):
        assert(self.sess!=None)
        self.make_generator()
        if mean==None:
            if log_std==None:
                mean=np.zeros(shape=[n]+self.shape[1:], dtype=np.float32)
            else:
                mean=np.zeros_like(log_std)
        if log_std==None:
            log_std=np.ones_like(mean)
        return self.sess.run(self.y_generator,feed_dict={self.generator_input.mean.get():mean,
                                                               self.generator_input.log_std.get():log_std})
    def run_x(self,x):
        assert(self.sess!=None)
        self.make_decoder_full()
        return self.sess.run(self.y_decoder_full,feed_dict={self.encoder_input.get():x})


































        