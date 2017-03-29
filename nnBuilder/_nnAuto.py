import numpy as np
import tensorflow as tf
import warnings,numbers,os,sys
from nnLayer import *
from nnInput import *
from _nnUtils import *








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

class NormalLearned(SimpleLayer):
    type="Normal_Learned"
    class Constget:
        def __init__(self,x):
            self.x=x
        def get(self):
            return self.x
    def init(self,kwargs):
        super().init(kwargs)
        self.mean_def=kwargs.pop("mean")
        self.log_std_def=kwargs.pop("log_std")
        self.use_mean=kwargs.pop("use_mean",False)
        if isinstance(self.log_std_def,numbers.Number):
            self.log_std=self.Constget(self.log_std_def)
        else:
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="log_std",set_y=False,**self.log_std_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="mean",set_y=False,**self.mean_def)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,type="Random",set_y=True,x="mean")
    def call(self):
        super().call()
        if self.use_mean:
            self.y=self.mean.get()
        else:
            self.y=self.mean.get()+self.y*tf.exp(self.log_std.get())
    
class BinaryLearned(SimpleLayer):#No sampling
    type="Binary_Learned"
    def init(self,kwargs):
        super().init(kwargs)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="prob",type="Sigmoid_Feature")
    def call(self):
        super().call()
def _lst_to_net(x):
    if type(x)==list:
        x=dict(type="Network",layers=x)
    return x
    

class VariationalAutoencoder(SimpleLayer):
    type="Variational_Autoencoder"
    def init(self,kwargs):
        super().init(kwargs)
        self.encoder_def=_lst_to_net(kwargs.pop("encoder"))    #Encoder definition (minus final layer)
        self.decoder_def=_lst_to_net(kwargs.pop("decoder"))    #Decoder definition (minus final layer)
        self.encoder_type=kwargs.pop("encoder_type","Normal") #Use binary instead of normal for the decoder (not implemented)
        self.decoder_type=kwargs.pop("decoder_type","Normal") #Use binary instead of normal for the decoder (sampling not implemented)
        if self.encoder_type!="Binary":
            self.encoder_mean=kwargs.pop("encoder_mean")                           #Mean for the encoded variables
            self.encoder_log_std=kwargs.pop("encoder_log_std",self.encoder_mean)   #Std for the encoded variables
        if self.decoder_type!="Binary":
            self.decoder_mean=kwargs.pop("decoder_mean")                           #Mean for the encoded variables
            self.decoder_log_std=kwargs.pop("decoder_log_std",self.decoder_mean)   #Std for the encoded variables
            
            
            
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,type="Identity_Label")
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="encoder",**self.encoder_def)
        if self.encoder_type=="Binary":
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="ze",type="Binary_Learned")
        else:
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,type="Normal_Learned",attr="ze",
                              mean=self.encoder_mean,log_std=self.encoder_log_std)
        self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="decoder",**self.decoder_def)
        if self.decoder_type=="Binary":
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="zd",type="Binary_Learned")
        else:
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,type="Normal_Learned",attr="zd",
                              mean=self.decoder_mean,log_std=self.decoder_log_std,use_mean=self.decoder_type=="Normal_Deterministic")
    def call(self):
        super().call()
        self.in_shape=self.encoder.x.get_shape().as_list()
        self.in_shape[0]=None
        self.shape=self.ze.get().get_shape().as_list()
        self.shape[0]=None
        #self.loss_reconstruction=tf.reduce_mean(tf.square(self.y-self._x))
        #self.loss_reconstruction=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.decoder.get(),self.x))
        #self.loss_kl=0.5*tf.reduce_sum(tf.square(self.z.mean.get()) + tf.exp(self.z.log_std.get()) - self.z.log_std.get() - 1)
        #self.loss=(self.loss_reconstruction+self.loss_kl)/tf.cast(tf.reduce_prod(tf.shape(self.x)),tf.float32)
        #self.loss_reconstruction=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.decoder.get(),self._x))
        #self.loss_kl=0.5*tf.reduce_mean(tf.square(self.z.mean.get()) + tf.exp(self.z.log_std.get()) - self.z.log_std.get() - 1)
        #self.loss=self.loss_reconstruction
        self.y=self.ze.get()
    def loss_kl(self,reduce=True):
        axes=list(range(1,len(self.shape)))
        if self.encoder_type=="Binary":
            assert(False),"Not implemented"
        else:
            y=0.5*tf.reduce_sum(tf.square(self.ze.mean.get()) + tf.exp(self.ze.log_std.get()) - self.ze.log_std.get() - 1,axis=axes)
        if reduce:
            y=tf.reduce_mean(y)
        return y
    def loss_reconstruction(self,reduce=True):
        axes=list(range(1,len(self.in_shape)))
        if self.decoder_type=="Binary":
            y=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.decoder.get(),self.x),axis=axes)
        else:#Not working well
            #norm_x=(self.x-self.zd.mean.get())/tf.exp(self.zd.log_std.get())
            #x_red=tf.reduce_sum(tf.square(norm_x),axis=axes)
            #log_det=tf.reduce_sum(np.log(np.sqrt(np.pi*2))+self.zd.log_std.get(),axis=axes)
            reg=1e-6
            #self.labels=-tf.log(tf.reciprocal(.999*self.x+.0005)-1.)
            self.norm_x=tf.square((self.x-self.zd.mean.get())/(tf.exp(self.zd.log_std.get())+reg))/2
            self.log_det=np.log(np.sqrt(np.pi*2))+tf.maximum(self.zd.log_std.get(),np.log(reg))
            #self.norm_x=norm_x
            #self.x_red=x_red
            #self.log_det=log_det
            #y=x_red/2+log_det#-tf.log(tf.exp(-x_red/2)/det)
            y=tf.reduce_sum(self.norm_x+self.log_det,axis=axes)
        if reduce:
            y=tf.reduce_mean(y)
        return y
    def loss(self,reduce=True):
        y=self.loss_reconstruction(False)+self.loss_kl(False)
        if reduce:
            y=tf.reduce_mean(y)
        return y
    def make_encoder(self):
        if "ze_encode" not in dir(self):
            self.encoder_input=Layer(type="Input",shape=self.in_shape)#Bad
            self.encoder_encode=self.encoder.copy(x=self.encoder_input)
            self.ze_encoder=self.ze.copy(x=self.encoder_encode)
    def make_decoder(self):
        if "decoder_decode" not in dir(self):
            self.decoder_input=Layer(type="Input",shape=self.shape)
            self.decoder_decode=self.decoder.copy(x=self.decoder_input)
            self.zd_decoder=self.zd.copy(x=self.decoder_decode)
    def make_decoder_full(self):
        self.make_encoder()
        if "decoder_full" not in dir(self):
            self.decoder_full=self.decoder.copy(x=self.ze_encode)
            self.zd_decoder_full=self.zd.copy(x=self.decoder_full)
    def make_generator(self):
        if "y_generator" not in dir(self):
            self.generator_input=Layer(type="Normal_Learned",#Use numpy random instead?
                                       mean=dict(type="Input",shape=self.shape),
                                       log_std=dict(type="Input",shape=self.shape))
            self.decoder_generator=self.decoder.copy(x=self.generator_input)
            self.zd_generator=self.zd.copy(x=self.decoder_generator)
    def encode(self,x):
        assert(self.sess!=None)
        self.make_encoder()
        return self.sess.run(self.ze_encoder,feed_dict={self.encoder_input.get():x})
    def encode_stats(self,x):
        assert(self.sess!=None)
        self.make_encoder()
        stats=self.binary_encoder and self.ze_encoder.prob or [self.ze_encoder.mean,self.ze_encoder.log_std]
        return self.sess.run(stats,feed_dict={self.encoder_input.get():x})
    def decode(self,x):
        assert(self.sess!=None)
        self.make_decoder()
        return self.sess.run(self.zd_decoder,feed_dict={self.decoder_input.get():x})
    def decode_stats(self,x):
        assert(self.sess!=None)
        self.make_decoder()
        stats=self.binary_encoder and self.zd_decoder.prob or [self.zd_decoder.mean,self.zd_decoder.log_std]
        return self.sess.run(stats,feed_dict={self.decoder_input.get():x})
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
        return self.sess.run(self.zd_generator,feed_dict={self.generator_input.mean.get():mean,
                                                               self.generator_input.log_std.get():log_std})
    def run_x(self,x):
        assert(self.sess!=None)
        self.make_decoder_full()
        return self.sess.run(self.zd_decoder_full,feed_dict={self.encoder_input.get():x})
    def run_x_stats(self,x):
        assert(self.sess!=None)
        self.make_decoder_full()
        stats=self.binary_decoder and self.zd_decoder_full.prob or [self.zd_decoder_full.mean,self.zd_decoder_full.log_std]
        return self.sess.run(stats,feed_dict={self.encoder_input.get():x})





class VariationalAutoencoderStacked(SimpleLayer):
    type="Variational_Autoencoder_Stacked"
    def init(self,kwargs):
        super().init(kwargs)
        self.encoders_def=kwargs.pop("encoders")
        self.decoders_def=kwargs.pop("decoders")
        self.encoders_mean=kwargs.pop("encoders_mean")
        self.encoders_log_std=kwargs.pop("encoders_log_std",self.encoders_mean)
        self.decoders_mean=kwargs.pop("decoders_mean")
        self.decoders_log_std=kwargs.pop("decoders_log_std",self.decoders_mean)
        self.decoders_type=kwargs.pop("decoders_type")
        self.n=len(self.encoders_def)
        for i in range(self.n):
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="encoder%i"%i,set_y=True,
                                  **_lst_to_net(self.encoders_def[i]))
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="ze%i"%i,set_y=True,
                                  type="Normal_Learned",mean=self.encoders_mean[i],log_std=self.encoders_log_std[i])
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="decoder%i"%i,set_y=False,
                                  **_lst_to_net(self.decoders_def[i]))
            self.add_sublayer_def(sublayer_type=self.SUBLAYER_PROPER_MANAGED,attr="zd%i"%i,set_y=False,x="decoder%i"%i,
                                  type="Normal_Learned",mean=self.decoders_mean[i],log_std=self.decoders_log_std[i],
                                  use_mean=self.decoders_type[i]=="Normal_Deterministic")
        
        
        
        
    def call(self):
        #self.encoders=[]
        #self.ze=[]
        #self.decoders=[]
        #self.decoders_stacked=[]
        #self.zd=[]
        #self.zd_stacked=[]
        super().call()
        self.encoders=[getattr(self,"encoder%i"%i) for i in range(self.n)]
        self.ze=[getattr(self,"ze%i"%i) for i in range(self.n)]
        self.decoders=[getattr(self,"decoder%i"%i) for i in range(self.n)]
        self.zd=[getattr(self,"zd%i"%i) for i in range(self.n)]
        self.shapes=[self.x.get_shape().as_list()]+[self.ze[i].get().get_shape().as_list() for i in range(self.n)]
        for shape in self.shapes:
            shape[0]=None
        
        
        '''for i in range(self.n):
            self.add_sublayer(Layer(x=self.y,**_lst_to_net(self.encoders_def[i])),attr="encoders",set_y=True)
            self.add_sublayer(Layer(x=self.y,type="Normal_Learned",mean=self.encoders_mean[i],log_std=self.encoders_log_std[i])
                              ,attr="ze",set_y=True)
            self.add_sublayer(Layer(x=self.y,**_lst_to_net(self.decoders_def[i])),attr="decoders",set_y=False)
            self.add_sublayer(Layer(x=self.decoders[-1],type="Normal_Learned",mean=self.decoders_mean[i],log_std=self.decoders_log_std[i],
                                   use_mean=self.decoders_type[i]=="Normal_Deterministic")
                              ,attr="zd",set_y=False)
            self.shapes.append(self.ze[i].get().get_shape().as_list())
            self.shapes[-1][0]=None'''
        for i in range(self.n-1,-1,-1):
            self.add_sublayer(self.decoders[i].copy(x=self.y),attr="decoders_stacked",set_y=True)
            self.add_sublayer(self.zd[i].copy(x=self.y),attr="zd_stacked",set_y=True)
        self.decoders_stacked.reverse()
        self.zd_stacked.reverse()
        self.y=self.ze[-1].get()
    def loss_kl(self,i=None,reduce=True):
        if i==None:
            return sum([self.loss_kl(i,reduce) for i in range(self.n)])
        axes=list(range(1,len(self.shapes[i+1])))
        y=0.5*tf.reduce_sum(tf.square(self.ze[i].mean.get()) + tf.exp(self.ze[i].log_std.get()) - self.ze[i].log_std.get() - 1,axis=axes)
        if reduce:
            y=tf.reduce_mean(y)
        return y
    def loss_reconstruction(self,i=None,reduce=True,stacked=False):
        if i==None:
            return sum([self.loss_reconstruction(i,reduce,stacked) for i in range(self.n)])
        axes=list(range(1,len(self.shapes[i])))
        reg=1e-6
        if stacked:
            zd=self.zd_stacked[i]
        else:
            zd=self.zd[i]
        norm_x=tf.square((self.encoders[i].x-zd.mean.get())/(tf.exp(zd.log_std.get())+reg))/2
        log_det=np.log(np.sqrt(np.pi*2))+tf.maximum(zd.log_std.get(),np.log(reg))
        y=tf.reduce_sum(norm_x+log_det,axis=axes)
        if reduce:
            y=tf.reduce_mean(y)
        return y
    def loss(self,i=None,reduce=True,stacked=False):
        y=self.loss_reconstruction(i=i,reduce=False,stacked=stacked)+self.loss_kl(i=i,reduce=False)
        if reduce:
            y=tf.reduce_mean(y)
        return y
    def make_encoder(self):
        if "ze_encode" not in dir(self):
            self.encoder_input=Layer(type="Input",shape=self.shapes[0])
            self.encoder_encode=[]
            self.ze_encoder=[]
            for i in range(self.n):
                self.encoder_encode.append(self.encoders[i].copy(x=i==0 and self.encoder_input or self.ze_encoder[i-1]))
                self.ze_encoder.append(self.ze[i].copy(x=self.encoder_encode[i]))
    def make_decoder(self):
        if "decoder_decode" not in dir(self):
            self.decoder_input=Layer(type="Input",shape=self.shapes[-1])
            self.decoder_decode=[]
            self.zd_decoder=[]
            for i in range(self.n-1,-1,-1):
                self.decoder_decode.append(self.decoders[i].copy(x=i==self.n-1 and self.decoder_input or self.zd_decoder[i+1]))
                self.zd_decoder.append(self.zd[i].copy(x=self.decoder_decode[i]))
    def make_decoder_full(self):
        self.make_encoder()
        if "decoder_full" not in dir(self):
            self.decoder_full=[]
            self.zd_decoder_full=[]
            for i in range(self.n-1,-1,-1):
                self.decoder_full.append(self.decoders[i].copy(x=i==self.n-1 and self.ze_encoder[-1] or self.zd_decoder_full[i+1]))
                self.zd_decoder_full.append(self.zd[i].copy(x=self.decoder_full[i]))
    '''def make_generator(self):
        if "zd_generator" not in dir(self):
            self.generator_input=Layer(type="Normal_Learned",#Use numpy random instead?
                                       mean=dict(type="Input",shape=self.shapes[-1]),
                                       log_std=dict(type="Input",shape=self.shape)[-1])
            for i in range(self.n-1,-1,-1):
                self.decoder_decode.append(self.decoders[i].copy(x=i==self.n-1 and self.generator_input or self.zd_decoder[i+1]))
                self.zd_decoder.append(self.zd[i].copy(x=self.decoder_decode[i]))
            
            self.decoder_generator=self.decoder.copy(x=self.generator_input)
            self.zd_generator=self.zd.copy(x=self.decoder_generator)
    '''def encode(self,x):
        assert(self.sess!=None)
        self.make_encoder()
        return self.sess.run(self.ze_encoder,feed_dict={self.encoder_input.get():x})
    def encode_stats(self,x):
        assert(self.sess!=None)
        self.make_encoder()
        stats=self.binary_encoder and self.ze_encoder.prob or [self.ze_encoder.mean,self.ze_encoder.log_std]
        return self.sess.run(stats,feed_dict={self.encoder_input.get():x})
    def decode(self,x):
        assert(self.sess!=None)
        self.make_decoder()
        return self.sess.run(self.zd_decoder,feed_dict={self.decoder_input.get():x})
    def decode_stats(self,x):
        assert(self.sess!=None)
        self.make_decoder()
        stats=self.binary_encoder and self.zd_decoder.prob or [self.zd_decoder.mean,self.zd_decoder.log_std]
        return self.sess.run(stats,feed_dict={self.decoder_input.get():x})
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
        return self.sess.run(self.zd_generator,feed_dict={self.generator_input.mean.get():mean,
                                                               self.generator_input.log_std.get():log_std})
    def run_x(self,x):
        assert(self.sess!=None)
        self.make_decoder_full()
        return self.sess.run(self.zd_decoder_full,feed_dict={self.encoder_input.get():x})
    def run_x_stats(self,x):
        assert(self.sess!=None)
        self.make_decoder_full()
        stats=self.binary_decoder and self.zd_decoder_full.prob or [self.zd_decoder_full.mean,self.zd_decoder_full.log_std]
        return self.sess.run(stats,feed_dict={self.encoder_input.get():x})'''




























        