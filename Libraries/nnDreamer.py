import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from functools import partial
from IPython.display import clear_output, Image, display, HTML
import PIL.Image
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from nnBuilder import *
from io import BytesIO
#from nnDreamer import *
#from dream import *
from dataProducer import *

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize_(img, size):
    img = tf.expand_dims(tf.expand_dims(img, 0), 3)
    return tf.image.resize_bicubic(img, size,align_corners=True)[0,:,:,0]
resize = tffunc(np.float32, np.int32)(resize_)





#k = np.float32([np.exp(-(i)**2/2) for i in range(-2,3)])#np.float32([1,4,6,4,1])
k = np.float32([np.exp(-(i/2)**2/2) for i in range(-4,5)])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()#*np.eye(1, dtype=np.float32)

def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi#, lo2
#lapSplit = tffunc(np.float32)(lap_split)

def _lap_blur(img):
    return tf.nn.conv2d(img, k5x5, [1,1,1,1], 'SAME')
def _lap_down(img):
    return tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
def _lap_up(img):
    return tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(img)*[1,2,2,1], [1,2,2,1])
lap_blur2 = tffunc(np.float32)(_lap_blur)
lap_down2 = tffunc(np.float32)(_lap_down)
lap_up2 = tffunc(np.float32)(_lap_up)
def lap_blur(img):
    return lap_blur2(img[None,:,:,None])[0,:,:,0]
def lap_down(img):
    return lap_down2(img[None,:,:,None])[0,:,:,0]
def lap_up(img):
    return lap_up2(img[None,:,:,None])[0,:,:,0]


def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]
#lapSplitN = tffunc(np.float32)(lap_split_n)

def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    #img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    muls=[1.8**3,1.8**2,1.8,1.]
    for i in range(len(tlevels)):
        tlevels[i]=tlevels[i]*muls[i]
    out = lap_merge(tlevels)
    #return out[0,:,:,:]
    return out







class Dreamer:
    def __init__(self,x_,score,img=None,use_lap=False):
        self.x_=x_
        self.score=score
        self.use_lap=use_lap
        self.size=x_.get_shape()[1].value
        self.img=None
        self.has_saved=False
        if img != None:
            self.setImg(img)
        self.step=0.01
        self.reg=1e-4
        self.clip=5.
        self.tf_step_size=tf.placeholder_with_default(np.float32(self.step),[])
        self.tf_grad_norm_reg=tf.placeholder_with_default(np.float32(self.reg),[])
        self.tf_grad_clip=tf.placeholder_with_default(np.float32(self.clip),[])
        self.tf_grad = tf.gradients(self.score, self.x_)[0]
        self.tf_grad_norm= tf.reshape(tf.add(tf.reduce_mean(tf.abs(self.tf_grad),[1,2,3]),self.tf_grad_norm_reg),[-1,1,1,1])
        self.tf_step=tf.div(self.tf_grad,self.tf_grad_norm)
        self.tf_step_clipped=tf.maximum(tf.minimum(self.tf_step,self.tf_grad_clip),-self.tf_grad_clip)
        self.tf_step_full=tf.mul(self.tf_step_size,self.tf_step_clipped)
        
        self.lap_grad=lap_normalize(self.tf_grad,3)
        self.lap_grad_norm= tf.reshape(tf.add(tf.reduce_mean(tf.abs(self.lap_grad),[1,2,3]),self.tf_grad_norm_reg),[-1,1,1,1])
        self.lap_step=tf.div(self.lap_grad,self.lap_grad_norm)
        self.lap_step_clipped=tf.maximum(tf.minimum(self.lap_step,self.tf_grad_clip),-self.tf_grad_clip)
        self.lap_step_full=tf.mul(self.tf_step_size,self.lap_step_clipped)
        #self.lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=4))
        self.sess=None
    def train_step(self,unroll=False):
        assert(self.sess!=None)
        self.rand_roll()
        self._train_step()
        if unroll:
            self.unroll()
    def rand_roll(self):
        shift = [np.random.randint(self.shape[0]),np.random.randint(self.shape[1])]
        self.totalroll-=shift
        self.img=np.roll(self.img,-shift[1],2)
        self.img=np.roll(self.img,-shift[0],1)
        return shift
    def unroll(self):
        self.img=np.roll(self.img,-self.totalroll[1],2)
        self.img=np.roll(self.img,-self.totalroll[0],1)
        self.totalroll=np.array([0,0])
    def _train_step(self):
        newarr=np.concatenate(np.split(np.concatenate(np.split(
                        self.img,self.shape[0]//self.size,2),0),self.shape[1]//self.size,1),0)
        dic={self.x_:newarr,self.tf_step_size:self.step,self.tf_grad_norm_reg:self.reg,self.tf_grad_clip:self.clip}
        if self.use_lap:
            step=self.sess.run(self.lap_step_full,dic)
        else:
            step=self.sess.run(self.tf_step_full,dic)
        #if self.use_lap:
        #    step=self.lap_norm_func(step)
        for y in range(0,self.shape[1],self.size):
            for x in range(0,self.shape[0],self.size):
                #print(self.shape,x,y)
                self.img[0,y:y+self.size,x:x+self.size,:]+=step[(y//self.size)*(self.shape[0]//self.size)+x//self.size]
    def _train_step_scaled(self,scale):
        scaled_shape=self.shape[:2]//scale
        img_scaled=resize(self.img,scaled_shape)
        newarr=np.concatenate(np.split(np.concatenate(np.split(
                        img_scaled,scaled_shape[0]//self.size,2),0),scaled_shape[1]//self.size,1),0)
        dic={self.x_:newarr,self.tf_step_size:self.step,self.tf_grad_norm_reg:self.reg,self.tf_grad_clip:self.clip}
        if self.use_lap:
            step=self.sess.run(self.lap_step_full,dic)
        else:
            step=self.sess.run(self.tf_step_full,dic)
        #if self.use_lap:
        #    step=self.lap_norm_func(step)
        for y in range(0,scaled_shape[1],self.size):
            for x in range(0,scaled_shape[0],self.size):
                #print(self.shape,x,y)
                img_scaled[0,y:y+self.size,x:x+self.size,:]=step[(y//self.size)*(scaled_shape[0]//self.size)+x//self.size]
        step_full=resize(img_scaled,self.shape[:2])
        self.img+=step_full
    def train(self,n=100,step=None,reg=None,clip=None,unroll=True):
        self.step=step or self.step
        self.reg=reg or self.reg
        self.clip=clip or self.clip
        for i in range(n):
            self.train_step(False)
        if unroll:
            self.unroll()
    def randomize(self,size=None):
        if size==None:
            if self.img==None:
                size=self.size
            else:
                size=self.img.size
        self.setImg(np.random.normal(size=(size,size),loc=0.,scale=1.))
    def get(self,img=None):
        if img==None:
            return self.img[0,:,:,0]
        return img
    def get3d(self):
        return self.img[0,:,:,:]
    def setImg(self,img):
        self.shape=list(img.shape)
        if len(self.shape)==2:
            self.shape=self.shape+[1]
        assert(self.shape[0]%self.size==0 and self.shape[1]%self.size==0)
        self.img=img.reshape([1]+self.shape).astype(np.float32)
        self.totalroll=np.array([0,0])
    def save_np(self,File,safe=True):
        if safe:
            assert(not os.path.isfile(File))
        np.save(File,self.get())
    def make_saver_tf(self):
        shape=self.shape[0:2]+[1]
        if not self.has_saved or shape!=self.tf_img.get_shape():#Memory leak?
            self.tf_img=tf.Variable(tf.zeros(shape,dtype=tf.float32),trainable=False)
            self.tf_saver=tf.train.Saver({"data":self.tf_img},max_to_keep=10000)
            self.tf_ass_in=tf.placeholder(tf.float32,shape)
            self.tf_ass_op=self.tf_img.assign(self.tf_ass_in)
    def save_tf(self,File,safe=True):
        assert(self.sess!=None)
        if safe:
            assert(not os.path.isfile(File))
        self.make_saver_tf()
        self.sess.run(self.tf_ass_op,{self.tf_ass_in:self.get3d()})
        self.tf_saver.save(self.sess,File,write_meta_graph=False)
    def load_tf(self,File):
        assert(self.sess!=None)
        self.make_saver_tf()
        self.tf_saver.restore(self.sess,File)
        img=self.sess.run(self.tf_img)
        self.setImg(img.reshape(img.shape[:2]))
    def load_sample(self,folder):
        self.load_tf(folder+"/sample")
    def save_sample(self,folder,safe=True):
        self.save_tf(fixDir(folder)+"/sample",safe=safe)
    def stats(self,img=None):
        img=self.get(img)
        return (img.min(),img.max(),img.mean(),img.std())
    #def _evaluate_logits(self):
    #    x=np.random.randint(0,self.shape[1]-self.size)
    #    y=np.random.randint(0,self.shape[0]-self.size)
    #    return self.network.sess.run(self.network.logits,{self.network.x_:self.img[:,y:y+self.size,x:x+self.size,:]})
    #def evaluate_logits(self,n=10):
    #    return np.array([self._evaluate_logits() for i in range(n)]).mean(axis=0)
    def plot(self,img=None,cmap=plt.cm.Greys):
        return plt.imshow(self.get(img),cmap=cmap, interpolation='nearest')
    def plot3d(self,img=None,cmap='terrain',reg=10):
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        img=self.get(img)
        img[0,0]+=reg
        X, Y = np.meshgrid(range(img.shape[0]),range(img.shape[1]))  # `plot_surface` expects `x` and `y` data to be 2D
        plot=ha.plot_surface(X, Y, img,cmap=cmap)
        img[0,0]-=reg
        return plot
    def show(self,img=None):
        showarray(self.get(img))
    def start(self,sess):
        self.sess=sess
    def stop(self):
        self.sess=None
    
class SimpleDreamer(Dreamer):
    def __init__(self,x_,logits,img=None,sigmoid=False):
        n=logits.get_shape()[-1].value
        self.logits=tf.reshape(logits,[-1,n])
        labels=tf.tile(tf.expand_dims(tf.one_hot(0,n,1.,0.,dtype=tf.float32),0),tf.pack([tf.shape(self.logits)[0],1]))
        if sigmoid:
            print("sigmoid")
            score=-tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits,labels))
            self.softmaxLayer=BasicSigmoid(self.logits)
        else:
            print("softmax")
            score=-tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits,labels))
            self.softmaxLayer=BasicSoftmax(self.logits)
        self.y=self.softmaxLayer.get()
        Dreamer.__init__(self,x_,score,img)
    def _evaluate(self):
        x=np.random.randint(0,self.shape[1]-self.size+1)
        y=np.random.randint(0,self.shape[0]-self.size+1)
        return self.sess.run(self.y,{self.x_:self.img[:,y:y+self.size,x:x+self.size,:]})
    def evaluate(self,n=10):
        assert(self.sess!=None)
        return np.array([self._evaluate() for i in range(n)]).mean(axis=0).mean(axis=0)
    def _evaluate_logits(self):
        x=np.random.randint(0,self.shape[1]-self.size+1)
        y=np.random.randint(0,self.shape[0]-self.size+1)
        return self.sess.run(self.logits,{self.x_:self.img[:,y:y+self.size,x:x+self.size,:]})
    def evaluate_logits(self,n=10):
        assert(self.sess!=None)
        a=np.array([self._evaluate_logits() for i in range(n)])
        while len(a.shape)>1:
            a=a.mean(axis=0)
        return a

        

"""class DreamerHam(BasicDreamer):
    #Hamiltonian system H(x,p)=0.5p^2+V(x)
    def __init__(self,network,V=None,img=None,global_mean=False):
        BasicDreamer.__init__(self,network,img)
        self.p=None
        self.step_time=0.01
        self.step_diss_x=0.01
        self.step_diss_p=0.01
        self.reg=1.
        self.clip=5.
        if V==None:
            if network.logits.get_shape()[1]==2:
                self.tf_V=tf.reduce_mean(tf.matmul(self.network.logits,tf.expand_dims([-1.,1.],1)))
            else:
                sample_label=self.network.batchQueue.makeLabel(0,zero_mean=False)
                logZreg=tf.reduce_max(self.network.logits)#Avoid nan
                logZ=logZreg+tf.log(tf.reduce_sum(tf.exp(self.network.logits-logZreg)))
                self.tf_V=-tf.reduce_mean(tf.matmul(self.network.logits,tf.expand_dims(sample_label,1)))+logZ
        else:
            self.tf_V=V
        self.tf_grad = tf.gradients(self.tf_V, network.x_)[0]
        self.tf_grad_norm_reg=tf.placeholder_with_default(np.float32(self.reg),[])
        self.tf_grad_clip=tf.placeholder_with_default(np.float32(self.clip),[])
        if global_mean: ind=[0,1,2,3]
        else: ind=[1,2,3]
        self.tf_grad_norm_base= tf.reshape(tf.reduce_mean(tf.abs(self.tf_grad),ind),[-1,1,1,1])
        self.tf_grad_norm= tf.add(self.tf_grad_norm_base,self.tf_grad_norm_reg)
        self.tf_step=tf.div(self.tf_grad,self.tf_grad_norm)
        self.tf_grad_clip_reg=self.tf_grad_clip*self.tf_grad_norm_base/self.tf_grad_norm
        self.tf_step_clipped=tf.maximum(tf.minimum(self.tf_step,self.tf_grad_clip_reg),-self.tf_grad_clip_reg)
        self.tf_step_full=self.tf_step_clipped
    def setImg(self,img,p=None):
        BasicDreamer.setImg(self,img)
        if p==None:
            self.p=np.zeros_like(self.img)
        else:
            assert(p.shape==img.shape)
            self.p=p.reshape([1]+list(p.shape)+[1]).astype(np.float32)
    def _train_step(self):
        #Dissipation step: dp/d tau=-del H/del p = -p, dx/d tau=-del H/del x =-V'(x)
        #self.tf_step_diss=tf.placeholder_with_default(np.float32(0.01),[])
        #Time step: dp/dt=-del H/del x = -V'(x), dx/dt=del H/del p =p
        #self.tf_step_time=tf.placeholder_with_default(np.float32(0.01),[])
        newarr=np.concatenate(np.split(np.concatenate(np.split(
                        self.img,self.img.shape[1]//self.size,2),0),self.img.shape[2]//self.size,1),0)
        grads=self.network.sess.run(self.tf_step_full,{self.network.x_:newarr,self.tf_grad_norm_reg:self.reg,self.tf_grad_clip:self.clip})
        for y in range(0,self.shape[1],self.size):
            for x in range(0,self.shape[0],self.size):
                self.p[0,y:y+self.size,x:x+self.size,:]-=self.step_time*grads[(y//self.size)*(self.shape[0]//self.size)+x//self.size]+self.step_diss_p*self.p[0,y:y+self.size,x:x+self.size,:]
                self.img[0,y:y+self.size,x:x+self.size,:]+=self.step_time*self.p[0,y:y+self.size,x:x+self.size,:]-self.step_diss_x*grads[(y//self.size)*(self.shape[0]//self.size)+x//self.size]
    def rand_roll(self):
        (shiftx,shifty)=BasicDreamer.rand_roll(self)
        self.p=np.roll(self.p,-shiftx,2)
        self.p=np.roll(self.p,-shifty,1)
    def train(self,n=100,step_time=None,step_diss_x=None,step_diss_p=None,reg=None,clip=None,unroll=True):
        self.reg=reg or self.reg
        self.step_time=step_time or self.step_time
        self.step_diss_x=step_diss_x or self.step_diss_x
        self.step_diss_p=step_diss_p or self.step_diss_p
        self.clip=clip or self.clip
        BasicDreamer.train(self,n)
    def unroll(self):
        self.p=np.roll(self.p,-self.totalroll[1],2)
        self.p=np.roll(self.p,-self.totalroll[0],1)
        BasicDreamer.unroll(self)
    def getP(self):
        return self.p[0,:,:,0]
    def getP3d(self):
        return self.p[0,:,:,:]
    def statsP(self):
        return self.stats(p)
    def plotP(self):
        return plot(self,img=self.getP())
    def plotP3d(self,cmap='terrain',reg=10):
        return plot3d(self,img=self.getP(),cmap=cmap,reg=reg)
    def showP(self):
        show(self,img=self.getP())
    
    
    
class TrainingNetworkWithDreamer(TrainingNetwork):
    def __init__(self,dic,prob_score=False):
        TrainingNetwork.__init__(self,batchQueue=None,dic=dic,prob_score=prob_score)
        sample_label=self.batchQueue.makeLabel(0,zero_mean=False)
        sample_label_tiled=tf.tile(tf.expand_dims(sample_label,0),tf.pack([tf.shape(self.logits)[0],1]))
        self.dream_score=tf.mul(self.logits,sample_label_tiled)
        self.dreamer=Dreamer(self,self.dream_score)
    
class TrainingNetworkWithDreamerHam(TrainingNetwork):
    def __init__(self,dic):
        TrainingNetwork.__init__(self,batchQueue=None,dic=dic)
        self.dreamer=DreamerHam(self)
    
class DreamTrainer(TrainingNetworkWithDreamer):
    def __init__(self,dic,prob_score=False):
        assert(len(dic['producers'])==2 and dic['producers'][0]['type']=='sampler_data' 
               and dic['producers'][1]['type']=='sampler_dream')
        TrainingNetworkWithDreamer.__init__(self,dic=dic,prob_score=prob_score)
        self.sampler_dream=self.batchQueue.producers[1]
        self.dreamer_step=defaultinDict(dic,"dreamer_step",1.0)
        self.dreamer_nsteps=defaultinDict(dic,"dreamer_nsteps",100)
        self.dreamer_reg=defaultinDict(dic,"dreamer_reg",1e-0)
        self.dreamer_size=self.sampler_dream.dataShape[0]
        self.batches_per_dream=defaultinDict(dic,"batches_per_dream",10)
    def train(self,n,rand_dream=True):
        self.ensureSess()
        for i in range(0,n,self.batches_per_dream):
            self.update_dreamer(rand_dream)
            if i%100==0:
                self.dreamer.show()
            TrainingNetwork.train(self,self.batches_per_dream)
    def update_dreamer(self,rand_dream=True):
        if rand_dream:
            self.dreamer_reg=np.exp(-15*np.random.random())
            self.dreamer_step=np.exp(-6*np.random.random())
            self.dreamer_nsteps=int(np.exp(4*np.random.random()+3))
        self.dreamer.randomize(self.dreamer_size)
        self.dreamer.train(n=self.dreamer_nsteps,step=self.dreamer_step,reg=self.dreamer_reg,unroll=False)
        self.sampler_dream.assign(self.sess,self.dreamer.get3d())
    
    
'''class DreamerNP:
    def __init__(self,size,layersDef,target,File=None,img=None,use_logits=True):
        self.size=size
        self.File=File
        self.target=target
        self.img=None
        #The image to optimize
        #Build a network for the image slice
        self.tf_input=tf.placeholder(tf.float32,[1,size,size,1],name='input')
        self.network=Network(self.tf_input,layersDef=layersDef)
        #Setup a trainer
        if use_logits: 
            self.y=self.network.logits
            self.score = tf.nn.softmax_cross_entropy_with_logits(self.y,tf.expand_dims(target,0))
            #self.score = -tf.reduce_mean(tf.matmul(self.y,tf.expand_dims(target,1)))
        else: 
            self.y=self.network.y
            self.score = -tf.reduce_mean(tf.matmul(self.y,tf.expand_dims(target,1)))
        #self.score = tf.reduce_mean(tf.matmul(self.y,tf.expand_dims(target,1)))
        #self.score = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        self.tf_step_size=tf.placeholder_with_default(np.float32(0.01),[])
        self.tf_grad_norm_reg=tf.placeholder_with_default(np.float32(1e-4),[])
        self.tf_grad_clip=tf.placeholder_with_default(np.float32(5.),[])
        self.tf_grad = tf.gradients(self.score, self.tf_input)[0]
        self.tf_grad_norm= tf.add(tf.div(tf.reduce_sum(tf.abs(self.tf_grad)),size**2),self.tf_grad_norm_reg)
        self.tf_step=tf.div(self.tf_grad,self.tf_grad_norm)
        self.tf_step_clipped=tf.maximum(tf.minimum(self.tf_step,self.tf_grad_clip),-self.tf_grad_clip)
        self.tf_step_full=tf.mul(self.tf_step_size,self.tf_step_clipped)
        #Start a session and load the model
        self.sess=self.network.startSess()
        #tf.initialize_all_variables().run()
        if img != None:
            self.setImg(img)
        if File != None:
            self.network.loadVars(self.sess,File)
    def train(self,step=0.01,n=100,reg=1e-4):
        totalroll=[0,0]
        for i in range(n*self.step_size):
            #Roll only if needed
            shiftx = np.random.randint(self.shape[1])
            shifty = np.random.randint(self.shape[0])
            if shiftx>self.shape[1]-self.size:
                self.img=np.roll(self.img,-shiftx,2)
                totalroll[1]-=shiftx
                shiftx=0
            if shifty>self.shape[0]-self.size:
                self.img=np.roll(self.img,-shifty,1)
                totalroll[0]-=shifty
                shifty=0
            #print(shiftx,shifty)
            g=self.sess.run(self.tf_step_full,{self.tf_step_size:step,
                    self.tf_input:self.img[:,shifty:shifty+self.size,shiftx:shiftx+self.size,:],self.tf_grad_norm_reg:reg})
            self.img[:,shifty:shifty+self.size,shiftx:shiftx+self.size,:]-=g
        self.img=np.roll(self.img,-totalroll[1],2)
        self.img=np.roll(self.img,-totalroll[0],1)
    def randomize(self,size=None):
        if size==None:
            if self.img==None:
                size=self.size
            else:
                size=self.img.size
        self.setImg(np.random.normal(size=(size,size),loc=0.,scale=1.))
    def sample(self,folder,size,step=0.01,n=100,show=False,safe=True):
        self.loadVars(folder)
        self.randomize(size)
        self.train(step=step,n=n)
        if show:
            self.show_bmp()
        self.saveImg(folder+"/sample.npy",safe=safe)
    def sample_many(self,size,m,saveDir="",saveBaseName="model",step=0.01,n=100,show=False,safe=True):
        for i in range(m):
            self.sample(folder=saveDir+"/"+saveBaseName+"_%i"%i,size=size,step=step,n=n,show=show,safe=safe)
    def get(self):
        return self.img[0,:,:,0]
    def get_maybe(self,img):
        if img==None:
            return self.get()
        return img
    def setImg(self,img):
        self.step_size=int(np.ceil(img.size/(self.size*self.size)))
        self.shape=img.shape
        self.img=img.reshape([1]+list(img.shape)+[1]).astype(np.float32)
    def saveImg(self,File,safe=True):
        if safe:
            assert(not os.path.isfile(File))
        np.save(File,self.get())
    def loadVars(self,folder):
        self.network.loadVars(folder)
    def quit(self):
        self.network.ensureNoSess()
    def stats(self):
        img=self.get()
        return (img.min(),img.max(),img.mean(),img.std())
    def evaluate(self):
        return self.sess.run(self.network.y,{self.tf_input:self.img[:,:self.size,:+self.size,:]})
    def evaluate_logits(self):
        return self.sess.run(self.network.logits,{self.tf_input:self.img[:,:self.size,:+self.size,:]})
    def plot(self,img=None):
        return plt.imshow(self.get_maybe(img),cmap=plt.cm.Greys, interpolation='nearest')
    def plot3d(self,img=None,cmap='terrain',reg=10):
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        img=self.get_maybe(img)
        img[0,0]+=reg
        X, Y = np.meshgrid(range(img.shape[0]),range(img.shape[0]))  # `plot_surface` expects `x` and `y` data to be 2D
        plot=ha.plot_surface(X, Y, img,cmap=cmap)
        img[0,0]-=reg
        return plot
    def show(self):
        showarray(self.get_maybe(img))'''
"""

class DreamerConv:
    def __init__(self,x_,score,img=None,use_lap=False):
        self.x_=x_
        self.score=score
        self.use_lap=use_lap
        self.size=x_.get_shape()[1].value
        self.img=None
        self.has_saved=False
        if img != None:
            self.setImg(img)
        self.step=0.01
        self.reg=1e-4
        self.clip=5.
        self.tf_step_size=tf.placeholder_with_default(np.float32(self.step),[])
        self.tf_grad_norm_reg=tf.placeholder_with_default(np.float32(self.reg),[])
        self.tf_grad_clip=tf.placeholder_with_default(np.float32(self.clip),[])
        self.tf_grad = tf.gradients(self.score, self.x_)[0]
        self.tf_grad_norm= tf.reshape(tf.add(tf.reduce_mean(tf.abs(self.tf_grad),[1,2,3]),self.tf_grad_norm_reg),[-1,1,1,1])
        self.tf_step=tf.div(self.tf_grad,self.tf_grad_norm)
        self.tf_step_clipped=tf.maximum(tf.minimum(self.tf_step,self.tf_grad_clip),-self.tf_grad_clip)
        self.tf_step_full=tf.mul(self.tf_step_size,self.tf_step_clipped)
        
        self.lap_grad=lap_normalize(self.tf_grad,3)
        self.lap_grad_norm= tf.reshape(tf.add(tf.reduce_mean(tf.abs(self.lap_grad),[1,2,3]),self.tf_grad_norm_reg),[-1,1,1,1])
        self.lap_step=tf.div(self.lap_grad,self.lap_grad_norm)
        self.lap_step_clipped=tf.maximum(tf.minimum(self.lap_step,self.tf_grad_clip),-self.tf_grad_clip)
        self.lap_step_full=tf.mul(self.tf_step_size,self.lap_step_clipped)
        #self.lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=4))
        self.sess=None
    def train_step(self,unroll=False):
        assert(self.sess!=None)
        #self.rand_roll()
        self._train_step()
        #if unroll:
        #    self.unroll()
    '''def rand_roll(self):
        shift = [np.random.randint(self.shape[0]),np.random.randint(self.shape[1])]
        self.totalroll-=shift
        self.img=np.roll(self.img,-shift[1],2)
        self.img=np.roll(self.img,-shift[0],1)
        return shift
    def unroll(self):
        self.img=np.roll(self.img,-self.totalroll[1],2)
        self.img=np.roll(self.img,-self.totalroll[0],1)
        self.totalroll=np.array([0,0])'''
    def _train_step(self):
        #newarr=np.concatenate(np.split(np.concatenate(np.split(
        #                self.img,self.shape[0]//self.size,2),0),self.shape[1]//self.size,1),0)
        dic={self.x_:self.img,self.tf_step_size:self.step,self.tf_grad_norm_reg:self.reg,self.tf_grad_clip:self.clip}
        if self.use_lap:
            step=self.sess.run(self.lap_step_full,dic)
        else:
            step=self.sess.run(self.tf_step_full,dic)
        self.img+=step
        #if self.use_lap:
        #    step=self.lap_norm_func(step)
        #for y in range(0,self.shape[1],self.size):
        #    for x in range(0,self.shape[0],self.size):
                #print(self.shape,x,y)
        #        self.img[0,y:y+self.size,x:x+self.size,:]+=step[(y//self.size)*(self.shape[0]//self.size)+x//self.size]
    def train(self,n=100,step=None,reg=None,clip=None,unroll=True):
        self.step=step or self.step
        self.reg=reg or self.reg
        self.clip=clip or self.clip
        for i in range(n):
            self.train_step(False)
        #if unroll:
        #    self.unroll()
    def randomize(self,size=None):
        if size==None:
            if self.img==None:
                size=self.size
            else:
                size=self.img.size
        self.setImg(np.random.normal(size=(size,size),loc=0.,scale=1.))
    def get(self,img=None):
        if img==None:
            return self.img[0,:,:,0]
        return img
    def get3d(self):
        return self.img[0,:,:,:]
    def setImg(self,img):
        self.shape=list(img.shape)
        if len(self.shape)==2:
            self.shape=self.shape+[1]
        #assert(self.shape[0]%self.size==0 and self.shape[1]%self.size==0)
        self.img=img.reshape([1]+self.shape).astype(np.float32)
        self.totalroll=np.array([0,0])
    def save_np(self,File,safe=True):
        if safe:
            assert(not os.path.isfile(File))
        np.save(File,self.get())
    def make_saver_tf(self):
        shape=self.shape[0:2]+[1]
        if not self.has_saved or shape!=self.tf_img.get_shape():#Memory leak?
            self.tf_img=tf.Variable(tf.zeros(shape,dtype=tf.float32),trainable=False)
            self.tf_saver=tf.train.Saver({"data":self.tf_img},max_to_keep=10000)
            self.tf_ass_in=tf.placeholder(tf.float32,shape)
            self.tf_ass_op=self.tf_img.assign(self.tf_ass_in)
    def save_tf(self,File,safe=True):
        assert(self.sess!=None)
        if safe:
            assert(not os.path.isfile(File))
        self.make_saver_tf()
        self.sess.run(self.tf_ass_op,{self.tf_ass_in:self.get3d()})
        self.tf_saver.save(self.sess,File,write_meta_graph=False)
    def load_tf(self,File):
        assert(self.sess!=None)
        self.make_saver_tf()
        self.tf_saver.restore(self.sess,File)
        img=self.sess.run(self.tf_img)
        self.setImg(img.reshape(img.shape[:2]))
    def load_sample(self,folder):
        self.load_tf(folder+"/sample")
    def save_sample(self,folder,safe=True):
        self.save_tf(fixDir(folder)+"/sample",safe=safe)
    def stats(self,img=None):
        img=self.get(img)
        return (img.min(),img.max(),img.mean(),img.std())
    #def _evaluate_logits(self):
    #    x=np.random.randint(0,self.shape[1]-self.size)
    #    y=np.random.randint(0,self.shape[0]-self.size)
    #    return self.network.sess.run(self.network.logits,{self.network.x_:self.img[:,y:y+self.size,x:x+self.size,:]})
    #def evaluate_logits(self,n=10):
    #    return np.array([self._evaluate_logits() for i in range(n)]).mean(axis=0)
    def plot(self,img=None,cmap=plt.cm.Greys):
        return plt.imshow(self.get(img),cmap=cmap, interpolation='nearest')
    def plot3d(self,img=None,cmap='terrain',reg=10):
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')
        img=self.get(img)
        img[0,0]+=reg
        X, Y = np.meshgrid(range(img.shape[0]),range(img.shape[1]))  # `plot_surface` expects `x` and `y` data to be 2D
        plot=ha.plot_surface(X, Y, img,cmap=cmap)
        img[0,0]-=reg
        return plot
    def show(self,img=None):
        showarray(self.get(img))
    def start(self,sess):
        self.sess=sess
    def stop(self):
        self.sess=None
    

class SimpleDreamerConv(DreamerConv):
    def __init__(self,x_,logits,img=None,sigmoid=False):
        n=logits.get_shape()[-1].value
        self.logits=tf.reshape(logits,[-1,n])
        labels=tf.tile(tf.expand_dims(tf.one_hot(0,n,1.,0.,dtype=tf.float32),0),tf.pack([tf.shape(self.logits)[0],1]))
        if sigmoid:
            score=-tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits,labels))
            self.softmaxLayer=BasicSigmoid(self.logits)
        else:
            score=-tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits,labels))
            self.softmaxLayer=BasicSoftmax(self.logits)
        self.y=self.softmaxLayer.get()
        DreamerConv.__init__(self,x_,score,img)
    def _evaluate(self):
        x=np.random.randint(0,self.shape[1]-self.size+1)
        y=np.random.randint(0,self.shape[0]-self.size+1)
        return self.sess.run(self.y,{self.x_:self.img[:,y:y+self.size,x:x+self.size,:]})
    def evaluate(self,n=10):
        assert(self.sess!=None)
        return np.array([self._evaluate() for i in range(n)]).mean(axis=0).mean(axis=0)
    def _evaluate_logits(self):
        x=np.random.randint(0,self.shape[1]-self.size+1)
        y=np.random.randint(0,self.shape[0]-self.size+1)
        return self.sess.run(self.logits,{self.x_:self.img[:,y:y+self.size,x:x+self.size,:]})
    def evaluate_logits(self,n=10):
        assert(self.sess!=None)
        a=np.array([self._evaluate_logits() for i in range(n)])
        while len(a.shape)>1:
            a=a.mean(axis=0)
        return a
























