import tensorflow as tf
import numpy as np
import os
from utils import *
#from IPython.display import clear_output, Image, display, HTML

class DataProducer:
    def __init__(self,size,name="producer",mult=1,dic={}):
        self.name=defaultinDict(dic,"name",name)
        self.mult=defaultinDict(dic,"mult",mult)
        self.size=size
        self.label=None
        #self.ass_op=None
    def start(self,sess):#Virtual
        return
    def save(self,dic={}):
        dic["type"]="invalid"
        dic["name"]=self.name
        dic["mult"]=self.mult
        return dic
    def setLabel(self,index=0,numClasses=1,label=None):
        if label!=None:
            self.label=label
        else:
            self.label=tf.one_hot(index,numClasses,1.,0.,name='label_'+self.name)
    def get(self):
        return [self.data,self.label]

class Sampler(DataProducer):
    def __init__(self,size,dataShape,name="sampler",mult=1,dic={}):
        DataProducer.__init__(self,size,name,mult,dic)
        dataShape=defaultinDict(dic,"dataShape",dataShape)
        self.normalize=defaultinDict(dic,"normalize",False)
        self.noise=defaultinDict(dic,"noise",None)
        assert(dataShape!=None)
        self.dataShape=np.array(dataShape).astype(np.int32)
        self.dataShapeFix=self.dataShape-[size,size,0]
        self.index_queue=tf.train.range_input_producer(self.dataShapeFix.prod())
        self.index=self.index_queue.dequeue()
        self.row=tf.div(self.index,self.dataShapeFix[1])
        self.col=tf.mod(self.index,self.dataShapeFix[1])
        self.tf_arr=tf.Variable(tf.random_normal(self.dataShape, mean=0., stddev=1., dtype=tf.float32),trainable=False)
        self.data=tf.slice(self.tf_arr, tf.pack([self.row,self.col,0]), [self.size,self.size,1])
        print(self.data.get_shape())
        if self.normalize:
            self.data=tf.image.per_image_whitening(self.data)
        if self.noise!=None:
            self.data=self.data+tf.random_normal([self.size,self.size,1], mean=0., stddev=self.noise, dtype=tf.float32)
        print(self.data.get_shape())
        self.saver=tf.train.Saver({"data":self.tf_arr},max_to_keep=10000)
    def save(self,dic={}):
        DataProducer.save(self,dic)
        dic["dataShape"]=self.dataShape
        dic["type"]="sampler"
        dic["mult"]=self.mult
        return dic


class SamplerData(Sampler):
    def __init__(self,size,File=None,dataShape=None,name="sampler_data",mult=1,dic={}):
        File=defaultinDict(dic,"file",File)
        assert(File!=None)
        self.File=File
        dataShape=defaultinDict(dic,"dataShape",dataShape)
        if not os.path.isfile(File) or dataShape==None:
            dataShape=convertData(File)
        Sampler.__init__(self,size,dataShape,name=name,mult=mult,dic=dic)
    def start(self,sess):
        #print("Loading img")
        self.saver.restore(sess,self.File)
    def save(self,dic={}):
        Sampler.save(self,dic)
        dic["type"]="sampler_data"
        dic["file"]=self.File
        return dic
    @staticmethod
    def load(dic,size):
        assert(dic["type"]=="sampler_data")
        return SamplerData(size,dic=dic)
    
class SamplerDream(Sampler):
    def __init__(self,size,dataShape=None,name="sampler_dream",mult=1,dic={}):
        dataShape=defaultinDict(dic,"dataShape",dataShape)
        assert(dataShape[0]==dataShape[1] and dataShape[2]==1)
        Sampler.__init__(self,size,dataShape,name=name,mult=mult,dic=dic)
        self.assign_input=tf.placeholder(tf.float32,self.dataShape)
        self.assign_op=self.tf_arr.assign(self.assign_input)
    def assign(self,sess,img):
        sess.run(self.assign_op,{self.assign_input:img})
    def save(self,dic={}):
        Sampler.save(self,dic)
        dic["type"]="sampler_dream"
        return dic
    @staticmethod
    def load(dic,size):
        assert(dic["type"]=="sampler_dream")
        return SamplerDream(size,dic=dic)

class NoiseProducer(DataProducer):
    def __init__(self,size,mean=0., std=1.,name='noise',mult=1,dic={}):
        DataProducer.__init__(self,size,name,mult,dic)
        self.mean=defaultinDict(dic,"mean",mean)
        self.std=defaultinDict(dic,"std",std)
        self.data=tf.random_normal([size,size,1], mean=mean, stddev=std, dtype=tf.float32)
    def save(self,dic={}):
        DataProducer.save(self,dic)
        dic["type"]="noise"
        dic["mean"]=self.mean
        dic["std"]=self.std
        return dic
    @staticmethod
    def load(dic,size):
        assert(dic["type"]=="noise")
        return NoiseProducer(size,dic=dic)
 
#Resize not supported on GPU
'''class NoiseProducerScaled(NoiseProducer):
    def __init__(self,size0,size,mean=0., std=1.,name='noise_scaled',mult=1,device='/gpu:0'):
        NoiseProducer.__init__(self,size0,mean,std,name,mult)
        self.size=size
        self.size0=size0
        with tf.device(device):
            self.im=tf.expand_dims(self.data,0)
            self.data=tf.reshape(tf.image.resize_bilinear(self.im,[size,size]),[size,size,1])
    def save(self):
        return {"type":"noise_scaled","name":self.name,"mean":self.mean,"std":self.std,"size0":self.size0,"mult":self.mult}
    @staticmethod
    def load(producer,size):
        assert(producer["type"]=="noise_scaled")
        return NoiseProducerScaled(size=size,name=defaultinDict(producer,"name",None),
                            mean=defaultinDict(producer,"mean",0.),
                            std=defaultinDict(producer,"std",1.),
                            size0=defaultinDict(producer,"size0",None),
                            mult=defaultinDict(producer,"mult",1))'''

def producerFactory(size,dic):
    t=dic["type"]
    if t=="sampler_data":
        return SamplerData(size,dic=dic)
    elif t=="noise":
        return NoiseProducer(size,dic=dic)
    else:
        assert(False)


class BatchProducer:
    def __init__(self,batch=None,size=None,dic={}):
        self.batch=defaultinDict(dic,"batch",batch)
        self.size=defaultinDict(dic,"size",size)
        self.sess=None
        self.producer=producerFactory(self.size,dic["producer"])
        self.batchData=tf.train.batch([self.producer.data], batch_size=self.batch, num_threads=16, capacity=4096)
    def get(self):
        return self.batchData
    def start(self,sess):
        self.sess=sess
        self.producer.start(sess)
    def stop(self):
        self.sess=None
    def save(self,dic={}):
        dic["batch"]=self.batch
        dic["size"]=self.size
        dic["producer"]=producer.save()
        return dic
    
    
class BatchQueue:
    def __init__(self,batch=None,size=None,producers=None,finish=True,dic={}):
        self.batch=defaultinDict(dic,"batch",batch)
        self.size=defaultinDict(dic,"size",size)
        self.producers=[]
        self.mults=[]
        self.batchInput=[]
        self.numClasses=0
        self.ready=False
        self.sess=None
        producers=defaultinDict(dic,"producers",producers)
        if producers!=None:
            self._load(producers,finish)
    def addProducer(self,producer):
        assert(not self.ready)
        assert(producer.size==self.size)
        self.producers.append(producer)
        self.mults.append(producer.mult)
        self.numClasses+=1
    def finish(self):
        assert(not self.ready)
        for i,producer in enumerate(self.producers):
            producer.setLabel(i,self.numClasses)
            self.batchInput+=[producer.get()]*self.mults[i]
        self.batchData,self.batchLabels=tf.train.shuffle_batch_join(self.batchInput, 
                                                    self.batch,capacity=self.batch*2,min_after_dequeue=self.batch*1)
        #self.coord = tf.train.Coordinator()
        self.ready=True
        return self.get()
    def get(self):
        assert(self.ready)
        return [self.batchData,self.batchLabels]
    def makeLabel(self,i,zero_mean=False):
        assert(self.ready)
        label=np.zeros(self.numClasses, dtype=np.float32)
        label[i]=1.
        if zero_mean:
            label-=1./self.numClasses
        return label
    def start(self,sess):
        self.sess=sess
        for producer in self.producers:
            producer.start(sess)
        #self.threads = tf.train.start_queue_runners(coord=self.coord, start=True)
    def stop(self):
        #self.coord.request_stop()
        #self.coord.join(self.threads)
        self.sess=None
    def save(self,dic={}):
        dic["batch"]=self.batch
        dic["size"]=self.size
        dic["producers"]=[producer.save() for producer in self.producers]
        return dic
    def _load(self,producers,finish=True):
        for producer in producers:
            if producer["type"]=="sampler_data":
                self.addProducer(SamplerData.load(producer,self.size))
            elif producer["type"]=="sampler_dream":
                self.addProducer(SamplerDream.load(producer,self.size))
            elif producer["type"]=="noise":
                self.addProducer(NoiseProducer.load(producer,self.size))
            else:
                assert(False)
        if finish: self.finish()
    @staticmethod
    def load(dic,finish=True,prob_score=False):
        return BatchQueue(dic=dic,finish=finish,prob_score=prob_score)
        

