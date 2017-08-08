
# coding: utf-8

# In[2]:

import numpy as np
import theano
import string
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.utils import shuffle


# In[3]:

import theano.tensor as T


# In[4]:

def init_weight(Wi,Wo):
    return np.random.randn(Wi,Wo)/ np.sqrt(Wi+Wo)
def get_robert_frost():
    word2idx = {"START":0,"END":1}
    sentences = []
    current = 2
    with open('RobertFrost') as f:
        for line in f:
            line  = line.strip()
            if line:
                sentence = []
                line = line.translate(None, string.punctuation)
                for word in line.split():
                    if word not in word2idx:
                        word2idx[word] = current
                        current = current + 1
                    sentence.append(word2idx[word])
                sentences.append(sentence)
    return sentences,word2idx


# In[5]:

sentences,vocabulary = get_robert_frost()


# In[6]:

class lstm():
    def __init__(self,H,D,V):
        # number of hidden units
        self.H = H
        #dimension of the word embedding model
        self.D = D
        #vocabulary size model
        self.V = V
    def layer_init_weight(self):
        Wlx_init = init_weight(self.D,self.H)
        Wlh_init = init_weight(self.H,self.H)
        blh_init = np.zeros(self.H)
        
        thWlx = theano.shared(Wlx_init)
        thWlh = theano.shared(Wlh_init)
        thblh = theano.shared(blh_init)
        
        return thWlx,thWlh,thblh
    def set_lstm_variables(self):
        
        self.C0 = theano.shared(np.zeros(self.H))
        self.h0 = theano.shared(np.zeros(self.H))
        
        self.Wfx,self.Wfh,self.bfh = self.layer_init_weight()
        self.Wix,self.Wih,self.bih = self.layer_init_weight()
        self.WCx,self.WCh,self.bCh = self.layer_init_weight()
        self.Wox,self.Woh,self.boh = self.layer_init_weight()
    
    def fit(self,X,alpha=10e-5,mu=0.99,print_period=10,epochs=1):
        
        We_init = init_weight(self.V,self.D)
        self.We = theano.shared(We_init)
        
        thX = T.ivector('thX')
        thY = T.ivector('thY')
        Ei = self.We[thX]
        
        
        self.set_lstm_variables()
        ##set up output variables
        self.Wo = theano.shared(init_weight(self.H,self.V))
        self.bo = theano.shared(np.zeros(self.V))
        
        #will set all lstm variable in instance variable params
        #will also set dparams variable
        self.params = [self.We,self.C0,self.h0,self.Wfx,self.Wfh,self.bfh,self.Wix,self.Wih,self.bih,
                      self.WCx,self.WCh,self.bCh,self.Wox,self.Woh,self.boh,self.Wo,self.bo]
        #delta of the params
        self.dparams = [theano.shared(param.get_value()*0) for param in self.params]
        
        def recurrance(x_t,h_t1,C_t1):
            ft = T.nnet.sigmoid(x_t.dot(self.Wfx)+ h_t1.dot(self.Wfh) + self.bfh)
            it = T.nnet.sigmoid(x_t.dot(self.Wix)+ h_t1.dot(self.Wih) + self.bih)
            Cdasht = T.tanh(x_t.dot(self.WCx)+ h_t1.dot(self.WCh) + self.bCh)
            ot = T.nnet.sigmoid(x_t.dot(self.Wox)+ h_t1.dot(self.Woh) + self.boh)
            C_t = ft*C_t1 + it*Cdasht
            ot = T.nnet.sigmoid(x_t.dot(self.Wox)+ h_t1.dot(self.Woh) + self.boh)
            h_t = ot*T.tanh(C_t)
            return h_t,C_t
        
        #forward run of sequence
        [h,C],_ = theano.scan(
                                fn = recurrance,
                                sequences=Ei,
                                outputs_info=[self.h0,self.C0],
                                n_steps=Ei.shape[0]
                            )
        
        py_x = T.nnet.softmax(h.dot(self.Wo) + self.bo)
        
        prediction = T.argmax(py_x,axis=1)
        
        cost = -1*T.mean(T.log(py_x[T.arange(thY.shape[0]),thY]))
        grads = T.grad(cost,self.params)
        
        
        updates = [(p,p + (mu*dp - alpha*g)) for g,p,dp in zip(grads,self.params,self.dparams)] + [
            (dp,mu*dp - alpha*g) for g,p,dp in zip(grads,self.params,self.dparams)
        ]
        self.train = theano.function(
            inputs = [thX,thY],
            outputs = [cost,prediction],
            updates = updates
        )
        
        self.predict = theano.function(
            inputs = [thX],
            outputs = [prediction]
        )
        
        count = 0
        costs = []
        n_total = sum([len(row) for row in X])
        for e in xrange(epochs):
            first = True
            cost = 0
            total_correct_words = 0
            X = shuffle(X)
            for row in X:
                trainX = [0] + row
                trainY = row + [1]
                [c,prediction] = self.train(thX=trainX,thY=trainY)
                for p,t in zip(prediction,trainY):
                    if p==t:
                        total_correct_words = total_correct_words + 1
                count = count + 1
                cost = cost + c
            print "Cost is",cost," Prediction is",(1.0*total_correct_words)/n_total
            costs.append(cost)
        plt.plot(costs)
    def predict():
        print X
    
        


# In[7]:

ls = lstm(30,30,len(vocabulary))


# In[ ]:

ls.fit(sentences,epochs=100)


# In[209]:

mat = np.arange(100)
mat = mat.reshape(50,2)
array = [1,2,3,3,5]


# In[1]:

23==34


# In[210]:

ast = np.array(sentences)
ast.shape


# In[ ]:



