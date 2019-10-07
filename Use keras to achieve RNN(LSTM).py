# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:22:22 2019

@author: andy3
"""

##########################################
#
#
#
#Use keras to achieve RNN 'layers'
#
#
#
#
#
#########################################

#(shape)Simple RNN=(numbers of time point,input_Features),Keras RNN layers=(batch_size,numbers of time point,input_features)
from keras.layers import SimpleRNN

#Simple RNN from Keras:

from keras.models import Sequential

from keras.layers import Embedding,SimpleRNN
model=Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32))
model.summary()

##################################################################
#________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#embedding_8 (Embedding)      (None, None, 32)          320000    
#_________________________________________________________________
#simple_rnn_1 (SimpleRNN)     (None, 32)                2080      
#=================================================================
#Total params: 322,080
#Trainable params: 322,080
#Non-trainable params: 0
#_________________________________________________________________
##################################################################
#The batch_size we don't define, so it showed None.And you can see the shape 2,because we dont set the "return_sequences=True"

#The complete SimpleRNN from keras:

model=Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32,return_sequences=True))
model.summary()
#################################################################
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#embedding_12 (Embedding)     (None, None, 32)          320000    
#_________________________________________________________________
#simple_rnn_4 (SimpleRNN)     (None, None, 32)          2080      
#=================================================================
#Total params: 322,080
#Trainable params: 322,080
#Non-trainable params: 0
#_________________________________________________________________
#################################################################
#After setting the return_sqquences=True, the shape becomes 3.

#We also can add several SimpkleRNN layers to increase the ability of neural network.
model=Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32,return_sequences=True))
model.add(SimpleRNN(32))
model.summary()
#################################################################
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#embedding_15 (Embedding)     (None, None, 32)          320000    
#_________________________________________________________________
#simple_rnn_7 (SimpleRNN)     (None, None, 32)          2080      
#_________________________________________________________________
#simple_rnn_8 (SimpleRNN)     (None, None, 32)          2080      
#_________________________________________________________________
#simple_rnn_9 (SimpleRNN)     (None, None, 32)          2080      
#_________________________________________________________________
#simple_rnn_10 (SimpleRNN)    (None, 32)                2080      
#=================================================================
#Total params: 328,320
#Trainable params: 328,320
#Non-trainable params: 0
#_________________________________________________________________
#################################################################
#####Now, we prepare the IMDB datasets and input our RNN training and validation.

from keras.datasets import imdb

from keras.preprocessing import sequence
#Considering the features of words of numbers
max_features=5000
#We just discuss 500 words with each discussion
maxlen=500

batch_size=32
print('Loading the data...')

(input_train,y_train),(input_test,y_test)=imdb.load_data(num_words=max_features)
print(len(input_train),'train sequences')
print(len(input_test),'test sequences')
print('Pad sequences(samples*time)')
#只看每篇評論的500格文字，多的去除，不足填滿
input_train=sequence.pad_sequences(input_train,maxlen=maxlen)
input_test=sequence.pad_sequences(input_test,maxlen=maxlen)
print('input_train shape:',input_train.shape)
print('input_test shape:',input_test.shape)

###########################################################
#
#
#uSE eMBEDDING layers and SimpleRNN to train the Neural Network
#
#
###########################################################
from keras.layers import Dense,LSTM

model=Sequential()
model.add(Embedding(max_features,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(input_train,y_train,epochs=10,batch_size=128,validation_split=0.2)
model.save_weights('pre_trained_glove_model.h5')

import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Traning accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('The accuracy of training and validation')
plt.legend()

plt.figure()
plt.plot(epochs,loss,'bo',label='Training loss quantity')
plt.plot(epochs,val_loss,'b',label='Validation loss quantity')
plt.title('Training and validation loss')
plt.legend()

plt.show()