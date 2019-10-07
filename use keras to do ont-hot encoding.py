# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:03:58 2019

@author: andy3
"""

#################################################
#
#
#One-hot-encoding
#
#token trans to vector method(basic methiod) for 'word of elements'
#
#In keras,it alos can do the one-hot encoding
#################################################
from keras.preprocessing.text import Tokenizer
samples=['Ally is the best mage in the world.','Her RL is shit.','I love to travel','but hate to arrive.','It is harder to crack a prejudice than an atom.']
tokenizer=Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
print('111:',tokenizer.word_index)
sequences=tokenizer.texts_to_sequences(samples)
print(tokenizer.texts_to_sequences(samples))
print(sequences)

one_hot_results=tokenizer.texts_to_matrix(samples,mode='binary')
print(one_hot_results)
print(one_hot_results[0][:15])

