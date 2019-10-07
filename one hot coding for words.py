# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:06:38 2019

@author: andy3
"""

#################################################
#
#
#One-hot-encoding
#
#token trans to vector method(basic methiod) for 'words'
#################################################

import numpy as np
###initial data: this list of each element is sample for dict.
samples=['Ally is the best mage in the world.','Her RL is shit.','I love to travel， but hate to arrive.','It is harder to crack a prejudice than an atom.']
##create a dict for saving all tokens and code number.
token_index={}
#===================the create dict of setting were showed below:
for sample in samples:
    for word in sample.split():
        if  word not in token_index:
            token_index[word]=len(token_index)+1
#we finish the dict!
#======================token trans to vector(dict->vector)
max_length=10
results=np.zeros(shape=(len(samples),max_length,max(token_index.values())+1))
print(results.shape)

for i,sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index=token_index.get(word)
        results[i,j,index]=1.
print(results)