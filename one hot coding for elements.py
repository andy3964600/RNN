# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:41:30 2019

@author: andy3
"""
#################################################
#
#
#One-hot-encoding
#
#token trans to vector method(basic methiod) for 'word of elements'
#################################################
import numpy as np
import string
###initial data: this list of each element is sample for dict.
samples=['Ally is the best mage in the world.','Her RL is shit.','I love to travel， but hate to arrive.','It is harder to crack a prejudice than an atom.']
elements=string.printable
print(len(elements))

token_index=dict(zip(elements,range(1,len(elements)+1)))
max_length=50
results=np.zeros((len(samples),max_length,max(token_index.values())+1))
print(results.shape)
for i,sample in enumerate(samples):
    for j,element in enumerate(sample):
        index=token_index.get(element)
        results[i,j,index]=1.
