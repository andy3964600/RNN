# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:09:34 2019

@author: andy3
"""

import numpy as np
#you can set different the quantity of temperature
def past_temperature_probability_distribution(origin_probability_distribution,temperature=0.01):
    
    probability_distribution=np.log(origin_probability_distribution)/temperature
    
    probability_distribution=np.exp(probability_distribution)
    
    return probability_distribution/np.sum(probability_distribution)

origin_pro=np.array([0.6,0.3,0.1])

past_temp_pro=past_temperature_probability_distribution(origin_pro)

print(past_temp_pro)
