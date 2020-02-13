# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:10:10 2019

@author: USER
"""
import numpy as np
from scipy import ndimage #for median filter,extract foreground



def median_filter_stack(img_stack,kernel_radius):
    #img_stack is 3d numpy array, 1st index = img index
    #replace each pixel by the median in kernel_size*kernel_size neighborhood
    med_stack = np.ndarray(img_stack.shape, dtype=np.float32)
    for img_idx in range(img_stack.shape[0]):
        img_current = img_stack[img_idx,:,:]
        med_stack[img_idx] = ndimage.median_filter(img_current, kernel_radius)
    return(med_stack)
