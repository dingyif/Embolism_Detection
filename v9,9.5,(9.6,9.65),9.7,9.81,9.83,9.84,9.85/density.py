# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:41:15 2020

@author: USER
"""

import cv2
import numpy as np
import math

def density_of_a_rect(img,window_width):
#    #binary image(0/1)
#    img = np.array([[1, 0, 0, 1, 0],
#                   [0, 0, 0, 1, 0],
#                   [0, 1, 0, 1, 0],
#                   [1, 1, 0, 1, 0],
#                   [1, 0, 0, 0, 1]], dtype='uint8')
#     window_width = 3#the width of rectangle window, assumes to be an odd number
#    #now assumes the rectangle is square
    
    img_width = img.shape[0]
    img_height = img.shape[1]
    
    #pad first so that pixels at the edges still work correctly
    pad_sz = math.floor(window_width/2)
    pad_img = np.pad(img,pad_sz)
    
    prev_img = cv2.integral(pad_img)
    #summing all the previous pixels. 
    #Previous here means all the pixels above and to the left of that pixel (inclusive of that pixel)
    
    
    #the sum of all the pixels in the rectangular window 
    #  Sum = Bottom right + top left - top right - bottom left
    prev_img_width = prev_img.shape[0]
    prev_img_height = prev_img.shape[1]
    bottom_right = np.copy(prev_img[(prev_img_width-img_width):,(prev_img_height-img_height):])
    top_left = np.copy(prev_img[0:img_width,0:img_height])
    top_right = np.copy(prev_img[0:img_width,(prev_img_height-img_height):])
    bottom_left = np.copy(prev_img[(prev_img_width-img_width):,0:img_height])
    sum_img = bottom_right + top_left - top_right - bottom_left
    density_img = sum_img/(window_width*window_width)
    return(density_img)
