# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:16:30 2020

@author: USER
"""
from PIL import Image
import numpy as np
import cv2
import sys
from scipy import ndimage 
import math
import matplotlib.pyplot as plt


def plot_gray_img(gray_img,title_str=""):
    #gray img: 1D np.array
    plt.figure()
    plt.imshow(gray_img,cmap='gray')
    plt.title(title_str)

def get_pivot_center(stem_path,plot_interm): 
    '''
    Determine the pivot center from an input stem segmentation image (white: stem, black background) (stem_path)
    using fitting of 2nd degree polynomial
    if the stem isn't curved, it should output the bottomest point
    '''
    #stem_path = 'E:/Diane/Col/research/code/Done/Processed/version_output/v11/inglau3_stem.DONEGOOD.HANNAH.11.22/v11_0_1_200/m_3_is_stem_mat2_0.jpg'
    stem_img0=Image.open(stem_path).convert('L') #.convert('L'): gray-scale # 646x958
    stem_arr0 = np.float32(stem_img0)/255 #convert to array and {0,255} --> {0,1}
    row_num = stem_arr0.shape[0]#img height
    col_num = stem_arr0.shape[1]#img width
    
    plot_gray_img(stem_arr0)
    
    smooth_stem = ndimage.gaussian_filter(stem_arr0, sigma = 10)
    bin_smooth_stem = (smooth_stem>0.5).astype(np.uint8)
    num_cc, mat_cc, stats, centroids  = cv2.connectedComponentsWithStats(bin_smooth_stem, 8)
    centroid_col = centroids[1][0]#centroids[0] is bgd, centroids[1] is stem
    centroid_row = centroids[1][1]
    
    contours_stem, _ = cv2.findContours(bin_smooth_stem,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_stem)>1:
        #there's more than one contour
        #get the contour with max elements (i.e. max size)
        contour_size = np.array([one_contour.shape[0] for one_contour in contours_stem])
        max_cc_label = np.where(contour_size==max(contour_size))[0][0]
        contour_stem_max_sz = contours_stem[max_cc_label][:,0,:]#drop the second axis, cuz contours_stem[max_cc_label] is of shape: (number of pts,1,2)
    elif len(contours_stem)==0:
        sys.exit("[Error] Didn't find any contour")
    else:
        print("Good :) only one contour in stem")
        max_cc_label = 0
        contour_stem_max_sz = contours_stem[max_cc_label][:,0,:]
    
    
    contour_stem_list = contour_stem_max_sz.tolist()
    contour_stem_list.append(contour_stem_list[0])
    
    ##polylabel: doesn't work that well
    #from polylabel import polylabel #https://github.com/Twista/python-polylabel#doesn't work that well...
    #stem_center = polylabel([contour_stem_list]) 
    #
    #fig, ax = plt.subplots(figsize = (5,5))
    #ax.set_xlim(0, col_num)
    #ax.set_ylim(0, row_num)
    #plt.scatter(contour_stem_max_sz[:,0],contour_stem_max_sz[:,1], alpha=0.2)
    #plt.scatter(stem_center[0],stem_center[1])
    #plt.gca().invert_yaxis()#invert y-axis, so that it doesn't looked like flipped of raw image
    
    #contour_stem_x_diff = contour_stem_max_sz[1:,0]-contour_stem_max_sz[:-1,0] #next pt-prev pt #unused
    contour_stem_y_diff = contour_stem_max_sz[1:,1]-contour_stem_max_sz[:-1,1] 
    
    #stem is roughly like a rectangle, which have 4 sides of contours
    #assume one side of the contour won't have changes in the sign of contour_stem_y_diff 
    #TODO: have to check this
    #first place that change sign in y
    last_pt_idx = np.where(contour_stem_y_diff[:-1] * contour_stem_y_diff[1:] <= 0 )[0][0] +1
    one_side_contour = contour_stem_max_sz[0:(last_pt_idx+1)]
    
    ##plotting out
    #fig, ax = plt.subplots(figsize = (math.floor(6/row_num*col_num),6))
    #ax.set_xlim(0, col_num)
    #ax.set_ylim(0, row_num)
    #plt.scatter(one_side_contour[:,0], one_side_contour[:,1], alpha=0.2)
    #plt.gca().invert_yaxis()#invert y-axis, so that it doesn't looked like flipped of raw image
    #plt.show()
    
    '''
    2nd degree polynomial fitting for determining concave direction
    '''
    #x,y values for 2-degree polynomial fitting
    y = one_side_contour[:,0]#use column index as the y in poly fitting
    x = one_side_contour[:,1]#row index
    
    poly_degree = 2
    poly_coeff = np.polyfit(x, y, poly_degree)#Polynomial coefficients, highest power first
    poly_func = np.poly1d(poly_coeff)#fitted polynomial function, i.e. poly_func(x) would be the fitted value of x using poly_coeff
    
    #for plotting out pts(blue) and fitted curve(orange)
    x_range = np.linspace(min(x), max(x), max(x)-min(x)+1).astype(int)#range of row index for plotting the fitted curve
    fitted_y = poly_func(x_range)#fitted value of x_range
    
    #fig, ax = plt.subplots(figsize = (math.floor(6/row_num*col_num),6))
    #ax.set_xlim(0, col_num)
    #ax.set_ylim(0, row_num)
    #plt.plot(y, x, '.',fitted_y , x_range, '-')
    #plt.gca().invert_yaxis()
    #plt.show()
    
    #detemine pivot's row position by looking at the 2nd order coefficient
    second_order_coeff = poly_coeff[0]
    near_0_threshold = 10**(-7)#if abs(second_order_coeff) <  near_0_threshold, then it's consider as no curving
    #second_order_coeff: a2 (~-2*10^(-4)),c2.2(~-2*10^(-4)), c5(~10^(-5)),in3(~-1*10^(-4)),in4(~4*10^(-5)),in5(~-3*10^(-4))
    '''
    if abs(second_order_coeff) <  near_0_threshold, then it's consider as no curving. If use_centroid=True, use the centroid of stem as pivot. Else, use the bottomest point as pivot.
    if f"(x) (second_order_coeff) > 0 --> concave up (like a U shape) --> use local min for the pivot's row position
    if f"(x) < 0 --> concave down (like a upside down U shape) --> use local  max  for the pivot's row position
    '''
    use_centroid = False
    
    if abs(second_order_coeff)<near_0_threshold:
        print("Polynomial is less than 2 degree, cuz second order coefficient=",second_order_coeff)
        is_2_deg = False
        if use_centroid==True:
            pivot_col_pos = centroid_col
            pivot_row_pos = centroid_row
        else:
            pivot_row_pos = max(np.where(stem_arr0==1)[0])#bottomest pt
    elif second_order_coeff>near_0_threshold:
        is_2_deg = True
        min_col_idx = np.where(fitted_y == min(fitted_y))[0][0]
        pivot_row_pos = x_range[min_col_idx]
    else:
        is_2_deg = True
        max_col_idx = np.where(fitted_y == max(fitted_y))[0][0]
        pivot_row_pos = x_range[max_col_idx]
    
    #assume the stem contours are parallel
    #use the mean of the column index of stem on the row slice (pivot's row position) for pivot's column index
    if is_2_deg==True or use_centroid==False:
        col_idx_is_stem = np.where(stem_arr0[pivot_row_pos,:]==1)[0]#get all the col index where row_index == pivot_row_pos AND is stem
        pivot_col_pos = int(np.round(np.mean(col_idx_is_stem)))
    
    if plot_interm==True:
        #plot out
        fig, ax = plt.subplots(figsize = (math.floor(6/row_num*col_num),6))
        ax.set_xlim(0, col_num)
        ax.set_ylim(0, row_num)
        plt.plot(contour_stem_max_sz[:,0], contour_stem_max_sz[:,1], '.', label="contour")
        plt.plot(fitted_y , x_range, '-', label="polynomial")
        plt.plot(pivot_col_pos,pivot_row_pos,'rx', label="pivot")
        plt.plot(centroid_col,centroid_row,'k.', label="centroid of stem bounding box")#centroid of bounding box of stem
        plt.legend(bbox_to_anchor=(1.04,1),loc="upper left")
        #https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        #use bbox_to_anchor=(1.04,1) to place legend out of the plot
        plt.gca().invert_yaxis()
        plt.show()
    
    pivot_center = np.array((pivot_col_pos,pivot_row_pos))
    
    return(pivot_center)

stem_path = 'E:/Diane/Col/research/code/Done/Processed/version_output/v11/inglau3_stem.DONEGOOD.HANNAH.11.22/v11_0_1_200/m_3_is_stem_mat2_0.jpg'
pivot_center = get_pivot_center(stem_path,plot_interm=True)



    