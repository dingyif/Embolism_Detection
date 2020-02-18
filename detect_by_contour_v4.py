# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:00:53 2019

@author: USER
"""

from PIL import Image,ImageDraw,ImageFont #ImageDraw,ImageFont for drawing text
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage #for median filter,extract foreground
import time #to time function performance
import tifffile as tiff
import pandas as pd
import cv2
import operator#for contour
import os,shutil#for creating/emptying folders
import sys#for printing error message
import gc
import density#self-written density.py
from collections import Counter#for count 1,2 in overlap mat
#############################################################################
#    Sanity check:
#    see if the sum of pixel values in an image of diff_pos_stack
#    is similar to those from true_diff_sum (the intermediate result they obtained from ImageJ)
#############################################################################
def plot_img_sum(img_3d,title_str,img_folder,is_save=False):
    sum_of_img = np.sum(np.sum(img_3d,axis=2),axis=1)#sum
    plt.figure()
    plt.plot(range(len(sum_of_img)),sum_of_img)
    plt.ylabel("sum of pixel values in an image")
    plt.xlabel("image relative index")
    plt.title(title_str)
    if is_save==True:
        plt.savefig(img_folder+'/m0_'+title_str+'.jpg',bbox_inches='tight')
    # x-axis= image index. y-axis=sum of each img
    return(sum_of_img)

#############################################################################
#    function for plotting overlap sum
#############################################################################
def plot_overlap_sum(img_3d, title_str = img_folder_name ,chunk_folder, is_save = False):
    diff_is_stem_mat2 = is_stem_mat2[1:,:,:] - is_stem_mat2[:-1,:,:]
    diff_shape = diff_is_stem_mat2.shape()
    count_num_2 = np.array([]) ; count_num_1 = np.array([])
    for i in Diff_shape[0]:
        diff = Diff_is_stem_mat2[i,:,:]
        counted_num = Counter([num for sublist in diff for num in sublist])
        count_num_2 = np.append(count_num_2, counted_num[2])
        count_num_1 = np.append(count_num_1, counted_num[1])
        prop = count_2/(count_1+count_2)
    plt.figure()
    plt.plot(range(len(prop)),prop)
    plt.ylabel("portion of overlap area")
    plt.xlabel("image relative index")
    plt.title(title_str)

    if is_save == True:
        plt.savefig(chunk_folder + "/m1_overlap_area_hist_img.jpg",bbox_inches='tight')
    return(prop)
    
#############################################################################
#    function for plotting
#############################################################################

def plot_gray_img(gray_img,title_str=""):
    #gray img: 1D np.array
    plt.figure()
    plt.imshow(gray_img,cmap='gray')
    plt.title(title_str)

#############################################################################
#    Thresholding (clip pixel values smaller than threshold to 0)
#    threshold is currently set to 3 (suggested by Chris)
#############################################################################

#############################################################################
#    Binarize: clip all positive px values to 1
#############################################################################

def to_binary(img_stack):
    #convert 0 to 0(black) and convert all postivie pixel values to 1(white)
    return((img_stack > 0)*1)

#############################################################################
#    Foreground Background Segmentation
#    for stem only, cuz there are clearly parts that are background (not stem)
#    want to avoid  artifacts from shrinking of gels/ moving of plastic cover
#    mask only the stem part to reduce false positive
#############################################################################

def extract_foreground(img_2d,chunk_folder, blur_radius=10.0,fg_threshold=0.1,expand_radius_ratio=5,is_save=False):
    '''
    smooth img_2d by gaussian filter w/blur radius
    background when smoothed img > fg_threhold
    expand the foreground object area by applying uniform filter 
    radius = blur_radiu*expand_radius_ratio then binarize it
    expansion is made to be more conservative
    displays two image, 1st img is how much expansion of foreground is done
           2nd img is the foreground(True) background(False) result
    return a 2d logical array of the same size as img_2d
    
    fg_threshold assumes embolism only happens at 
    the same given pixel at most 10% of the time
    i.e. this is an assumption on how much xylem channels overlay
    '''
    
    #smoothing (gaussian filter)
    smooth_img = ndimage.gaussian_filter(img_2d, blur_radius)
    
    not_stem = (smooth_img > fg_threshold)*1 
    
    #expand the stem part a bit by shrinking the not_stem
    unif_radius = blur_radius*expand_radius_ratio
    not_stem_exp =  to_binary(ndimage.uniform_filter(not_stem, size=unif_radius))
    
    is_stem = (not_stem_exp==0)#invert
    
    plot_gray_img(is_stem)#1(white) for stem part
    if is_save==True:
        plt.imsave(chunk_folder + "/m2_1_is_stem_before_max_area.jpg",is_stem,cmap='gray')
    
    num_cc_stem, mat_cc_stem = cv2.connectedComponents(is_stem.astype(np.uint8))
    unique_cc_stem_label = np.unique(mat_cc_stem) 
    #list of areas
    area = []
    if unique_cc_stem_label.size > 1: #more than 1 area 
        for cc_label_stem in unique_cc_stem_label[1:]:
            area.append(np.sum(mat_cc_stem == cc_label_stem))
    #real stem part
    if area:
        max_area = max(area)
        for cc_label_stem in unique_cc_stem_label[1:]:
            if np.sum(mat_cc_stem == cc_label_stem) < max_area:
                mat_cc_stem[mat_cc_stem == cc_label_stem] = 0
        is_stem = is_stem * mat_cc_stem
    else:#no part is being selected as stem --> treat entire img as stem
        is_stem = is_stem+1

    plot_gray_img(is_stem+not_stem)#expansion
    if is_save==True:
        plt.imsave(chunk_folder+"/m2_expansion_foreground.jpg",is_stem+not_stem,cmap='gray')
    plot_gray_img(is_stem)#1(white) for stem part
    if is_save==True:
        plt.imsave(chunk_folder + "/m3_is_stem.jpg",is_stem,cmap='gray')
    return(is_stem)#logical 2D array

def extract_foregroundRGB(img_2d,chunk_folder, blur_radius=10.0,expand_radius_ratio=3,is_save=False):
    '''
    assume stem is more green than backgorund
    '''
    
    #smoothing (gaussian filter)
    smooth_img = ndimage.gaussian_filter(img_2d, blur_radius)
    
    is_stem_mat = (smooth_img > np.mean(img_2d))*1 
    
    #plot_gray_img(is_stem_mat)#1(white) for stem part
    if is_save==True:
        plt.imsave(chunk_folder + "/m2_1_is_stem_mat_before_max_area.jpg",is_stem_mat,cmap='gray')
    
    num_cc_stem, mat_cc_stem = cv2.connectedComponents(is_stem_mat.astype(np.uint8))
    unique_cc_stem_label = np.unique(mat_cc_stem) 
    #list of areas
    area = []
    if unique_cc_stem_label.size > 1: #more than 1 area 
        for cc_label_stem in unique_cc_stem_label[1:]:
            area.append(np.sum(mat_cc_stem == cc_label_stem))
    #real stem part
    if area:
        max_area = max(area)
        for cc_label_stem in unique_cc_stem_label[1:]:
            if np.sum(mat_cc_stem == cc_label_stem) < max_area:
                mat_cc_stem[mat_cc_stem == cc_label_stem] = 0
        is_stem_mat = is_stem_mat * mat_cc_stem
    else:#no part is being selected as stem --> treat entire img as stem
        is_stem_mat = is_stem_mat+1
    
    #expand the stem part a bit by shrinking the not_stem
    not_stem = -is_stem_mat+1
    unif_radius = blur_radius*expand_radius_ratio
    not_stem_exp =  to_binary(ndimage.uniform_filter(not_stem, size=unif_radius))
    
    is_stem_mat = (not_stem_exp==0)#invert
    
    #plot_gray_img(is_stem+not_stem)#expansion
    if is_save==True:
        plt.imsave(chunk_folder+"/m2_expansion_foreground.jpg",is_stem_mat+not_stem,cmap='gray')
    #plot_gray_img(is_stem)#1(white) for stem part
    if is_save==True:
        plot_gray_img(is_stem_mat)#1(white) for stem part
        plt.imsave(chunk_folder + "/m3_is_stem_mat.jpg",is_stem_mat,cmap='gray')
    return(is_stem_mat)#logical 2D array

#############################################################################
#    main function for detecting embolism
############################################################################# 
def find_emoblism_by_contour(bin_stack,img_idx,stem_area,final_area_th = 20000/255,area_th=30, area_th2=30,ratio_th=5,e2_sz=3,o2_sz=1,cl2_sz=3,plot_interm=False,shift_th=0.05,density_th=0.4,num_px_th=50):
    ############# step 1: connect the embolism parts in the mask (more false positive) ############# 
    #    opening(2*2) and closing(5*5)[closing_e] 
    #    --> keep contours with area>area_th and fill in the contour by polygons [contour_mask]
    #    --> expand by closing(25*25) and dilation(10,10), then find connected components [mat_cc]
    ################################################################################################
    kernel = np.ones((2,2),np.uint8) #square image kernel used for erosion
    #erosion = cv2.erode(bin_stack[img_idx,:,:].astype(np.uint8), kernel,iterations = 1) #refines all edges in the binary image
    opening1 = cv2.morphologyEx(bin_stack[img_idx,:,:].astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if plot_interm == True:
        #plot_gray_img(erosion,str(img_idx)+"_erosion")
        plot_gray_img(opening1,str(img_idx)+"_opening1")
    
    kernel_cl = np.ones((5,5),np.uint8)
    #closing_e = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel_cl)
    closing_e = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel_cl)
    #using a larger kernel (kernel_cl to connect the embolism part)
    #enlarge the boundaries of foreground (bright) regions
    if plot_interm == True:
        plot_gray_img(closing_e,str(img_idx)+"_closing_e")
    
    contours, _ = cv2.findContours(closing_e,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    #find contours with simple approximation
    
    #list to hold all areas
    areas = [cv2.contourArea(contour) for contour in contours]
    
    #remove particles with area < area_th
    area_index = [i for i,val in enumerate(areas) if val >= area_th]#extract the index in areas with areas >= area_th
    #enumerate is used in case there are multiply indices with the same areas value
    
    if len(area_index) == 0:#no particles with area >= area_th
        return(bin_stack[img_idx,:,:]*0)#just return no embolism
    
    else:
        cnt = operator.itemgetter(*area_index)(contours)
        
        closing_contour = cv2.fillPoly(closing_e, cnt, 125)
        #includes closing(0 or 255) and areas in cnt(125)
        contour_mask = (closing_contour==125)# = True if px is in cnt
        if plot_interm == True:
            plot_gray_img(contour_mask,str(img_idx)+"_contour_mask")
        
        #expand a bit to connect the components
        kernel_cla = np.ones((25,25),np.uint8)
        closing_a = cv2.morphologyEx((contour_mask*1).astype(np.uint8), cv2.MORPH_CLOSE, kernel_cla)
        if plot_interm == True:
            plot_gray_img(closing_a,str(img_idx)+"_closing_a")
        
        kernel_d = np.ones((10,10),np.uint8)
        dilate_img = cv2.dilate(closing_a, kernel_d,iterations = 1)
        if plot_interm == True:
            plot_gray_img(dilate_img,str(img_idx)+"_dilate_img")
        
        num_cc,mat_cc = cv2.connectedComponents(dilate_img.astype(np.uint8))
        
        ################### step 2: shrink the embolism parts (more false negative) ###################
        #    erosion(e2_sz*e2_sz) and opening(o2_sz*o2_sz) and then closing(cl2_sz*cl2_sz) [closing2]
        #    --> keep contours with area > area_th2 and fill in the contour by polygons [contour_mask2]
        ################################################################################################
        kernel_e2 = np.ones((e2_sz,e2_sz),np.uint8) #square image kernel used for erosion
        erosion2 = cv2.erode(bin_stack[img_idx,:,:].astype(np.uint8), kernel_e2,iterations = 1) #refines all edges in the binary image
        if plot_interm == True:
            plot_gray_img(erosion2, str(img_idx) + "_erosion2")
        
        kernel_o2 = np.ones((o2_sz,o2_sz),np.uint8)
        #https://homepages.inf.ed.ac.uk/rbf/HIPR2/open.htm
        opening = cv2.morphologyEx(erosion2.astype(np.uint8), cv2.MORPH_OPEN, kernel_o2)
        #tends to remove some of the foreground (bright) pixels from the edges of regions of foreground pixels.
        #However it is less destructive than erosion in general.
        
        if plot_interm == True:
            plot_gray_img(opening, str(img_idx) + "_opening2")
        
        #the closing here is needed for cases like ALCLAT1_leaf Subset: img_idx = 57
        #where embolism is break into small chunks and disappear mostly after opening
        kernel_cl2 = np.ones((cl2_sz,cl2_sz),np.uint8)
        closing2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_cl2)
        if plot_interm == True:
            plot_gray_img(closing2,str(img_idx)+"_closing2")

        #find contours with simple approximation
        contours2, hierarchy2 = cv2.findContours(closing2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        
        #list to hold all areas
        areas2 = [cv2.contourArea(contour2) for contour2 in contours2]
         
        #remove particles with area < area_th2
        area_index2 = [i for i,val in enumerate(areas2) if val >= area_th2]#extract the index in areas with areas>=area_th
        #enumerate is used in case there are multiply indices with the same areas value
        
        if len(area_index2) == 0:#no particles with area >= area_th
            return(bin_stack[img_idx,:,:]*0)#just return no embolism
        else:
            cnt2 = operator.itemgetter(*area_index2)(contours2)
            #cnt = [contours[area_index]] #only useful when area_index only has one element
            
            closing_contour2 = cv2.fillPoly(closing2, cnt2, 125)
            #includes closing(0 or 255) and areas in cnt(125)
            contour_mask2 = (closing_contour2==125)# = True if px is in cnt
            if plot_interm == True:
                plot_gray_img(contour_mask2,str(img_idx)+"_contour_mask2")
            
            ################### step 3: pixel values from step2 correspond to which connected component from step1 ###################
            #    for a pixel w/ value = 1 in [contour_mask2],
            #    --> the intersection of the connected component [mat_cc] where that pixel falls into and 
            #        binarized image [bin_stack] is treated as an candidate for embolism [add_part]
            #    -->if (the area of [add_part])/(the area of intersection of connected component and [contour_mask2]) < ratio_th,
            #        [add_part] is an embolism
            #        (this condition is needed to avoid the case that a small noise in [contour_mask] leads to a really big [add_part])
            ###########################################################################################################################
            
            intersect_img = mat_cc*contour_mask2 #intersect connected component mat with bin_contour_img2
            unique_cc_label = np.unique(intersect_img)#see which connected component should be kept
            final_mask = bin_stack[img_idx,:,:]*0
            if unique_cc_label.size > 1: #not only backgroung
                for cc_label in unique_cc_label[1:]: #starts with "1", cuz "0" is for bkg
                    add_part = (mat_cc == cc_label)*bin_stack[img_idx,:,:]
                    area_cm2 = np.sum((mat_cc == cc_label)*contour_mask2)#number of pixels in contour_mask2 that are 1
                    if plot_interm == True:
                        #print("number of pixel added/area of that connected component in dilate_img",np.sum(add_part)/np.sum(mat_cc == cc_label))
                        print(str(img_idx)+":"+str(np.sum(add_part)/area_cm2))
                    if np.sum(add_part)/area_cm2 < ratio_th:# and np.sum(add_part)/np.sum(mat_cc == cc_label)>0.19:
                        '''
                        1st condition
                        avoid the case that a small noise in contour_mask leads to a really big "add_part"
                        that are essentially just noises
                        case: ALCLAT1_leaf Subset: img_idx = 190
                        2nd condition
                        crude way of discarding parts with low density
                        '''
                        final_mask = final_mask + add_part
                    
            final_img = np.copy(final_mask)
            if plot_interm == True:
                plot_gray_img(final_mask,str(img_idx)+"_final_mask")
                print(img_idx,np.sum(final_img)/stem_area,np.sum(final_img))
            
            #####################################  step 4: reduce false positive ########################## 
            #    (image level)
            #    if the percentage of embolism in the [stem_area] is too large (probably it's because of shifting of images) 
            #        OR the embolism area is too small
            #    --> treat as if there's no embolism in the entire image
            #    (connected component level)
            #    Discard connected components w/small density or area
            ################################################################################################
            if np.sum(final_img)/stem_area > shift_th:
                '''
                percentage of embolism in the stem_area is too large
                probably is false positive due to shifting of img (case: stem img_idx=66,67)
                '''
                if plot_interm==True:
                    print("percentage of embolism in the stem_area is too large",np.sum(final_img)/stem_area,"> shift_th =",shift_th)
                final_img = final_img * 0
                return(final_img)

            if np.sum(final_img) < final_area_th:
                '''
                remove false positive with really small sizes
                '''
                if plot_interm==True:
                    print("embolism area is too small", np.sum(final_img),"< final_area_th =", final_area_th)
                final_img = final_img * 0
                return(final_img)
            
            
            #reduce false positive, discard those with small density
            final_img_cp = np.copy(final_img)#w/o np.copy, changes made in final_img_cp will affect final_img
            closing_3 = cv2.morphologyEx(final_img_cp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
            if plot_interm == True:
                plot_gray_img(closing_3,str(img_idx)+"_closing_3")
            
            final_mask3 = bin_stack[img_idx,:,:]*0
            num_cc3,mat_cc3 = cv2.connectedComponents(closing_3.astype(np.uint8))
            unique_cc_label3 = np.unique(mat_cc3)
            if unique_cc_label3.size > 1: #not only background
                for cc_label3 in unique_cc_label3[1:]: #starts with "1", cuz "0" is for bkg
                    inside_cc3 = (mat_cc3 == cc_label3)*bin_stack[img_idx,:,:]
                    if plot_interm == True:
                        print("estimated of density",np.sum(inside_cc3)/np.sum(mat_cc3 == cc_label3))
                        print("estimated number of pixels inside",np.sum(inside_cc3))
                    #print(str(img_idx)+":"+str(np.sum(add_part)/area_cm2))
                    if np.sum(inside_cc3)/np.sum(mat_cc3 == cc_label3)>density_th and np.sum(inside_cc3)>num_px_th :
                        #discarding parts with low density
                        #discarding parts with small area ( ~ < 50 pixels) #!!! threshold for small area: depends on the distance btw the camera and the stem 
                            
                        final_mask3 = final_mask3 + inside_cc3
            if plot_interm == True:
                plot_gray_img(final_mask3,str(img_idx)+"_final_mask3")
            final_img = final_mask3
            
            if plot_interm == True:
                plot_gray_img(final_img,str(img_idx)+"_final_img")
            return(final_img*255)
            
def find_emoblism_by_filter_contour(bin_stack,filter_stack,img_idx,stem_area,final_area_th = 20000/255,area_th=30, area_th2=30,ratio_th=5,c1_sz=25,d1_sz=10,e2_sz=3,o2_sz=1,cl2_sz=3,plot_interm=False,shift_th=0.05,density_th=0.4,num_px_th=50):
    ############# step 1: connect the embolism parts in the mask (more false positive) ############# 
    #    opening(2*2) and closing(5*5)[closing_e] 
    #    --> keep contours with area>area_th and fill in the contour by polygons [contour_mask]
    #    --> expand by closing(25*25) and dilation(10,10), then find connected components [mat_cc]
    ################################################################################################
    filter_img = filter_stack[img_idx,:,:]
    
#    if plot_interm == True:
#        plot_gray_img(filter_img,str(img_idx)+"_filter_img")
#    
#    contours, _ = cv2.findContours(filter_img.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    
    kernel = np.ones((2,2),np.uint8) #square image kernel used for erosion
    #erosion = cv2.erode(bin_stack[img_idx,:,:].astype(np.uint8), kernel,iterations = 1) #refines all edges in the binary image
    opening1 = cv2.morphologyEx(filter_img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if plot_interm == True:
        #plot_gray_img(erosion,str(img_idx)+"_erosion")
        plot_gray_img(opening1,str(img_idx)+"_opening1")
    
    kernel_cl = np.ones((5,5),np.uint8)
    #closing_e = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel_cl)
    closing_e = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel_cl)
    #using a larger kernel (kernel_cl to connect the embolism part)
    #enlarge the boundaries of foreground (bright) regions
    if plot_interm == True:
        plot_gray_img(closing_e,str(img_idx)+"_closing_e")
    
    contours, _ = cv2.findContours(closing_e,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    #find contours with simple approximation
    
    #list to hold all areas
    areas = [cv2.contourArea(contour) for contour in contours]
    
    #remove particles with area < area_th
    area_index = [i for i,val in enumerate(areas) if val >= area_th]#extract the index in areas with areas >= area_th
    #enumerate is used in case there are multiply indices with the same areas value
    
    if len(area_index) == 0:#no particles with area >= area_th
        return(bin_stack[img_idx,:,:]*0)#just return no embolism
    
    else:
        cnt = operator.itemgetter(*area_index)(contours)
        
        closing_contour = cv2.fillPoly(filter_img, cnt, 125)
        #includes closing(0 or 255) and areas in cnt(125)
        contour_mask = (closing_contour==125)# = True if px is in cnt
        if plot_interm == True:
            plot_gray_img(contour_mask,str(img_idx)+"_contour_mask")
        
        #expand a bit to connect the components
        kernel_cla = np.ones((c1_sz,c1_sz),np.uint8)
        closing_a = cv2.morphologyEx((contour_mask*1).astype(np.uint8), cv2.MORPH_CLOSE, kernel_cla)
        if plot_interm == True:
            plot_gray_img(closing_a,str(img_idx)+"_closing_a")
        
        kernel_d = np.ones((d1_sz,d1_sz),np.uint8)
        dilate_img = cv2.dilate(closing_a, kernel_d,iterations = 1)
        if plot_interm == True:
            plot_gray_img(dilate_img,str(img_idx)+"_dilate_img")
        
        num_cc,mat_cc = cv2.connectedComponents(dilate_img.astype(np.uint8))
        
        ################### step 2: shrink the embolism parts (more false negative) ###################
        #    erosion(e2_sz*e2_sz) and opening(o2_sz*o2_sz) and then closing(cl2_sz*cl2_sz) [closing2]
        #    --> keep contours with area > area_th2 and fill in the contour by polygons [contour_mask2]
        ################################################################################################
        kernel_e2 = np.ones((e2_sz,e2_sz),np.uint8) #square image kernel used for erosion
        erosion2 = cv2.erode(filter_img, kernel_e2,iterations = 1) #refines all edges in the binary image
        if plot_interm == True:
            plot_gray_img(erosion2, str(img_idx) + "_erosion2")
        
        kernel_o2 = np.ones((o2_sz,o2_sz),np.uint8)
        #https://homepages.inf.ed.ac.uk/rbf/HIPR2/open.htm
        opening = cv2.morphologyEx(erosion2.astype(np.uint8), cv2.MORPH_OPEN, kernel_o2)
        #tends to remove some of the foreground (bright) pixels from the edges of regions of foreground pixels.
        #However it is less destructive than erosion in general.
        
        if plot_interm == True:
            plot_gray_img(opening, str(img_idx) + "_opening2")
        
        #the closing here is needed for cases like ALCLAT1_leaf Subset: img_idx = 57
        #where embolism is break into small chunks and disappear mostly after opening
        kernel_cl2 = np.ones((cl2_sz,cl2_sz),np.uint8)
        closing2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_cl2)
        if plot_interm == True:
            plot_gray_img(closing2,str(img_idx)+"_closing2")

        #find contours with simple approximation
        contours2, hierarchy2 = cv2.findContours(closing2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        
        #list to hold all areas
        areas2 = [cv2.contourArea(contour2) for contour2 in contours2]
         
        #remove particles with area < area_th2
        area_index2 = [i for i,val in enumerate(areas2) if val >= area_th2]#extract the index in areas with areas>=area_th
        #enumerate is used in case there are multiply indices with the same areas value
        
        if len(area_index2) == 0:#no particles with area >= area_th
            return(bin_stack[img_idx,:,:]*0)#just return no embolism
        else:
            cnt2 = operator.itemgetter(*area_index2)(contours2)
            #cnt = [contours[area_index]] #only useful when area_index only has one element
            
            closing_contour2 = cv2.fillPoly(closing2, cnt2, 125)
            #includes closing(0 or 255) and areas in cnt(125)
            contour_mask2 = (closing_contour2==125)# = True if px is in cnt
            if plot_interm == True:
                plot_gray_img(contour_mask2,str(img_idx)+"_contour_mask2")
            
            ################### step 3: pixel values from step2 correspond to which connected component from step1 ###################
            #    for a pixel w/ value = 1 in [contour_mask2],
            #    --> the intersection of the connected component [mat_cc] where that pixel falls into and 
            #        binarized image [bin_stack] is treated as an candidate for embolism [add_part]
            #    -->if (the area of [add_part])/(the area of intersection of connected component and [contour_mask2]) < ratio_th,
            #        [add_part] is an embolism
            #        (this condition is needed to avoid the case that a small noise in [contour_mask] leads to a really big [add_part])
            ###########################################################################################################################
            
            intersect_img = mat_cc*contour_mask2 #intersect connected component mat with bin_contour_img2
            unique_cc_label = np.unique(intersect_img)#see which connected component should be kept
            final_mask = bin_stack[img_idx,:,:]*0
            if unique_cc_label.size > 1: #not only backgroung
                for cc_label in unique_cc_label[1:]: #starts with "1", cuz "0" is for bkg
                    add_part = (mat_cc == cc_label)*bin_stack[img_idx,:,:]
                    area_cm2 = np.sum((mat_cc == cc_label)*contour_mask2)#number of pixels in contour_mask2 that are 1
                    if plot_interm == True:
                        #print("number of pixel added/area of that connected component in dilate_img",np.sum(add_part)/np.sum(mat_cc == cc_label))
                        print(str(img_idx)+":"+str(np.sum(add_part)/area_cm2))
                    if np.sum(add_part)/area_cm2 < ratio_th:# and np.sum(add_part)/np.sum(mat_cc == cc_label)>0.19:
                        '''
                        1st condition
                        avoid the case that a small noise in contour_mask leads to a really big "add_part"
                        that are essentially just noises
                        case: ALCLAT1_leaf Subset: img_idx = 190
                        2nd condition
                        crude way of discarding parts with low density
                        '''
                        final_mask = final_mask + add_part
                    
            final_img = np.copy(final_mask)
            if plot_interm == True:
                plot_gray_img(final_mask,str(img_idx)+"_final_mask")
                print(img_idx,np.sum(final_img)/stem_area,np.sum(final_img))
            
            #####################################  step 4: reduce false positive ########################## 
            #    (image level)
            #    if the percentage of embolism in the [stem_area] is too large (probably it's because of shifting of images) 
            #        OR the embolism area is too small
            #    --> treat as if there's no embolism in the entire image
            #    (connected component level)
            #    Discard connected components w/small density or area
            ################################################################################################
            if np.sum(final_img)/stem_area > shift_th:
                '''
                percentage of embolism in the stem_area is too large
                probably is false positive due to shifting of img (case: stem img_idx=66,67)
                '''
                if plot_interm==True:
                    print("percentage of embolism in the stem_area is too large",np.sum(final_img)/stem_area,"> shift_th =",shift_th)
                final_img = final_img * 0
                return(final_img)

            if np.sum(final_img) < final_area_th:
                '''
                remove false positive with really small sizes
                '''
                if plot_interm==True:
                    print("embolism area is too small", np.sum(final_img),"< final_area_th =", final_area_th)
                final_img = final_img * 0
                return(final_img)
            
            
            #reduce false positive, discard those with small density
            final_img_cp = np.copy(final_img)#w/o np.copy, changes made in final_img_cp will affect final_img
            closing_3 = cv2.morphologyEx(final_img_cp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
            if plot_interm == True:
                plot_gray_img(closing_3,str(img_idx)+"_closing_3")
            
            final_mask3 = bin_stack[img_idx,:,:]*0
            num_cc3,mat_cc3 = cv2.connectedComponents(closing_3.astype(np.uint8))
            unique_cc_label3 = np.unique(mat_cc3)
            if unique_cc_label3.size > 1: #not only background
                for cc_label3 in unique_cc_label3[1:]: #starts with "1", cuz "0" is for bkg
                    inside_cc3 = (mat_cc3 == cc_label3)*bin_stack[img_idx,:,:]
                    if plot_interm == True:
                        print("estimated of density",np.sum(inside_cc3)/np.sum(mat_cc3 == cc_label3))
                        print("estimated number of pixels inside",np.sum(inside_cc3))
                    #print(str(img_idx)+":"+str(np.sum(add_part)/area_cm2))
                    if np.sum(inside_cc3)/np.sum(mat_cc3 == cc_label3)>density_th and np.sum(inside_cc3)>num_px_th :
                        #discarding parts with low density
                        #discarding parts with small area ( ~ < 50 pixels) #!!! threshold for small area: depends on the distance btw the camera and the stem 
                            
                        final_mask3 = final_mask3 + inside_cc3
            if plot_interm == True:
                plot_gray_img(final_mask3,str(img_idx)+"_final_mask3")
            final_img = final_mask3
            
            if plot_interm == True:
                plot_gray_img(final_img,str(img_idx)+"_final_img")
            return(final_img*255)


#not used here, but might be useful in the future?
def add_img_info_to_img(img_idx,img_stack,img_paths,img_folder):
    one_img = Image.fromarray(img_stack[img_idx,:,:])
    draw = ImageDraw.Draw(one_img)
    font = ImageFont.truetype("arial.ttf", 40)#45:font size
    font_small = ImageFont.truetype("arial.ttf", 28)
    draw.text((10,10), "Image "+str(img_idx+1)+" - "+str(img_idx+2), font=font)#, fill=(125))#text color
    draw.text((10,70),os.path.split(img_paths[img_idx])[1],font=font_small)
    draw.text((100,100),"to",font=font_small)
    draw.text((10,130),os.path.split(img_paths[img_idx+1])[1],font=font_small)
    one_img.save(img_folder +'/'+str(i)+'.jpg')

#add_img_info_to_img(231,final_combined_inv)
    
def add_img_info_to_stack(img_stack,img_paths,start_img_idx):    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,50)
    fontScale              = 1
    fontColor              = 0
    lineType               = 2
    
    stack_cp = np.copy(img_stack)#make a copy s.t. changes would be made on stack_cp instead of img_stack
    
    for img_idx in range(0,img_stack.shape[0]):
        one_img_arr = stack_cp[img_idx,:,:]
        img_ori_idx = img_idx+(start_img_idx-1)
        cv2.putText(one_img_arr,"Image "+str(img_ori_idx+1)+" - "+str(img_ori_idx+2),bottomLeftCornerOfText, font, 
            fontScale,fontColor,lineType)
        cv2.putText(one_img_arr,os.path.split(img_paths[img_ori_idx])[1],(10,90), font, 
            0.7,fontColor,lineType)
        cv2.putText(one_img_arr,"to",(100,130), font, 
            0.7,fontColor,lineType)
        cv2.putText(one_img_arr,os.path.split(img_paths[img_ori_idx+1])[1],(10,170), font, 
            0.7,fontColor,lineType)
        #cv2.imwrite(img_folder +'/out.jpg', one_img_arr)
    return(stack_cp)


#############################################################################
#    Confusion Matrix
#    To see the performance compared to those processed manually using ImageJ
############################################################################# 

def img_contain_emb(img_stack):
    return( np.any(np.any(img_stack,axis=2),axis=1)*1)#0 or 1

def confusion_mat_cluster(pred_stack, true_stack, has_embolism:list, true_has_emb: list, blur_radius: float) -> list :
    '''
    we need to check the blur radius need to be set different for different situation or not.
    confusion matrix at cluster level
    '''
    t_emb_img_n = [ i for i, value in enumerate(true_has_emb) if value == 1]
    pred_emb_img_n = [ i for i, value in enumerate(has_embolism) if value == 1]
    f_pos = 0
    f_neg = 0
    t_pos = 0
    t_neg = 0
    
    #NOT SURE: how do you define true negative at cluster level? same as it at img_level?
    tn_idx = np.where((has_embolism==true_has_emb)*(true_has_emb==0))[0]
    t_neg = len(tn_idx)
    
    #find all false postive img number and process all the clusters -> will be false positive cluster
    #need to check pred_emb_img_n is always bigger or not.
    #look at false positive imgs at images level (i.e. every img predicted w/emb, but true has no emb)
    diff_img_list = list(set(pred_emb_img_n) - set(t_emb_img_n))
    for img_num in diff_img_list:
        img_2d_fp = pred_stack[img_num,:,:]
        #clustering process
        smooth_fp_img = ndimage.gaussian_filter(img_2d_fp, sigma = blur_radius)
        num_cc_fp, mat_cc_fp = cv2.connectedComponents(smooth_fp_img.astype(np.uint8))
        #add up the false postive cluster
        f_pos += num_cc_fp - 1
    
    #look at true positive at images level
    #true_embolism_img, and predicted connected component
    for index in t_emb_img_n:
        img_2d_tp = true_stack[index,:,:]
        img_2d_prd_p = pred_stack[index,:,:]
        #smoothing (gaussian filter)
        smooth_tp_img = ndimage.gaussian_filter(img_2d_tp, sigma = blur_radius)
        #need to binary to this 
        smooth_pred_p_img = ndimage.gaussian_filter(img_2d_prd_p, sigma = blur_radius)
        #cc pred_p_img
        num_cc_pred_p, mat_cc_pred_p = cv2.connectedComponents(smooth_pred_p_img.astype(np.uint8))
        #clustering tp
        num_cc_tp, mat_cc_tp = cv2.connectedComponents(smooth_tp_img.astype(np.uint8))
        #plot_gray_img(mat_cc_pred_p>0)
        #plot_gray_img(mat_cc_tp>0)
        #binary the labelled cluster
        lab_cl_bin_1 = to_binary(mat_cc_pred_p)
        lab_cl_bin_2 = lab_cl_bin_1 * mat_cc_tp
        num_cc_lcb2 = len(np.unique(lab_cl_bin_2))
        
        
        if num_cc_pred_p-num_cc_tp>0:
            f_pos += num_cc_pred_p-num_cc_tp
        if num_cc_lcb2 >1: # not just background
            t_pos += num_cc_lcb2 -1
        if num_cc_tp-num_cc_lcb2 >0:
            f_neg += num_cc_tp-num_cc_lcb2 
            
    con_mat = np.ndarray((2,2), dtype=np.float32)
    column_names = ['Predict 0', 'Predict 1']
    row_names    = ['True 0','True 1']
    con_mat[0,0] = t_neg
    con_mat[1,1] = t_pos
    con_mat[0,1] = f_pos
    con_mat[1,0] = f_neg
    con_df = pd.DataFrame(con_mat, columns=column_names, index=row_names)
    return(con_df)


def confusion_mat_img(has_embolism,true_has_emb):
    #false positive
    fp_idx = np.where(has_embolism>true_has_emb)[0]
    con_fp = len(fp_idx)
    #false negative
    fn_idx = np.where(has_embolism<true_has_emb)[0]
    con_fn = len(fn_idx)
    #true positive
    tp_idx = np.where((has_embolism==true_has_emb)*(true_has_emb==1))[0]
    con_tp = len(tp_idx)
    #true negative
    tn_idx = np.where((has_embolism==true_has_emb)*(true_has_emb==0))[0]
    con_tn = len(tn_idx)
    
    con_mat = np.ndarray((2,2), dtype=np.float32)
    column_names = ['Predict 0', 'Predict 1']
    row_names    = ['True 0','True 1']
    con_mat[0,0] = con_tn
    con_mat[1,1] = con_tp
    con_mat[0,1] = con_fp
    con_mat[1,0] = con_fn
    con_df = pd.DataFrame(con_mat, columns=column_names, index=row_names)
    return([con_df,fp_idx,fn_idx,tp_idx,tn_idx])


def confusion_mat_pixel(pred_stack,true_stack):
    #confusion matrix(pixel level)
    #positive (true_stack > 0): embolism; 
    #negative (true_stack == 0)
    #false positive
    con_fp = np.sum(np.sum(np.sum((pred_stack > true_stack))))
    #false negative
    con_fn = np.sum(np.sum(np.sum((pred_stack < true_stack))))
    #true positive
    con_tp = np.sum(np.sum(np.sum((pred_stack == true_stack)*(true_stack > 0))))
    #true negative
    con_tn = np.sum(np.sum(np.sum((pred_stack == true_stack)*(true_stack == 0))))
    
    con_mat = np.ndarray((2,2), dtype=np.float32)
    column_names = ['Predict 0', 'Predict 1']
    row_names    = ['True 0','True 1']
    con_mat[0,0] = con_tn
    con_mat[1,1] = con_tp
    con_mat[0,1] = con_fp
    con_mat[1,0] = con_fn
    con_df = pd.DataFrame(con_mat, columns=column_names, index=row_names)
    return(con_df)

def calc_metric(con_mat):
    t_neg = con_mat['Predict 0']['True 0']
    t_pos = con_mat['Predict 1']['True 1']
    f_pos = con_mat['Predict 1']['True 0']
    f_neg = con_mat['Predict 0']['True 1']
    
    try: 
        sens = round(t_pos/(t_pos+f_neg)*100.0,2)#want sensitivity to be high (want fn to be low)
    except ZeroDivisionError:
        print(f'sens denominator equals to {t_pos+f_neg}')
    try:
        prec = round(t_pos/(t_pos+f_pos)*100,2)#want precision to be high(fp small --> less labor afterwards)
    except ZeroDivisionError:
        print(f'prec denominator equals to {t_pos+f_pos}')
    try:
        acc = round((t_pos+t_neg)/(t_neg+t_pos+f_pos+f_neg)*100,2)#accuracy
    except ZeroDivisionError:
        print(f'acc denominator equals to {t_neg+t_pos+f_pos+f_neg}')
    return([('sensitivity',sens),('precision',prec),('accuracy',acc)])