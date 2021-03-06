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
import seaborn as sns
import datetime
import scipy.signal
from unidip import UniDip#for unimodality_dip_test
import unidip.dip as dip#for unimodality_dip_test
#############################################################################
#    Sanity check:
#    see if the sum of pixel values in an image of diff_pos_stack
#    is similar to those from true_diff_sum (the intermediate result they obtained from ImageJ)
#############################################################################
def plot_img_sum(img_3d,title_str,img_folder,is_save=False):
    sum_of_img = np.sum(np.sum(img_3d,axis=2),axis=1)#sum
    fig = plt.figure()
    plt.plot(range(len(sum_of_img)),sum_of_img)
    plt.ylabel("sum of pixel values in an image")
    plt.xlabel("image index")
    plt.title(title_str)
    if is_save==True:
        plt.savefig(img_folder+'/m0_'+title_str+'.jpg',bbox_inches='tight')
    plt.close(fig)
    # x-axis= image index. y-axis=sum of each img
    return(sum_of_img)

#############################################################################
#    function for plotting overlap sum
#############################################################################
def plot_overlap_sum(is_stem_mat, title_str ,chunk_folder, is_save = False):
    #title_str = img_folder_name
    sum_is_stem_mat = is_stem_mat[1:,:,:] + is_stem_mat[:-1,:,:]
    prop = np.sum(np.sum((sum_is_stem_mat==2),2),1)/np.sum(np.sum((sum_is_stem_mat>=1),2),1)
    #AND(=overlap)/OR --> if not close to 1 --> shifting 
    fig = plt.figure()
    plt.plot(range(len(prop)),prop)
    plt.ylabel("portion of overlap area")
    plt.xlabel("image relative index")
    plt.title(title_str)

    if is_save == True:
        plt.savefig(chunk_folder + "/m1_overlap_area_hist_img.jpg",bbox_inches='tight')
    plt.close(fig)
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
    return((img_stack > 0)*1.0)

#############################################################################
#    Foreground Background Segmentation
#    for stem only, cuz there are clearly parts that are background (not stem)
#    want to avoid  artifacts from shrinking of gels/ moving of plastic cover
#    mask only the stem part to reduce false positive
#############################################################################

def extract_foreground(img_2d, chunk_folder, blur_radius=10.0,fg_threshold=0.1,expand_radius_ratio=5,is_save=False):
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

def extract_foregroundRGB(img_2d,img_re_idx,chunk_folder, blur_radius=10.0,expand_radius_ratio=3,is_save=False,use_max_area=True):
    '''
    assume stem is more green than backgorund
    '''
    
    #smoothing (gaussian filter)
    smooth_img = ndimage.gaussian_filter(img_2d, blur_radius)
    
    is_stem_mat = (smooth_img > np.mean(img_2d))*1 #why not change it to np.mean(smooth_img), cuz np.mean(img_2d) seems slightly bigger than that of smooth_img?
    
    #plot_gray_img(is_stem_mat)#1(white) for stem part
    if use_max_area:
        if is_save==True:
            plt.imsave(chunk_folder + "/s_"+str(img_re_idx)+"_G_2_1_is_stem_mat_before_max_area.jpg",is_stem_mat,cmap='gray')
    
        #doesn't work for unproc Alclat3_stem
        num_cc, mat_cc, stats, centroids  = cv2.connectedComponentsWithStats(is_stem_mat.astype(np.uint8), 8)#8-connectivity
        
        if num_cc>1:#more than 1 area, don't count bgd
            area = stats[1:, cv2.CC_STAT_AREA]
            max_cc_label = np.where(area==max(area))[0]+1#+1 cuz we exclude 0 in previous line
            is_stem_mat = (mat_cc==max_cc_label)*1
        else:#no part is being selected as stem --> treat entire img as stem
            is_stem_mat = is_stem_mat+1

        
#    num_cc_stem, mat_cc_stem = cv2.connectedComponents(is_stem_mat.astype(np.uint8))
#    unique_cc_stem_label = np.unique(mat_cc_stem) 
#    #list of areas
#    area = []
#    if unique_cc_stem_label.size > 1: #more than 1 area 
#        for cc_label_stem in unique_cc_stem_label[1:]:
#            area.append(np.sum(mat_cc_stem == cc_label_stem))
#    #real stem part
#    if area:
#        max_area = max(area)
#        for cc_label_stem in unique_cc_stem_label[1:]:
#            if np.sum(mat_cc_stem == cc_label_stem) < max_area:
#                mat_cc_stem[mat_cc_stem == cc_label_stem] = 0
#        is_stem_mat = is_stem_mat * mat_cc_stem
#    else:#no part is being selected as stem --> treat entire img as stem
#        is_stem_mat = is_stem_mat+1
    
    #expand the stem part a bit by shrinking the not_stem
    not_stem = -is_stem_mat+1
    unif_radius = blur_radius*expand_radius_ratio
    not_stem_exp =  to_binary(ndimage.uniform_filter(not_stem, size=unif_radius))
    
    is_stem_mat = (not_stem_exp==0)#invert
    
    #plot_gray_img(is_stem+not_stem)#expansion
    if is_save==True:
        plt.imsave(chunk_folder+"/s_"+str(img_re_idx)+"_G_2_expansion_foreground.jpg",is_stem_mat+not_stem,cmap='gray')
    #plot_gray_img(is_stem)#1(white) for stem part
    if is_save==True:
        plot_gray_img(is_stem_mat)#1(white) for stem part
        plt.imsave(chunk_folder + "/s_"+str(img_re_idx)+"_G_3_is_stem_matG.jpg",is_stem_mat,cmap='gray')
    return(is_stem_mat)#logical 2D array

def unimodality_dip_test(path:str, plot_show = False) -> bool:
    '''
    #http://www.nicprice.net/diptest/Hartigan_1985_AnnalStat.pdf
    #https://github.com/BenjaminDoran/unidip
    Given the image and conduct dip test to see whether it's unimodal or not.
    @path: image path
    @plot_show: see whether plot the histogram or not
    '''
    img = cv2.imread(path,0)
    img_array = img.ravel() 
    #input an array
    #return True if its unimodal distributed
    data = np.msort(img_array)
    #the probability of unimodal
    uni_prob = dip.diptst(data)[1]
    if uni_prob > 0.5:
        print(f'This image is unimodel distributed with probability of {uni_prob*100:.2f} %')
        unimodality = True
    else:
        print(f'This image is at least bimodel distributed with probability of {(1-uni_prob)*100:.2f} %')
        unimodality = False
    if plot_show:
        plt.figure()
        sns.distplot(img.ravel(), bins=256,kde= True, hist = True)
        plt.title('Histogram of the image')
        plt.show()
    return unimodality
    
def foregound_Th_OTSU(img_array, img_re_idx, chunk_folder, unif_radius=20, is_save = False, use_max_area = True):
    '''
    Given the raw img_array and used THRESH+OTSU to segement the foreground and used cc to get the biggest area
    @based on the knowledge that the img is bimodal image (which histogram have 2 peaks)
    '''
    #unif_radius=15 is too small for c2.2 img_idx=188,190 (emb near stem boundary), so set unif_radius to 20
    #convert color img to grayscale
    gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    #apply guassian filter to blue the egdes
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #apply OSTU threshold to segement the foreground object
    ret, is_stem_mat = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #BINARY_INV CAN BE ANTOHER OPTION
    if use_max_area:
        if is_save==True:
            plt.imsave(chunk_folder + "/s_"+str(img_re_idx)+"_OTSU_2_1_is_stem_mat_before_max_area.jpg",is_stem_mat,cmap='gray')
        num_cc, mat_cc, stats, centroids  = cv2.connectedComponentsWithStats(is_stem_mat.astype(np.uint8), 8)
        if num_cc > 1:
            area = stats[1:, cv2.CC_STAT_AREA]
            max_cc_label = np.where(area==max(area))[0]+1#+1 cuz we exclude 0 in previous line
            is_stem_mat = (mat_cc==max_cc_label)*1
        else:#no part is being selected as stem --> treat entire img as stem
            is_stem_mat = is_stem_mat + 1
        #expand the stem part a bit by shrinking the not_stem
        not_stem = -is_stem_mat+1
        not_stem_exp = to_binary(ndimage.uniform_filter(not_stem, size=unif_radius))
        is_stem_mat = (not_stem_exp==0)#invert
        #Another CC because (inglau 1) is seperate from tree bark, and need anthor seperation to take care of the loose part
        num_cc, mat_cc, stats, centroids  = cv2.connectedComponentsWithStats(is_stem_mat.astype(np.uint8), 8)
        if num_cc > 1:
            area = stats[1:, cv2.CC_STAT_AREA]
            max_cc_label = np.where(area==max(area))[0]+1#+1 cuz we exclude 0 in previous line
            is_stem_mat = (mat_cc==max_cc_label)*1
        else:#no part is being selected as stem --> treat entire img as stem
            is_stem_mat = is_stem_mat + 1
        #expand the stem part a bit by shrinking the not_stem
        not_stem = -is_stem_mat+1
        not_stem_exp = to_binary(ndimage.uniform_filter(not_stem, size=unif_radius))
        is_stem_mat = (not_stem_exp==0)#invert

     #plot_gray_img(is_stem+not_stem)#expansion
    if is_save==True:
        plt.imsave(chunk_folder+"/s_"+str(img_re_idx)+"_OTSU_2_expansion_foreground.jpg",is_stem_mat+not_stem,cmap='gray')
    #plot_gray_img(is_stem)#1(white) for stem part
    if is_save==True:
        plot_gray_img(is_stem_mat)#1(white) for stem part
        plt.imsave(chunk_folder + "/s_"+str(img_re_idx)+"_OTSU_3_is_stem_mat.jpg",is_stem_mat,cmap='gray')
    return is_stem_mat*1

def foreground_B(img_2d,img_nrow,img_re_idx,chunk_folder,quan_th=0.9, G_max = 160,blur_radius=10.0,expand_radius_ratio=9,is_save=False,use_max_area=True):
    '''
    assume stem is more blue than bark (i.e. stem is whiter than bark)
    G_max is introduced because of in3_stem
    assume height of stem > img_nrow/2
    '''
    
    #smoothing (gaussian filter)
    smooth_img = ndimage.gaussian_filter(img_2d, blur_radius)
   
    is_stem_mat = (smooth_img > min(np.quantile(smooth_img,quan_th),G_max))*1
    
    #plot_gray_img(is_stem_mat)#1(white) for stem part
    if use_max_area:
        if is_save==True:
            plt.imsave(chunk_folder + "/s_"+str(img_re_idx)+"_B_2_1_is_stem_mat_before_max_area.jpg",is_stem_mat,cmap='gray')
        
        num_cc, mat_cc, stats, centroids  = cv2.connectedComponentsWithStats(is_stem_mat.astype(np.uint8), 8)#8-connectivity
        
        if num_cc>1:#more than 1 area, don't count bgd
            area = stats[1:, cv2.CC_STAT_AREA]
            cc_h = stats[1:,cv2.CC_STAT_HEIGHT]
            cc_valid_geo = 1*(cc_h>img_nrow/2)#assume stem is at least half as tall as img_nrow
            #(cas5_stem img_idx>1200)
            area_valid = area*cc_valid_geo#map invalid one's area to 0
            max_cc_label = np.where(area_valid==max(area_valid))[0]+1#+1 cuz we exclude 0 in previous line
            is_stem_mat = (mat_cc==max_cc_label)*1
        else:#no part is being selected as stem --> treat entire img as stem
            is_stem_mat = is_stem_mat+1
    
    #expand the stem part a bit by shrinking the not_stem
    not_stem = -is_stem_mat+1
    unif_radius = blur_radius*expand_radius_ratio
    not_stem_exp =  to_binary(ndimage.uniform_filter(not_stem, size=unif_radius))
    
    is_stem_mat = (not_stem_exp==0)#invert
    
    #plot_gray_img(is_stem+not_stem)#expansion
    if is_save==True:
        plt.imsave(chunk_folder+"/s_"+str(img_re_idx)+"_B_2_foreground_B.jpg",is_stem_mat+not_stem,cmap='gray')
    #plot_gray_img(is_stem)#1(white) for stem part
    if is_save==True:
        plot_gray_img(is_stem_mat)#1(white) for stem part
        plt.imsave(chunk_folder + "/s_"+str(img_re_idx)+"_B_3_is_stem_matB.jpg",is_stem_mat,cmap='gray')
    return(is_stem_mat)#logical 2D array

def corr_image(im1_gray, im2_gray,plot_interm=False):
    '''
    detect a shift using correlation btw 2 consecutive imgs
    #https://stackoverflow.com/questions/24768222/how-to-detect-a-shift-between-images
    '''
    
    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)
   
    corr_img = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')
    # calculate the correlation image; note the flipping of onw of the images
    if plot_interm==True:
        plot_gray_img(corr_img)
    
    pos_bright = np.unravel_index(np.argmax(corr_img), corr_img.shape)#The brightest spot position
    
    ori_center_y = round(im1_gray.shape[0]/2)
    ori_center_x = round(im1_gray.shape[1]/2)
    shift_down = ori_center_y - pos_bright[0]
    shift_right = ori_center_x - pos_bright[1]
    return shift_down,shift_right
#############################################################################
#    main function for detecting embolism
############################################################################# 
def find_emoblism_by_contour(bin_stack,img_idx,stem_area,final_area_th = 20000/255,area_th=30, area_th2=30,ratio_th=5,e2_sz=3,o2_sz=1,cl2_sz=3,plot_interm=False,max_emb_prop=0.05,density_th=0.4,num_px_th=50):
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
            if np.sum(final_img)/stem_area > max_emb_prop:
                '''
                percentage of embolism in the stem_area is too large
                probably is false positive due to shifting of img (case: stem img_idx=66,67)
                '''
                if plot_interm==True:
                    print("percentage of embolism in the stem_area is too large",np.sum(final_img)/stem_area,"> max_emb_prop =",max_emb_prop)
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
            
def find_emoblism_by_filter_contour(bin_stack,filter_stack,img_idx,stem_area,final_area_th = 20000/255,area_th=30, area_th2=30,ratio_th=5,c1_sz=25,d1_sz=10,e2_sz=3,o2_sz=1,cl2_sz=3,plot_interm=False,max_emb_prop=0.05,density_th=0.4,num_px_th=50,resize=False):
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
    if resize:
        kernel = np.ones((3,3),np.uint8)#[Dingyi]
    else:
        kernel = np.ones((2,2),np.uint8) #square image kernel used for erosion
    #erosion = cv2.erode(bin_stack[img_idx,:,:].astype(np.uint8), kernel,iterations = 1) #refines all edges in the binary image
    opening1 = cv2.morphologyEx(filter_img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if plot_interm == True:
        #plot_gray_img(erosion,str(img_idx)+"_erosion")
        plot_gray_img(opening1,str(img_idx)+"_opening1")
    
    if resize:
        kernel_cl = np.ones((4,4),np.uint8)
    else:
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
                        if plot_interm == True:
                            print("[keep] ratio is small enough",np.sum(add_part)/area_cm2," < ratio_th=",ratio_th)
                    else:
                        if plot_interm == True:
                            print("[discard] ratio is too big",np.sum(add_part)/area_cm2," >= ratio_th=",ratio_th)
                    
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
            if np.sum(final_img)/stem_area > max_emb_prop:
                '''
                percentage of embolism in the stem_area is too large
                probably is false positive due to shifting of img (case: stem img_idx=66,67)
                '''
                if plot_interm==True:
                    print("percentage of embolism in the stem_area is too large",np.sum(final_img)/stem_area,"> max_emb_prop =",max_emb_prop)
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
                    #print(str(img_idx)+":"+str(np.sum(add_part)/area_cm2))
                    if np.sum(inside_cc3)/np.sum(mat_cc3 == cc_label3)<=density_th:
                        #discarding parts with low density
                        if plot_interm==True:
                            print("[discard] estimated of density",np.sum(inside_cc3)/np.sum(mat_cc3 == cc_label3),"< density_th=",density_th)
                    elif np.sum(inside_cc3)<=num_px_th :
                        #discarding parts with small area ( ~ < 50 pixels) #!!! threshold for small area: depends on the distance btw the camera and the stem 
                        if plot_interm==True:
                            print("[discard] estimated number of pixels inside",np.sum(inside_cc3),"<=",num_px_th)
                    else:
                        if plot_interm==True:
                            print("[keep] estimated of density",np.sum(inside_cc3)/np.sum(mat_cc3 == cc_label3),"estimated number of pixels inside",np.sum(inside_cc3))
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

def confusion_mat_cluster(pred_stack, true_stack, has_embolism:list, true_has_emb: list, blur_radius: float, chunk_folder = "", is_save=False) -> list :
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
    
    tp_area = []
    tp_height = []
    tp_width=[]
    fp_area = []
    fp_height = []
    fp_width=[]
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
        
        num_cc, mat_cc, stats, centroids  = cv2.connectedComponentsWithStats(smooth_fp_img.astype(np.uint8), 8)#8-connectivity
        
        cc_width = stats[1:,cv2.CC_STAT_WIDTH]#"1:", ignore bgd:0
        cc_height = stats[1:,cv2.CC_STAT_HEIGHT]
        cc_area = stats[1:, cv2.CC_STAT_AREA]
        
        fp_area.append(cc_area[0])
        fp_height.append(cc_height[0])
        fp_width.append(cc_width[0])
        
        f_pos += num_cc - 1
        
#        num_cc_fp, mat_cc_fp = cv2.connectedComponents(smooth_fp_img.astype(np.uint8))
#        #add up the false postive cluster
#        f_pos += num_cc_fp - 1
#        unique_mat_cc_fp_label = np.unique(mat_cc_fp) 
#        if unique_mat_cc_fp_label.size > 1: #more than 1 area 
#            for cc_fp_label_stem in unique_mat_cc_fp_label[1:]:
#                fp_area.append(np.sum(mat_cc_fp == cc_fp_label_stem))
    
    #look at true positive at images level
    #true_embolism_img, and predicted connected component
    for index in t_emb_img_n:
        img_2d_tp = true_stack[index,:,:]
        img_2d_prd_p = pred_stack[index,:,:]
        #smoothing (gaussian filter)
        smooth_tp_img = ndimage.gaussian_filter(img_2d_tp, sigma = blur_radius)
        #need to binary to this 
        smooth_pred_p_img = ndimage.gaussian_filter(img_2d_prd_p, sigma = blur_radius)
        
#        #cc pred_p_img
#        num_cc_pred_p, mat_cc_pred_p = cv2.connectedComponents(smooth_pred_p_img.astype(np.uint8))
#        #clustering tp
#        num_cc_tp, mat_cc_tp = cv2.connectedComponents(smooth_tp_img.astype(np.uint8))
#        #plot_gray_img(mat_cc_pred_p>0)
#        #plot_gray_img(mat_cc_tp>0)
#        #binary the labelled cluster
#        lab_cl_bin_1 = to_binary(mat_cc_pred_p)
#        lab_cl_bin_2 = lab_cl_bin_1 * mat_cc_tp
#        num_cc_lcb2 = len(np.unique(lab_cl_bin_2))
#        unique_mat_cc_tp_label = np.unique(mat_cc_tp) 
#        if unique_mat_cc_tp_label.size > 1: #more than 1 area 
#            for cc_tp_label_stem in unique_mat_cc_tp_label[1:]:
#                tp_area.append(np.sum(mat_cc_tp == cc_tp_label_stem))  
        
        
        num_cc_p, mat_cc_p, stats_p, centroids_p  = cv2.connectedComponentsWithStats(smooth_pred_p_img.astype(np.uint8), 8)#8-connectivity
        num_cc_t, mat_cc_t, stats_t, centroids_t  = cv2.connectedComponentsWithStats(smooth_tp_img.astype(np.uint8), 8)
        
        lab_cl_bin_1 = to_binary(mat_cc_p)
        lab_cl_bin_2 = lab_cl_bin_1 * mat_cc_t
        num_cc_lcb2 = len(np.unique(lab_cl_bin_2))
        
        cc_width = stats_t[1:,cv2.CC_STAT_WIDTH]#"1:", ignore bgd:0
        cc_height = stats_t[1:,cv2.CC_STAT_HEIGHT]
        cc_area = stats_t[1:, cv2.CC_STAT_AREA]
        
        tp_area.append(cc_area[0])
        tp_height.append(cc_height[0])
        tp_width.append(cc_width[0])
        
        if num_cc_p-num_cc_t>0:
            f_pos += num_cc_p-num_cc_t
        if num_cc_lcb2 >1: # not just background
            t_pos += num_cc_lcb2 -1
        if num_cc_t-num_cc_lcb2 >0:
            f_neg += num_cc_t-num_cc_lcb2 
        
#        if num_cc_pred_p-num_cc_tp>0:
#            f_pos += num_cc_pred_p-num_cc_tp
#        if num_cc_lcb2 >1: # not just background
#            t_pos += num_cc_lcb2 -1
#        if num_cc_tp-num_cc_lcb2 >0:
#            f_neg += num_cc_tp-num_cc_lcb2 
            
    con_mat = np.ndarray((2,2), dtype=np.float32)
    column_names = ['Predict 0', 'Predict 1']
    row_names    = ['True 0','True 1']
    con_mat[0,0] = t_neg
    con_mat[1,1] = t_pos
    con_mat[0,1] = f_pos
    con_mat[1,0] = f_neg
    con_df = pd.DataFrame(con_mat, columns=column_names, index=row_names)
    
    #plot results
    fig = plt.figure()
    ax = sns.distplot(tp_area,label = 'True Positive',norm_hist=False,kde=False, bins=50)#assumes max(tp_area)>max(fp_area)?
    ax = sns.distplot(fp_area,label = 'False Positive',norm_hist=False,kde=False, bins=50)
    ax.set_title('Connected Component Area Histogram (TP vs FP)')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Area of a Connected Component')
    ax.legend()
    if is_save == True:
        plt.savefig(chunk_folder + '/m4_TP FP Histogram.jpg')
    plt.close(fig)
    
    fig = plt.figure()
    ax = sns.distplot(tp_area,label = 'True Positive',norm_hist=True,kde=False, bins=50)
    ax = sns.distplot(fp_area,label = 'False Positive',norm_hist=True,kde=False, bins=50)
    ax.set_title('Connected Component Area Histogram (TP vs FP)')
    ax.set_ylabel('Density')
    ax.set_xlabel('Area of a Connected Component')
    ax.legend()
    if is_save == True:
        plt.savefig(chunk_folder + '/m4_TP FP Histogram_Density.jpg')
    plt.close(fig)
        
    fig = plt.figure()
    ax = sns.distplot(tp_height,label = 'True Positive',norm_hist=True,kde=False, bins=50)
    ax = sns.distplot(fp_height,label = 'False Positive',norm_hist=True,kde=False, bins=50)
    ax.set_title('Connected Component Height Histogram (TP vs FP)')
    ax.set_ylabel('Density')
    ax.set_xlabel('Height of a Connected Component')
    ax.legend()
    if is_save == True:
        plt.savefig(chunk_folder + '/m4_Height Histogram_Density.jpg')
    plt.close(fig)
    
    fig = plt.figure()
    ax = sns.distplot(tp_width,label = 'True Positive',norm_hist=True,kde=False, bins=50)
    ax = sns.distplot(fp_width,label = 'False Positive',norm_hist=True,kde=False, bins=50)
    ax.set_title('Connected Component Width Histogram (TP vs FP)')
    ax.set_ylabel('Density')
    ax.set_xlabel('Width of a Connected Component')
    ax.legend()
    if is_save == True:
        plt.savefig(chunk_folder + '/m4_Width Histogram_Density.jpg')
    plt.close(fig)
    return(con_df,tp_area,fp_area,tp_height,fp_height,tp_width,fp_width)


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

def fill_holes(imInput, threshold):
    """
    The method used in this function is found from
    https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

    """

    # Threshold.
    th, thImg = cv2.threshold(imInput, threshold, 255, cv2.THRESH_BINARY_INV)
    #bubbles: white contour, bgd:black

    # Copy the thresholded image.
    imFloodfill = thImg.copy()

    # Get the mask.
    h, w = thImg.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill white from point (0, 0).
    cv2.floodFill(imFloodfill, mask, (0,0), 255)
    #space inside bubbles: black; bgd:white

    # Invert the floodfilled image.
    imFloodfillInv = 255-imFloodfill
    #space inside white: black; bgd:black

    # Combine the two images.
    imOut = to_binary(thImg + imFloodfillInv)

    return imOut

def detect_bubble(input_stack,blur_radius=11,hough_param1=25,hough_param2=10, minRadius = 0, maxRadius = 40, dp=1):
    '''
    bubble detection using opencv
    http://www.huyaoyu.com/technical/2017/12/13/bubble-detection-by-using-opencv.html
    original img in the above ref: bubbles:black, bgd: white
    '''
    bubble_stack = np.zeros(input_stack.shape)
    has_bubble_vec = np.zeros(input_stack.shape[0])
    for img_idx in range(0,input_stack.shape[0]):
        # Load the image.
        gray_img = input_stack[img_idx,:,:]
        #img_stack won't work: cuz stem itself would be considered as circles too
        #th_stack won't work: cuz it'll only focus on where the value is high, but later on we'll work w/ binary imgs, so a lot of bubbles w/low pixel value in th_stack are not discovered
#        blurred = cv2.GaussianBlur(gray_img.astype(np.uint8), (blur_radius, blur_radius), 0)
#        plot_gray_img(blurred,"[detect_bubble]: Blurred")
    
#        # Fille the "holes" on the image.
#        filled = fill_holes(blurred, np.quantile(blurred,0.8))
#        #doesn't seem to work in our case, because the edge of our circle isn't connected completely
#        # maybe can consider using it when gray_img only looks at is_stem_mat part 
#        plot_gray_img(filled,"[detect_bubble]: Filled")
    
        # Find circles by the Hough transformation.
        circles = cv2.HoughCircles(gray_img.astype(np.uint8), cv2.HOUGH_GRADIENT, dp=dp, minDist=10, param1 = hough_param1, param2 = hough_param2, minRadius = minRadius, maxRadius = maxRadius)
        #dp – Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
        #param1 – First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
        #param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. 
        #        The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
        #Usually the function detects the centers of circles well. However, it may fail to find correct radii. 
        
        output_img = np.zeros(gray_img.shape)
        # Draw circles on the original image.
        if circles is not None:
            for i in range(circles.shape[1]):
                c = circles[0, i]
    
                cv2.circle( output_img, center=(c[0], c[1]), radius=c[2], color=(255, 255, 255), thickness=-2)
                #negative thickness means filled circle
                #print("i = %d, r = %f" % (i, c[2]))
    
            #cv2.imshow("Marked", gray_img)
            #plot_gray_img(output_img)
            bubble_stack[img_idx,:,:]=output_img/255 #0:bgd or 1:bubbles
            has_bubble_vec[img_idx] = 1
#        else:
#            print(img_idx,": has no bubble")
        
    return bubble_stack,has_bubble_vec

def calc_bubble_area_prop(bubble_stack,is_stem_mat2,chunk_folder,is_save=False,plot_interm=True):
    '''
    calculate bubble area/stem area of each img
    '''
    bubble_area_prop_vec=np.sum(np.sum(bubble_stack,2),1)/np.sum(np.sum(is_stem_mat2,2),1)
    


    '''
    To decide bubble_area_prop_max--> considered as poor quality -->no emb:
    '''
    #from v9.4 hough_param2=10
    #bound for bubble_area_prop_max: foldername(chunk_size) / largest bubble_area has emb img_idx / smallest bubble_area no emb img_idx
    #>0.2,0.081, <0.31: cas5_stem(50) / 28 (not poor qual by eyes, but thought emb as bubble),1 / 8
    #>0.159, <0.268: cas2.2_stem(300) / 4(bubble_area_prop=1.025), 2nd largest bubble_area has emb img_idx =232 / 40
    #>0.059, <0.31: inglau4_stem(100) / 52 / 8
    #>0.003, --: inglau3_stem(200) / 177 / --
    #>0.024, <0.213,0.222,0.389: Alclat2_stem(400) / 324 / 0,1,54
    #look at max cc area of each img in bubble_stack: to separate cases for a2_stem and c5_stem: 
    #c2.2 img_idx=5,40
    
    #from v9.4 hough_param2=15, bubble_area_prop_max=0.1 
    #c5 >0.014, <0.173
    #c2.2 >0.159 (img 232), <0.268:
    num_bins=50
    fig = plt.figure()
    n_th, bins_th, patches_th= plt.hist(bubble_area_prop_vec,bins=num_bins)
    if plot_interm==False:
        plt.close(fig)
    
    #usually 1st bin is way TOO large --> don't show 1st bin(0)in hist for better visualization
    fig1 = plt.figure()
    plt.plot(bins_th[2:],n_th[1:])
    #plt.xlim(bins_th[1],bins_th[-1])
    plt.ylabel("frequency")
    plt.xlabel("bubble area")
    plt.title("histogram of bubble area (ignoring the 1st bin)")
    if is_save == True:
        plt.savefig(chunk_folder + '/m4.5_histogram of bubble area (ignoring the 1st bin).jpg')
    if plot_interm==False:
        plt.close(fig1)
    
    fig2 = plt.figure()
    plt.plot(range(len(bubble_area_prop_vec)),bubble_area_prop_vec)
    plt.ylabel("bubble area")
    plt.xlabel("image index")
    plt.title("bubble area in each image")
    if is_save == True:
        plt.savefig(chunk_folder + '/m4.5_bubble area in each image.jpg')
    if plot_interm==False:
        plt.close(fig2)
    return(bubble_area_prop_vec)

def max_cc_area(img):
    #assumes img has at least 2 connected components if including background
    #or else max() would run into problem
    _, _, stats, _  = cv2.connectedComponentsWithStats(img.astype(np.uint8), 8)#8-connectivity
    return(max(stats[1:, cv2.CC_STAT_AREA]))

def calc_bubble_cc_max_area_p(bubble_stack,is_stem_mat2,chunk_folder,is_save=False,plot_interm=True):
    '''
    calculate max(bubble connected component area)/stem area of each img
    '''
    #motivation: look at max cc area of each img in bubble_stack: to separate cases for a2_stem and c5_stem: 
    #c2.2 img_idx=5,40
    
    #from v9.5
    # c5(50) improved
    #bound for bubble_area_prop_max: foldername(chunk_size) / largest bubble_area has emb img_idx / smallest bubble_area no emb img_idx
    #>0.077, <0.23: cas5_stem(50) / 24 / 2
    #>0.019, <0.2 Alclat2_stem(400) / 364 / 0 (should decrease theshold s.t. img_idx=1 --> shift )
    #>0.073, <0.247 cas2.2_stem(300) / 232 / 9 (ignore img_idx=4: true_emb in poor qual. <0.149: img_idx=9, poor qual)
    #>0.003, --: inglau3_stem(200) / 177 / -- (exactly same as before cuz everything looks nice after 1st stage :) )
    #>0.041,<0.212: inglau4_stem(100) / 52 / 5 (should decrease theshold  cuz more fp, img_idx = 8(shift) --> <0.1689)
        
    
    bubble_area_prop_vec=np.sum(np.sum(bubble_stack,2),1)/np.sum(np.sum(is_stem_mat2,2),1)
    
    bubble_cc_max_area_vec = np.zeros(bubble_stack.shape[0])
    for img_idx in np.where(bubble_area_prop_vec>0)[0]:
        img=bubble_stack[img_idx,:,:]
        bubble_cc_max_area_vec[img_idx]=max_cc_area(img)
    bubble_cc_max_area_prop_vec = bubble_cc_max_area_vec/np.sum(np.sum(is_stem_mat2,2),1)
    
    
    '''
    To decide bubble_cc_max_area_max--> considered as poor quality -->no emb:
    '''
    num_bins=50
    fig = plt.figure()
    n_th, bins_th, patches_th= plt.hist(bubble_cc_max_area_prop_vec,bins=num_bins)
    if plot_interm==False:
        plt.close(fig)

    #usually 1st bin is way TOO large --> don't show 1st bin(0)in hist for better visualization
    fig1 = plt.figure()
    plt.plot(bins_th[2:],n_th[1:])
    #plt.xlim(bins_th[1],bins_th[-1])
    plt.ylabel("frequency")
    plt.xlabel("max(bubble c.c. area)")
    plt.title("histogram of max of bubble connected component area(ignoring the 1st bin)")
    if is_save == True:
        plt.savefig(chunk_folder + '/m4.6_histogram of max of bubble connected component area (ignoring the 1st bin).jpg')
    if plot_interm==False:
        plt.close(fig1)
    
    fig2 = plt.figure()
    plt.plot(range(len(bubble_cc_max_area_prop_vec)),bubble_cc_max_area_prop_vec)
    plt.ylabel("max(bubble c.c. area)")
    plt.xlabel("image index")
    plt.title("max(bubble c.c. area) in each image")
    if is_save == True:
        plt.savefig(chunk_folder + '/m4.6_max(bubble c.c. area) in each image.jpg')
    if plot_interm==False:
        plt.close(fig2)
    return(bubble_cc_max_area_prop_vec)

def subset_vec_set(input_vec,start_img_idx,set1_idx,output_row_name):
    '''
    subset a vector based on an index set
    then ordered based on vector value, big to small
    '''
    #for deciding bubble_area_prop_max
    img_idx_vec = np.stack((range(0,len(input_vec)),np.around(input_vec,3)))#2 rows
    img_idx_vec[0,:] += (start_img_idx-1)
    #1st row indicator of has emb in truth, 2nd row img idx, 3rd row input_vec
    
    relative_arg_order = np.flip(np.argsort(input_vec[set1_idx],))#flip:bubble area from big to small
    set1_pd=pd.DataFrame(img_idx_vec[:,set1_idx][:,relative_arg_order],index=['img_idx',output_row_name])#Sorted by bubble_area_prop
    
    return set1_pd

def print_used_time(start_time):
    #time
    finish_time = datetime.datetime.now()
    seconds_in_day = 24 * 60 * 60
    difference = finish_time - start_time
    diff_min_sec = divmod(difference.days * seconds_in_day + difference.seconds, 60)
    print('used time: ',diff_min_sec[0],'min ',diff_min_sec[1],'sec')
    return(diff_min_sec)

def remove_cc_by_geo(input_stack,final_stack_prev_stage,has_embolism1,blur_radius,cc_height_min,cc_area_min,cc_area_max,cc_width_min,cc_width_max,weak_emb_height_min,weak_emb_area_min):
    '''
    remove cc by basic geometric shape propoerties: short/too small/too big/too wide/wide but not tall
    '''
    invalid_emb_set = []#entire img being cleaned to 0
    cleaned_but_not_all_invalid_set=[]#some cc in the img being cleaned to 0
    weak_emb_cand_set = []#entire img being cleaned to 0, but some cc are too short/small --> possibly weak emb
    output_stack = np.zeros(final_stack_prev_stage.shape)
    for img_idx in np.where(has_embolism1)[0]:
        img = input_stack[img_idx,:,:]
        #clustering process
        smooth_img = ndimage.gaussian_filter(img, sigma = blur_radius)
        
        num_cc, mat_cc, stats, centroids  = cv2.connectedComponentsWithStats(smooth_img.astype(np.uint8), 8)#8-connectivity
        
        cc_width = stats[1:,cv2.CC_STAT_WIDTH]#"1:", ignore bgd:0
        cc_height = stats[1:,cv2.CC_STAT_HEIGHT]
        cc_area = stats[1:, cv2.CC_STAT_AREA]
        
        cc_valid_labels = np.where((cc_height > cc_height_min)*(cc_area > cc_area_min)*(cc_area < cc_area_max)*(cc_width > cc_width_min)*(cc_width < cc_width_max)*(cc_width < cc_height))[0]
        
        has_short_emb = np.any((cc_height <= cc_height_min)*(cc_height > weak_emb_height_min)*(cc_width > cc_width_min)*(cc_width < cc_width_max)*(cc_width < cc_height))#shorter
        has_small_emb = np.any((cc_area <= cc_area_min)*(cc_area > weak_emb_area_min)*(cc_width > cc_width_min)*(cc_width < cc_width_max)*(cc_width < cc_height))#smaller
        
        mat_cc_valid = img*0
        if cc_valid_labels.size > 0:#not all invalid
            for cc_idx in (cc_valid_labels+1):#+1 cuz ignore bgd before
                mat_cc_valid += 1*(mat_cc==cc_idx)
            mat_cc_valid = cv2.dilate(mat_cc_valid.astype(np.uint8), np.ones((2,2),np.uint8),iterations = 1)
            #expand a bit cuz uses "filter_stack"*final_stack as input_stack
            if cc_valid_labels.size < num_cc-1:#-1 cuz of bgd
                cleaned_but_not_all_invalid_set.append(img_idx)
        else:
            invalid_emb_set.append(img_idx)
            if has_short_emb or has_small_emb:
                weak_emb_cand_set.append(img_idx)
        output_stack[img_idx,:,:]= mat_cc_valid*final_stack_prev_stage[img_idx,:,:]#or *(img>0)*255
    return output_stack,invalid_emb_set,cleaned_but_not_all_invalid_set,weak_emb_cand_set

def mat_reshape(x_mat, height: int = 256, width:int = 256):
  '''
  reshape x_mat into ndarray height* width
  '''
  x_mat_reshape = np.ndarray((x_mat.shape[0],height,width))
  for i,mat in enumerate(x_mat):
    x_mat_reshape[i] = cv2.resize(x_mat[i],(width,height))
  return x_mat_reshape

def rescue_weak_emb_by_dens(input_stack,final_stack_prev_stage,weak_emb_cand_set,blur_radius,cc_height_min,cc_area_min,cc_area_max,cc_width_min,cc_width_max,weak_emb_height_min,weak_emb_area_min,cc_dens_min,plot_interm=False):
    '''
    rescue weak emb (either in short_emb_labels OR small_emb_labels) from weak_emb_cand_set if the density of cc > cc_dens_min
    '''
    output_stack = np.zeros(final_stack_prev_stage.shape)
    has_weak_emb_set =[]
    for img_idx in weak_emb_cand_set:
        img = input_stack[img_idx,:,:]
        #clustering process
        smooth_img = ndimage.gaussian_filter(img, sigma = blur_radius)
        
        num_cc, mat_cc, stats, centroids  = cv2.connectedComponentsWithStats(smooth_img.astype(np.uint8), 8)#8-connectivity
        
        cc_width = stats[1:,cv2.CC_STAT_WIDTH]#"1:", ignore bgd:0
        cc_height = stats[1:,cv2.CC_STAT_HEIGHT]
        cc_area = stats[1:, cv2.CC_STAT_AREA]
        
        short_emb_labels = np.where((cc_height <= cc_height_min)*(cc_height > weak_emb_height_min)*(cc_width > cc_width_min)*(cc_width < cc_width_max)*(cc_width < cc_height))[0]#shorter
        small_emb_labels = np.where((cc_area <= cc_area_min)*(cc_area > weak_emb_area_min)*(cc_width > cc_width_min)*(cc_width < cc_width_max)*(cc_width < cc_height))[0]#smaller
        
        weak_emb_labels = np.unique(np.concatenate((short_emb_labels,small_emb_labels)))#join 2 list
        
        mat_cc_valid = img*0
        if plot_interm==True:
            print(img_idx)#8/16/194 (cas5.5 stem, chunk_idx =5, chunk_sz=200)/ 148(cas5.5 stem, chunk_idx =0, chunk_sz=200)
            #bin_img = bin_stack[img_idx,:,:]
        if weak_emb_labels.size > 0:
            for cc_idx in (weak_emb_labels+1):
                cc_mask = 1*(mat_cc==cc_idx)
                cc_dens = np.sum(cc_mask*smooth_img)/cc_area[cc_idx-1]
                #cc_dens is larger than 1 because smooth_img comes from input_stack = filter_stack*final_stack_prev_stage
                # and filter_stack comes from th_stack, which can be up to 255. Also, final_stack_prev_stage is either 0 or 255.
                if plot_interm==True:
                    print(cc_dens)#1995/420/807/1253
#                    print(cc_idx)#3/2/2/1
#                    #print(np.sum(cc_mask*bin_img))#308/250/116
#                    print(np.sum(cc_mask*smooth_img))#2751855/389290/451552/886046
#                    print(cc_area[cc_idx-1])#1379/925/559/707
#                    print(cc_width[cc_idx-1])#43/37/26/27
#                    print(cc_height[cc_idx-1])#46/41/27/33
                if cc_dens>cc_dens_min:
                    mat_cc_valid += cc_mask
            output_stack[img_idx,:,:] = mat_cc_valid*final_stack_prev_stage[img_idx,:,:]
            if mat_cc_valid.any():
                if plot_interm==True:
                    print(img_idx,"has weak emb")
                    print("-------------------")
                has_weak_emb_set.append(img_idx)
            
            elif plot_interm==True:
                    print("no weak emb: no cc larger then cc_dens_min",cc_dens_min)
                    print("-------------------")
        elif plot_interm==True:
            print("no weak emb")
            print("-------------------")
    return output_stack,has_weak_emb_set
            