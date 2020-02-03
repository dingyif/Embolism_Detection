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

img_list = []
img_folder_rel = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
img_folder = os.path.join(img_folder_rel,'ToyImgFiles','CASARD2.2_leaf_Subset')
#is_stem = False#set to False for leaf
start_img_idx = 176
end_img_idx = 500
is_save = True

#automatically decide what is_stem should be
#assuming img_folder containes either "stem" or "leaf"
if "stem" in img_folder :
    is_stem = True
elif "leaf" in img_folder:
    is_stem = False
else:
    sys.exit("Error: image folder name doesn't contain strings like stem or leaf")

#############################################################################
#    Load Images
#############################################################################
img_dir_path = os.path.join(img_folder, 'Images')
img_num = end_img_idx-start_img_idx+1
img_paths = sorted(glob.glob(img_dir_path+'/*.png')) #assuming png

img_re_idx = 0 #relative index for images in start_img_idx to end_img_idx
for filename in img_paths[start_img_idx-1:end_img_idx]: #original img: 958 rowsx646 cols
    img=Image.open(filename).convert('L') #.convert('L'): gray-scale # 646x958
    img_array=np.float32(img) #convert from Image object to numpy array (black 0-255 white); 958x646
    #pu in the correct data structure
    if img_re_idx == 0:
        img_nrow = img_array.shape[0]
        img_ncol = img_array.shape[1]
        img_stack = np.ndarray((img_num,img_nrow,img_ncol), dtype=np.float32)
    img_stack[img_re_idx] = img_array
    img_re_idx = img_re_idx + 1

#############################################################################
#    Difference between consecutive images
#    and Clip negative pixel values to 0 
#############################################################################

# !!! ImageJ seems to find embolism in "leafs" 
# from the part where vein becomes "darker"...

diff_stack = img_stack[1:,:,:] - img_stack[:-1,:,:] #difference btw consecutive img
if is_stem==False:
    diff_stack= -diff_stack#!!! don't know why, but this is needed the leafs from the diff_sum graphs
diff_pos_stack = (diff_stack >= 0)*diff_stack #discard the negative ones

#read the tiff file
true_diff  = tiff.imread(img_folder+'/2 Image Difference Substack ('+str(start_img_idx)+'-'+str(end_img_idx-1)+').tif')
if is_save==True:
    diff_combined = np.concatenate((true_diff,diff_pos_stack.astype(np.uint8)),axis=2)
    tiff.imsave(img_folder+'/combined_2_diff.tif', diff_combined)

#############################################################################
#    Sanity check:
#    see if the sum of pixel values in an image of diff_pos_stack
#    is similar to those from true_diff_sum (the intermediate result they obtained from ImageJ)
#############################################################################
def plot_img_sum(img_3d,title_str):
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

#check their difference, 2 figures below should look alike, but they're slightly different...
diff_sum = plot_img_sum(diff_pos_stack,"My Difference Stack (Positive only)")
true_diff_sum = plot_img_sum(true_diff,"True Difference Stack")

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
threshold = 3
th_stack = (diff_stack >= threshold)*diff_stack #clip values < threshold to 0

#############################################################################
#    Binarize: clip all positive px values to 1
#############################################################################

def to_binary(img_stack):
    #convert 0 to 0(black) and convert all postivie pixel values to 1(white)
    return((img_stack > 0)*1)

bin_stack = to_binary(th_stack)

#############################################################################
#    Foreground Background Segmentation
#    for stem only, cuz there are clearly parts that are background (not stem)
#    want to avoid  artifacts from shrinking of gels/ moving of plastic cover
#    mask only the stem part to reduce false positive
#############################################################################

if is_stem==True:
    mean_img = np.mean(bin_stack,axis=0)
    plot_gray_img(mean_img)
    if is_save==True:
        plt.imsave(img_folder + "/m1_mean_of_binary_img.jpg",mean_img,cmap='gray')

def extract_foreground(img_2d,blur_radius=10.0,fg_threshold=0.1,expand_radius_ratio=5,is_save=False):
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
    
    #expand the stem part a bit
    unif_radius = blur_radius*expand_radius_ratio
    not_stem_exp =  to_binary(ndimage.uniform_filter(not_stem, size=unif_radius))
    
    is_stem = (not_stem_exp==0)#invert
    plot_gray_img(is_stem+not_stem)#expansion
    if is_save==True:
        plt.imsave(img_folder+"/m2_expansion_foreground.jpg",is_stem+not_stem,cmap='gray')
    plot_gray_img(is_stem)#1(white) for stem part
    if is_save==True:
        plt.imsave(img_folder + "/m3_is_stem.jpg",is_stem,cmap='gray')
    return(is_stem)#logical 2D array

if is_stem==True:
    is_stem_mat = extract_foreground(mean_img,expand_radius_ratio=8,is_save=True)
    '''
    the above is_stem_mat might be too big
    (one matrix that determines whether it's stem for all images)
    for each img, 
        use the fact that stem is brighter than background from original img 
        --> threshold by mean of each image to get another estimate of whether 
        it's stem or not for each img
        --> then intersect with is_stem_mat, to get final the final is_stem_mat2 
    '''
    mean_each_img = np.mean(np.mean(img_stack,axis=2),axis=1)#mean of each image
    mean_each_stack = np.repeat(mean_each_img[:,np.newaxis],img_stack.shape[1],1)
    mean_each_stack = np.repeat(mean_each_stack[:,:,np.newaxis],img_stack.shape[2],2) 
    bigger_than_mean =  img_stack > mean_each_stack#thresholded by mean
    is_stem_mat2 = bigger_than_mean[:-1,:,:]*(is_stem_mat*1)
    #drop the last img s.t. size would be the same as diff_stack
    #multiply by is_stem_mat to crudely remove the noises (false positive) outside of is_stem_mat2

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
            if np.sum(final_img)/stem_area > shift_th or np.sum(final_img) < final_area_th:
                '''
                percentage of embolism in the stem_area is too large
                probably is false positive due to shifting of img (case: stem img_idx=66,67)
                remove false positive with really small sizes
                '''
                final_img = final_img * 0
            
            
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


final_stack1 = np.ndarray(bin_stack.shape, dtype=np.float32)
has_embolism = np.zeros(bin_stack.shape[0])#1: that img is predicted to have embolism

if is_stem == False:
    #Leaf
    is_stem_mat2 = np.ones(bin_stack.shape)
    area_th = 30
    area_th2 = 3#30
    final_area_th = 0
    shift_th = 1
    density_th = 0.4
    num_px_th = 50
    ratio_th=5
    
    #for 2nd stage
    emb_pro_th = 0.008 #tuned by ALCLAT1_leaf 10000/(img_nrow*img_ncol)~=0.0081
    dens_rect_window_width = 10#15#window too small--> too much noise(false positive pixels), window too big-->can't find emb (false negative pixels) ex: not able to detect narrow veins with embolism
    dens_img_th = 0.6#0.5
    dens_ero_kernel_sz = 3
    dens_exp_kernel_sz = 10#3
    density_th2 = 0.5#0.6
    num_px_min = 0.00032#0.00008#tuned by ALCLAT1_leaf 100/(img_nrow*img_ncol)~= 0.00008
    num_px_max = 0.02#0.008 #ALCLAT1_leaf img_idx=83 true emb part: 0.019
    quantile_per = 0.95
    quantile_th = 0.75#0.7
    plot_interm = False
    
    emb_pro_th_min = 0.0001
    rect_width_th = 0.05#100/1286
    rect_height_th = 0.02#200/959
    cc_rect_th = 0.35#ALCLAT1_leaf img_idx=167 true emb part:0.34, img_idx=73(false positive part:0.36)
    
#    final_area_th2 = 80
#    emb_freq_th = 10/191#tuned by A_leaf
#    cc_th = 3
else:
    area_th = 1
    area_th2 = 3#10
    final_area_th = 78
    shift_th = 0.06#0.05
    density_th = 0.4
    num_px_th = 50
    ratio_th=35  
    final_area_th2 = 120
    emb_freq_th = 5/349 #depends on which stages the photos are taken
    cc_th = 3
    
bin_stem_stack = bin_stack*is_stem_mat2

for img_idx in range(0,bin_stem_stack.shape[0]):
    stem_area = np.sum(is_stem_mat2[img_idx,:,:])
    final_stack1[img_idx,:,:] = find_emoblism_by_contour(bin_stem_stack,img_idx,stem_area = stem_area,final_area_th = final_area_th,
                                                        area_th=area_th, area_th2=area_th2,ratio_th=ratio_th,e2_sz=1,o2_sz=2,cl2_sz=2,
                                                        plot_interm=False,shift_th=shift_th,density_th=density_th,num_px_th=num_px_th)
    #if np.any(final_stack[img_idx,:,:]):
    #    has_embolism[img_idx] = 1
        
'''
debugging:
    
bin_stack = bin_stem_stack
ratio_th=35
e2_sz=1
o2_sz=2
cl2_sz=2
plot_interm=True
'''
if is_stem==True:
    #############################################################################
    #Don't count as embolism if it keeps appearing (probably is plastic cover)
    #############################################################################
    final_stack_sum = np.sum(final_stack1,axis=0)/255 #the number of times embolism has occurred in a pixel
    #each pixel in final_stack is 0 or 255, hence we divide by 255 to make it more intuitive 
    #plot_gray_img(final_stack_sum,"final_stack_sum")
    not_emb_part = (final_stack_sum/(img_num-1) > emb_freq_th)
    #plot_gray_img(not_emb_part,"not_emb_part")
    num_cc_fss,mat_cc_fss = cv2.connectedComponents((final_stack_sum>cc_th).astype(np.uint8))
    not_emb_cc = np.unique(not_emb_part*mat_cc_fss)
    not_emb_mask = not_emb_part*0
    if not_emb_cc.size > 1: #not only background
        for not_emb_cc_label in not_emb_cc[1:]: #starts with "1", cuz "0" is for bkg
           not_emb_mask = not_emb_mask + (mat_cc_fss == not_emb_cc_label)*1
    plot_gray_img(not_emb_mask,"not_emb_mask")
    not_emb_mask_exp = cv2.dilate(not_emb_mask.astype(np.uint8), np.ones((10,10),np.uint8),iterations = 1)#expand a bit
    plot_gray_img(not_emb_mask_exp,"not_emb_mask_exp")
    emb_cand_mask = -(not_emb_mask_exp-1)#inverse, switch 0 and 1
    #plot_gray_img(emb_cand_mask,"emb_cand_mask")
    final_stack = np.repeat(emb_cand_mask[np.newaxis,:,:],img_num-1,0)*final_stack1
    
    #treat whole img as no emb, if the number of embolized pixels is too small in an img
    num_emb_each_img = np.sum(np.sum(final_stack/255,axis=2),axis=1)#the number of embolized pixels in each img
    emb_cand_each_img = (num_emb_each_img>final_area_th2)*1#a vector of length = (img_num-1), 0 means the img should be treated as no emb, 1 means to keep the same way as it is
    emb_cand_each_img1 = np.repeat(emb_cand_each_img[:,np.newaxis],final_stack.shape[1],1)
    emb_cand_stack = np.repeat(emb_cand_each_img1[:,:,np.newaxis],final_stack.shape[2],2)
    final_stack = emb_cand_stack*final_stack
else:    
    final_stack = np.copy(final_stack1)
    '''
    2nd stage for separating the case where embolism parts is connected to the 
    noises at the edges of imgs (Reduce false positive)
    density based segmentation + connected component
    '''
    #TODO (can try): num_emb_each_img --> too small, discard. Too big: use v2?
    #still need the previous stage to first remove cases like shifting
    #density estimation is sort of like the filter idea (just with a uniform kernel instead of gaussian/median filter)
    num_emb_each_img = np.sum(np.sum(final_stack1/255,axis=2),axis=1)#the number of embolized pixels in each img
    #num_emb_each_img --> too small, discard. Too big: density based segmentation+connected component
    #density based segmentation
    density_seg_idx = np.nonzero(num_emb_each_img/(img_nrow*img_ncol)>emb_pro_th)#images index that'll be performed density based segmentation on#TODO: tune this
    for img_idx in density_seg_idx[0]:
        
        density_img = density.density_of_a_rect(bin_stem_stack[img_idx,:,:],dens_rect_window_width)
        density_img_th = density_img>dens_img_th #thresholding
        if plot_interm == True:
            plot_gray_img(density_img_th)
        density_img_ero = cv2.erode(density_img_th.astype(np.uint8), np.ones((dens_ero_kernel_sz,dens_ero_kernel_sz),np.uint8),iterations = 1)#erose to seperate embolism from noises
        if plot_interm == True:
            plot_gray_img(density_img_ero,str(img_idx)+"_density_img_ero")
        density_img_exp = cv2.dilate(density_img_ero.astype(np.uint8), np.ones((dens_exp_kernel_sz,dens_exp_kernel_sz),np.uint8),iterations = 1)#expand to connect
        if plot_interm == True:
            plot_gray_img(density_img_exp,str(img_idx)+"_density_img_exp")
        #TODO: erosion and dilate again?
        final_img = np.zeros(density_img_exp.shape)
        num_cc4,mat_cc4 = cv2.connectedComponents(density_img_exp.astype(np.uint8))
        unique_cc_label4 = np.unique(mat_cc4)
        if unique_cc_label4.size > 1: #not only background
            for cc_label4 in unique_cc_label4[1:]: #starts with "1", cuz "0" is for bkg
                inside_cc4 = (mat_cc4 == cc_label4)*bin_stack[img_idx,:,:]
                if np.sum(inside_cc4)/(img_nrow*img_ncol)>num_px_min and np.sum(inside_cc4)/(img_nrow*img_ncol)<num_px_max: #and np.sum(inside_cc4)/np.sum(mat_cc4 == cc_label4)>density_th2:
                    #discarding parts with small area or big area                      
                    #estimated of density/avg density are not very different from noise..., hence use top 90% quantile
                    if np.quantile(density_img[np.nonzero((mat_cc4 == cc_label4))], quantile_per) >= quantile_th:
                        #discard parts with top 90% quantile being small
                        
                        #to discard if the rectangle bounding the connected component is too WIDE　and TALL 
                        #and the 
                        emb_px_pos = np.where(mat_cc4 == cc_label4)
                        rect_top = min(emb_px_pos[0])
                        rect_bottom = max(emb_px_pos[0])
                        rect_left = min(emb_px_pos[1])
                        rect_right = max(emb_px_pos[1])
                        rect_width = rect_right-rect_left
                        rect_height = rect_bottom-rect_top
                        if rect_width/img_ncol<rect_width_th or rect_height/img_nrow<rect_height_th or len(emb_px_pos[0])/(rect_width*rect_height)<cc_rect_th:#np.sum(mat_cc4 == cc_label4) is the same as len(emb_px_pos[0])
                            final_img = final_img + inside_cc4
                            if plot_interm == True:
                                print(cc_label4)
                                print("estimated number of pixels inside",np.sum(inside_cc4))
                                print("estimated of density",np.sum(inside_cc4)/np.sum(mat_cc4 == cc_label4))
                                print("avg density",np.sum((mat_cc4 == cc_label4)*density_img)/np.sum(mat_cc4 == cc_label4))
                                print("Q1:",np.quantile(density_img[np.nonzero((mat_cc4 == cc_label4))], quantile_per))
                        
        if plot_interm == True:
            plot_gray_img(final_img,str(img_idx)+"_final_img")
        final_stack[img_idx,:,:] = final_img*255
    #if the proportion of embolised pixels are smaller than emb_pro_th_min, treat as no emb
    num_emb_each_img_after = np.sum(np.sum(final_stack/255,axis=2),axis=1)
    treat_as_no_emb_idx = np.nonzero(num_emb_each_img_after/(img_nrow*img_ncol)<emb_pro_th_min)[0]
    final_stack[treat_as_no_emb_idx,:,:] = np.zeros(final_stack[treat_as_no_emb_idx,:,:].shape)
#    #############################################################################
#    #2nd stage for separating the case where embolism parts is connected to the 
#    #noises at the edges of imgs (Reduce false positive)
#    #############################################################################
#    final_stack_sum = np.sum(final_stack1,axis=0)/255 #the number of times embolism has occurred in a pixel
#    #each pixel in final_stack is 0 or 255, hence we divide by 255 to make it more intuitive 
#    #plot_gray_img(final_stack_sum,"final_stack_sum")
#    not_emb_part = (final_stack_sum/(img_num-1) > emb_freq_th)
#    #plot_gray_img(not_emb_part,"not_emb_part")
#    #not_emb_mask = cv2.erode(not_emb_part.astype(np.uint8), np.ones((2,2),np.uint8),iterations = 1)#shrink/smooth a bit
#    not_emb_mask_exp = cv2.dilate(not_emb_part.astype(np.uint8), np.ones((25,25),np.uint8),iterations = 1)#expand a bit
#    plot_gray_img(not_emb_mask_exp,"not_emb_mask_exp")
#    emb_cand_mask = -(not_emb_mask_exp-1)#inverse, switch 0 and 1
#    #plot_gray_img(emb_cand_mask,"emb_cand_mask")
#    final_stack = np.repeat(emb_cand_mask[np.newaxis,:,:],img_num-1,0)*final_stack1
#    
#    #treat whole img as no emb, if the number of embolized pixels is too small in an img
#    num_emb_each_img = np.sum(np.sum(final_stack/255,axis=2),axis=1)#the number of embolized pixels in each img
#    emb_cand_each_img = (num_emb_each_img>final_area_th2)*1#a vector of length = (img_num-1), 0 means the img should be treated as no emb, 1 means to keep the same way as it is
#    emb_cand_each_img1 = np.repeat(emb_cand_each_img[:,np.newaxis],final_stack.shape[1],1)
#    emb_cand_stack = np.repeat(emb_cand_each_img1[:,:,np.newaxis],final_stack.shape[2],2)
#    final_stack = emb_cand_stack*final_stack


#not used here, but might be useful in the future?
def add_img_info_to_img(img_idx,img_stack):
    one_img = Image.fromarray(img_stack[img_idx,:,:])
    draw = ImageDraw.Draw(one_img)
    font = ImageFont.truetype("arial.ttf", 40)#45:font size
    font_small = ImageFont.truetype("arial.ttf", 28)
    draw.text((10,10), "Image "+str(img_idx+1)+" - "+str(img_idx+2), font=font)#, fill=(125))#text color
    draw.text((10,70),img_paths[img_idx].split("\\")[-1],font=font_small)
    draw.text((100,100),"to",font=font_small)
    draw.text((10,130),img_paths[img_idx+1].split("\\")[-1],font=font_small)
    one_img.save(img_folder +'/'+str(i)+'.jpg')

#add_img_info_to_img(231,final_combined_inv)
    
def add_img_info_to_stack(img_stack):    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,50)
    fontScale              = 1
    fontColor              = 0
    lineType               = 2
    
    stack_cp = np.copy(img_stack)#make a copy s.t. changes would be made on stack_cp instead of img_stack
    
    for img_idx in range(0,img_stack.shape[0]):
        one_img_arr = stack_cp[img_idx,:,:]
        cv2.putText(one_img_arr,"Image "+str(img_idx+1)+" - "+str(img_idx+2),bottomLeftCornerOfText, font, 
            fontScale,fontColor,lineType)
        cv2.putText(one_img_arr,img_paths[img_idx].split("\\")[-1],(10,90), font, 
            0.7,fontColor,lineType)
        cv2.putText(one_img_arr,"to",(100,130), font, 
            0.7,fontColor,lineType)
        cv2.putText(one_img_arr,img_paths[img_idx+1].split("\\")[-1],(10,170), font, 
            0.7,fontColor,lineType)
        #cv2.imwrite(img_folder +'/out.jpg', one_img_arr)
    return(stack_cp)


#combined with true tif file
true_mask  = tiff.imread(img_folder+'/4 Mask Substack ('+str(start_img_idx)+'-'+str(end_img_idx-1)+') clean.tif')
if is_save==True:
    combined_list = (true_mask,final_stack.astype(np.uint8),(bin_stack*255).astype(np.uint8))
    final_combined = np.concatenate(combined_list,axis=2)
    final_combined_inv =  -final_combined+255 #invert 0 and 255 s.t. background becomes white
    final_combined_inv_info =  add_img_info_to_stack(final_combined_inv)
    tiff.imsave(img_folder+'/combined_4_final.tif', final_combined_inv_info)


#############################################################################
#    Confusion Matrix
#    To see the performance compared to those processed manually using ImageJ
############################################################################# 

def img_contain_emb(img_stack):
    return( np.any(np.any(img_stack,axis=2),axis=1)*1)#0 or 1

true_has_emb = img_contain_emb(true_mask)
has_embolism = img_contain_emb(final_stack)

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

con_img_list = confusion_mat_img(has_embolism,true_has_emb)
print(con_img_list[0])
#confusion matrix at img-level
'''
Tuned/Train using leaf ALC, testing by leaf CAS

Leaf ALC
        Predict 0  Predict 1
True 0      137.0       24.0
True 1        2.0       28.0

density
        Predict 0  Predict 1
True 0       76.0       85.0
True 1        4.0       26.0

desnsity_2:increases num_px_min, increases num_px_max
        Predict 0  Predict 1
True 0       50.0      111.0
True 1        1.0       29.0

density_3:emb_pro_th_min=0.0001 (~5min)
        Predict 0  Predict 1
True 0       60.0      101.0
True 1        1.0       29.0

density_4: rect_width_th,rect_height_th,cc_rect_th (~5min)
        Predict 0  Predict 1
True 0       71.0       90.0
True 1        1.0       29.0

cc_rect_th=0.4
        Predict 0  Predict 1
True 0       69.0       92.0
True 1        1.0       29.0

cc_rect_th=0.4
        Predict 0  Predict 1
True 0       71.0       90.0
True 1        1.0       29.0

-----------------------

Leaf CAS
        Predict 0  Predict 1
True 0      302.0        4.0
True 1       15.0        3.0

desnsity_2:
        Predict 0  Predict 1
True 0      241.0       65.0
True 1        0.0       18.0

Stem
        Predict 0  Predict 1
True 0      295.0       17.0
True 1        2.0       35.0

v3:1+4+5
        Predict 0  Predict 1
True 0      253.0       59.0
True 1        0.0       37.0

v3:1+4+5+6
        Predict 0  Predict 1
True 0      266.0       46.0
True 1        0.0       37.0

v3:1+4+5+6+7 (final_area_th2 = 80)
        Predict 0  Predict 1
True 0      294.0       18.0
True 1        0.0       37.0

v3:1+4+5+6+7 (final_area_th2 = 120)
        Predict 0  Predict 1
True 0      303.0        9.0
True 1        3.0       34.0
'''
print("false positive img index",con_img_list[1])
print("false negative img index",con_img_list[2])

F_positive = os.path.join(img_folder,'false_positive')
F_negative = os.path.join(img_folder,'false_negative')
T_negative = os.path.join(img_folder,'true_positive')

if is_save == True:
    #create/empty folder
    con_output_path = [F_positive,F_negative,T_negative]
    for foldername in con_output_path:
        if not os.path.exists(foldername):#create new folder if not existed
            os.makedirs(foldername)
        else:#empty the existing folder
            shutil.rmtree(foldername)#delete
            os.makedirs(foldername)#create
    #save images into false_positive, false_negative, true_positive subfolders
    for i in con_img_list[1]:
        plt.imsave(img_folder + "/false_positive/"+str(i)+'.jpg',final_combined_inv_info[i,:,:],cmap='gray')
    
    for i in con_img_list[2]:
        plt.imsave(img_folder + "/false_negative/"+str(i)+'.jpg',final_combined_inv_info[i,:,:],cmap='gray')
    
    for i in con_img_list[3]:
        plt.imsave(img_folder + "/true_positive/"+str(i)+'.jpg',final_combined_inv_info[i,:,:],cmap='gray')
#but there could still be cases where there are false positive pixels in true positive img


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

con_df_px = confusion_mat_pixel(final_stack,true_mask)
print(con_df_px)
total_num_pixel = final_stack.shape[0]*final_stack.shape[1]*final_stack.shape[2]
print(con_df_px/total_num_pixel)

'''
#######ALCLAT1_leaf Subset
          Predict 0  Predict 1
True 0  235408544.0    75285.0
True 1      16434.0    55066.0

density
          Predict 0  Predict 1
True 0  235380272.0   103561.0
True 1      25180.0    46320.0


density_2:increases num_px_min, increases num_px_max
false negative img index [58]
          Predict 0  Predict 1
True 0  234922016.0   561816.0
True 1      13983.0    57517.0
        Predict 0  Predict 1
True 0   0.997311   0.002385
True 1   0.000059   0.000244

density_3:emb_pro_th_min=0.0001
          Predict 0  Predict 1
True 0  234922944.0   560894.0
True 1      13983.0    57517.0
        Predict 0  Predict 1
True 0   0.997315   0.002381
True 1   0.000059   0.000244

density_4: rect_width_th,rect_height_th,cc_rect_th
          Predict 0  Predict 1
True 0  235339792.0   144043.0
True 1      15196.0    56304.0
        Predict 0  Predict 1
True 0   0.999085   0.000612
True 1   0.000065   0.000239

cc_rect_th=0.4
          Predict 0  Predict 1
True 0  235322688.0   161149.0
True 1      13983.0    57517.0
        Predict 0  Predict 1
True 0   0.999012   0.000684
True 1   0.000059   0.000244

cc_rect_th=0.35
          Predict 0  Predict 1
True 0  235335712.0   148119.0
True 1      13983.0    57517.0
        Predict 0  Predict 1
True 0   0.999068   0.000629
True 1   0.000059   0.000244

####### Leaf CAS
          Predict 0  Predict 1
True 0  399552512.0     7427.0
True 1      14448.0     6373.0

density_2
          Predict 0  Predict 1
True 0  399519936.0    40020.0
True 1       8907.0    11914.0
        Predict 0  Predict 1
True 0   0.999848    0.00010
True 1   0.000022    0.00003
#######Stem
final_area_th
          Predict 0  Predict 1
True 0  215905376.0    14504.0
True 1      14121.0    50923.0

v3:1+4+5
          Predict 0  Predict 1
True 0  215894944.0    24942.0
True 1      11707.0    53337.0

v3:1+4+5+6
          Predict 0  Predict 1
True 0  215899040.0    20840.0
True 1      12251.0    52793.0

v3:1+4+5+6+7 (final_area_th2 = 80)
          Predict 0  Predict 1
True 0  215900320.0    19574.0
True 1      12251.0    52793.0

v3:1+4+5+6+7 (final_area_th2 = 120)
        Predict 0  Predict 1
True 0  215901312.0    18575.0
True 1      12431.0    52613.0
'''