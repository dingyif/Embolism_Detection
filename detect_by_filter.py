# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:10:10 2019

@author: USER
"""

from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage #for median filter,extract foreground
import time #to time function performance
import tifffile as tiff
import pandas as pd


img_list = []
img_folder = 'C:\ToyImgFiles\ALCLAT1_leaf Subset'
img_dir_path = img_folder + '\Images'
start_img_idx = 116
end_img_idx = 307
is_save = True
is_stem = False#set to False for leaf

img_num = end_img_idx-start_img_idx+1
img_paths = glob.glob(img_dir_path+'/*.png') #assuming png

img_re_idx = 0 #relative index for images in start_img_idx to end_img_idx
for filename in img_paths[start_img_idx-1:end_img_idx]: #original img: 958 rowsx646 cols
    img=Image.open(filename).convert('L') #.convert('L'): gray-scale # 646x958
    img_array=np.float32(img) #convert from Image object to numpy array (black 0-255 white); 958x646
    if img_re_idx==0:
        img_nrow = img_array.shape[0]
        img_ncol = img_array.shape[1]
        img_stack = np.ndarray((img_num,img_nrow,img_ncol), dtype=np.float32)
    img_stack[img_re_idx] = img_array
    img_re_idx = img_re_idx + 1

#img = Image.fromarray(img_array)# convert from array to Image object
#plt.img.show() #display Image object in a new window
#plt.imshow(img_array,cmap='gray') #display gray-scale image array in console
# increasing dire: row: top->bottom; col: left-> right
diff_stack = img_stack[1:,:,:] - img_stack[:-1,:,:] #difference btw consecutive img
if is_stem==False:
    diff_stack= -diff_stack#!!! don't know why, but this is needed the leafs from the diff_sum graphs
diff_pos_stack = (diff_stack >= 0)*diff_stack #discard the negative ones

true_diff  = tiff.imread(img_folder+'/2 Image Difference Substack ('+str(start_img_idx)+'-'+str(end_img_idx-1)+').tif')
if is_save==True:
    diff_combined = np.concatenate((true_diff,diff_pos_stack.astype(np.uint8)),axis=2)
    tiff.imsave(img_folder+'/combined_2_diff.tif', diff_combined)

def plot_img_sum(img_3d,title_str):
    sum_of_img = np.sum(np.sum(img_3d,axis=2),axis=1)#sum
    plt.figure()
    plt.plot(range(len(sum_of_img)),sum_of_img)
    plt.ylabel("sum of pixel values in an image")
    plt.xlabel("image relative index")
    plt.title(title_str)
    plt.savefig(img_folder+'/m0_'+title_str+'.jpg',bbox_inches='tight')
    # x-axis= image index. y-axis=sum of each img
    return(sum_of_img)

#check their difference, 2 figures below should look alike, but they're slightly different...
diff_sum = plot_img_sum(diff_pos_stack,"My Difference Stack (Positive only)")
true_diff_sum = plot_img_sum(true_diff,"True Difference Stack")

###normalization: 
#min_stack_vec = np.min(np.min(diff_stack,axis=2),axis=1) 
##min of each img in the stack, len=img_num

def plot_gray_img(gray_img):
    #gray img: 1D np.array
    plt.figure()
    plt.imshow(gray_img,cmap='gray')


### thresholding (clip values smaller than threshold to 0)
##!! should be take care of if we want to adjust for shift in the future
threshold = 3
th_stack = (diff_stack >= threshold)*diff_stack #clip values < threshold to 0

##frequency of pixel values in th_stack
#num_bins = 20
#n_th, bins_th, patches_th = plt.hist(th_stack.flatten(),bins=num_bins)
#th_freq = np.column_stack((bins_th[:-1],bins_th[1:],n_th))#first 2 cols: pixel value range.
## last col = frequency = the number of pixels in that given pixel value range
## overall pixel value range 0-132 for img 101 to 450

#count_bin, bins, patches = plt.hist(th_stack[24,700:900,360:450].flatten(),bins=10)
#plt.xlim(threshold, max(bins))#don't show the 0's
#plt.ylim(0,1.1*max(count_bin[np.min(np.where(bins>threshold))-1:]))
#plt.imshow(th_stack[24,:,:],cmap='gray')


### idea from https://imagejdocu.tudor.lu/gui/process/noise

### remove outlier (is this useful???)
#This is a selective median filter that replaces a pixel by the median of the pixels 
#in the surrounding if it deviates from the median by more than a certain value (the threshold). 
#if radius=2 --> median filter size = 5

#img_idx = 300
#outlier_th = 50 #how much deviation from median is considered as an outlier
##50 isn't realistic... ? but that's the default
#img_current = th_stack[img_idx,:,:]
#med_img = ndimage.median_filter(img_current, 5)
#
##can do this in stack
#is_not_outlier = abs(img_current - med_img) <= outlier_th
#rm_out_img = is_not_outlier*img_current + (~is_not_outlier)*med_img
#plt.imshow(rm_out_img,cmap='gray')
#plt.imshow(img_current,cmap='gray')

### despeckle: median filter: remove salt-pepper noise 
# potential problem, edges might be more thinned because of this?
def median_filter_stack(img_stack,kernel_radius):
    #img_stack is 3d numpy array, 1st index = img index
    #replace each pixel by the median in kernel_size*kernel_size neighborhood
    med_stack = np.ndarray(img_stack.shape, dtype=np.float32)
    for img_idx in range(img_stack.shape[0]):
        img_current = img_stack[img_idx,:,:]
        med_stack[img_idx] = ndimage.median_filter(img_current, kernel_radius)
    return(med_stack)



def to_binary(img_stack):
    #convert 0 to 0(black) and convert all postivie pixel values to 1(white)
    return((img_stack > 0)*1)

bin_stack = to_binary(th_stack)

mean_img = np.mean(bin_stack,axis=0)
plot_gray_img(mean_img)
if is_save==True:
    plt.imsave(img_folder + "/m1_mean_of_binary_img.jpg",mean_img,cmap='gray')

##sub group
#in_grp = range(0,66,1)#0,1,2,...,66
#mean_img1 = np.mean(bin_stack[in_grp,:,:],axis=0)
#plot_gray_img(mean_img1)
#if is_save==True:
#    plt.imsave(img_folder + "/m1_mean_of_binary_img_0_66.jpg",mean_img1,cmap='gray')
    
def extract_foreground(img_2d,blur_radius=10.0,fg_threshold=0.1,expand_radius_ratio=5,is_save=False):
    #smooth img_2d by gaussian filter w/blur radius
    #background when smoothed img > fg_threhold
    #expand the foreground object area by applying uniform filter w/radius = 
    #        blur_radiu*expand_radius_ratio then binarize it
    #expansion is made to be more conservative
    #displays two image, 1st img is how much expansion of foreground is done
    #       2nd img is the foreground(True) background(False) result
    #return a 2d logical array of the same size as img_2d
    
    #fg_threshold assumes embolism only happens at 
    #the same given pixel at most 10% of the time
    #which is assumption on how much xylem channels overlay
    
    
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
else:
    is_stem_mat = 1

stem_bin_stack = (is_stem_mat*th_stack > 0)*255

if is_stem==True:
    #st_time = time.time()
    med_stack = median_filter_stack(is_stem_mat*th_stack,5)
    #print(time.time()-st_time)#18 second for 350 imgs
else:
    med_stack = median_filter_stack(is_stem_mat*th_stack,5)

med_bin_stack = (med_stack > 0)*255#255(white) for embolism,0 for background

##subgroup
#is_stem1_crude = extract_foreground(mean_img1,expand_radius_ratio=5,is_save=False) 
##could be noisy because subsampled images that produce mean_img1 might have 
##smaller signal to noise ratio in certain areas
#is_stem1 = is_stem_mat*is_stem1_crude
#if is_save==True:
#        plt.imsave(img_folder + "/m3_is_stem_0_66.jpg",is_stem1,cmap='gray')
#med_stack1 = median_filter_stack(is_stem1*th_stack[in_grp,:,:],5)# mean img not from all images
#med_bin_stack1 = (med_stack1 > 0)*255

if is_save==True:
    true_mask  = tiff.imread(img_folder+'/4 Mask Substack ('+str(start_img_idx)+'-'+str(end_img_idx-1)+') clean.tif')
    combined_list = (true_mask,med_bin_stack.astype(np.uint8),stem_bin_stack.astype(np.uint8))
    final_combined = np.concatenate(combined_list,axis=2)
    final_combined_inv =  -final_combined+255 #invert 0 and 255 s.t. background becomes white
    tiff.imsave(img_folder+'/combined_4_final.tif', final_combined_inv)



##subgroup
#true_mask  = tiff.imread(img_folder+'/4 Mask Substack ('+str(start_img_idx)+'-'+str(end_img_idx-1)+') clean.tif')
#final_combined_grp = np.concatenate((true_mask[in_grp,:,:],med_bin_stack[in_grp,:,:].astype(np.uint8),med_bin_stack1.astype(np.uint8),stem_bin_stack[in_grp,:,:].astype(np.uint8)),axis=2)
##true, result, mean img not from all images, no median only extract stem
##slightly less noise
#final_combined_inv_grp =  -final_combined_grp+255 #invert 0 and 255 s.t. background becomes white
#tiff.imsave(img_folder+'/combined_4_final_0_66.tif', final_combined_inv_grp)

if is_stem==True:
    plot_gray_img(final_combined_inv[1,:,:]) #false positive...(caused by slight shift?)
    plot_gray_img(final_combined_inv[24,:,:]) #embolism
    plot_gray_img(final_combined_inv[183,:,:]) #embolism (but shrinked in size)
    plot_gray_img(final_combined_inv[66,:,:]) #false positive(big shift)
    plt.imsave(img_folder + "/m_ex_2_false_positive.jpg",final_combined_inv[1,:,:],cmap='gray')
    plt.imsave(img_folder + "/m_ex_25_correct.jpg",final_combined_inv[24,:,:],cmap='gray')
    plt.imsave(img_folder + "/m_ex_184_shrinked.jpg",final_combined_inv[183,:,:],cmap='gray')
    plt.imsave(img_folder + "/m_ex_67_false_positive_big_shift.jpg",final_combined_inv[66,:,:],cmap='gray')

##maybe to do is_stem_mat in every bin of images?
#has_shift = np.sum(np.sum(bin_stack,axis=2),axis=1)
#plt.figure()
#plt.plot(range(len(has_shift)),has_shift)
#plt.ylabel("Sum of pixel values in one binarized img")
#plt.xlabel("image relative index")
#plt.title("Shift Detection")
#plt.savefig(img_folder + "/m4_shift_detection.jpg",bbox_inches='tight')
##,bbox_inches='tight' s.t. ylabel won't be cropped off

#high pass filter but too much detail...
#kernel = np.array([[-1, -1, -1, -1, -1],
#                   [-1,  1,  2,  1, -1],
#                   [-1,  2,  4,  2, -1],
#                   [-1,  1,  2,  1, -1],
#                   [-1, -1, -1, -1, -1]])
#highpass_5x5 = ndimage.convolve(blur_mean_img, kernel)
#plt.imshow(highpass_5x5)

def confusion_mat_pixel(pred_stack,true_stack):
    #confusion matrix(pixel level)
    #positive (true_stack > 0): embolism; 
    #negative (true_stack == 0)
    #false positive
    con_fp = np.sum(np.sum(np.sum((pred_stack > true_stack))))
    #fasle negative
    con_fn = np.sum(np.sum(np.sum((pred_stack < true_stack))))
    #true positive
    con_tp = np.sum(np.sum(np.sum((pred_stack == true_stack)*(true_stack > 0))))
    #tru negative
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

con_df_px = confusion_mat_pixel(med_bin_stack,true_mask)
con_df_px
    #ALCLAT1_leaf Subset
    #          Predict 0  Predict 1
    #True 0  229378816.0  6105020.0
    #True 1       9861.0    61639.0