# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:00:53 2019

@author: USER
"""

from PIL import Image,ImageDraw,ImageFont
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage #for median filter,extract foreground
import time #to time function performance
import tifffile as tiff #for numpy array in tiff formate; read bioimage
import pandas as pd
import cv2
import operator#for contour
import os,shutil#for creating/emptying folders
import sys#for printing error message

img_list = []
'''
Absolute path
'''
img_folder_rel = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
img_folder = os.path.join(img_folder_rel,'ToyImgFiles','ALCLAT1_leaf Subset')
#is_stem = False#set to False for leaf
start_img_idx = 116
end_img_idx = 307
is_save = False

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
img_paths = sorted(glob.glob(img_dir_path + '/*.png')) #assuming png 
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
    #i.e. this is an assumption on how much xylem channels overlay
    
    
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
    
    # the above is_stem_mat might be too big
    # (one matrix that determines whether it's stem for all images)
    # for each img, 
    #    use the fact that stem is brighter than background from original img 
    #    --> threshold by mean of each image to get another estimate of whether 
    #        it's stem or not for each img
    #    --> then intersect with is_stem_mat, to get final the final is_stem_mat2 
    mean_each_img = np.mean(np.mean(img_stack,axis=2),axis=1)#mean of each image
    mean_each_stack = np.repeat(mean_each_img[:,np.newaxis],img_stack.shape[1],1)
    mean_each_stack = np.repeat(mean_each_stack[:,:,np.newaxis],img_stack.shape[2],2) 
    bigger_than_mean =  img_stack>mean_each_stack#thresholded by mean
    is_stem_mat2 = bigger_than_mean[:-1,:,:]*(is_stem_mat*1)
    #drop the last img s.t. size would be the same as diff_stack
    #multiply by is_stem_mat to crudely remove the noises (false positive) outside of is_stem_mat2

#############################################################################
#    main function for detecting embolism
############################################################################# 
def find_emoblism_by_contour(bin_stack,img_idx,stem_area,final_area_th = 20000/255,area_th=30, area_th2=30,ratio_th=5,e2_sz=3,o2_sz=1,cl2_sz=3,plot_interm=False,shift_th=0.05):
    ############# step 1: connect the embolism parts in the mask (more false positive) ############# 
    #    erosion(2*2) and closing(5*5)[closing_e] 
    #    --> keep contours with area>area_th and fill in the contour by polygons [contour_mask]
    #    --> expand by closing(25*25) and dilation(10,10), then find connected components [mat_cc]
    ################################################################################################
    kernel = np.ones((2,2),np.uint8) #square image kernel used for erosion
    erosion = cv2.erode(bin_stack[img_idx,:,:].astype(np.uint8), kernel,iterations = 1) #refines all edges in the binary image
    if plot_interm == True:
        plot_gray_img(erosion,str(img_idx)+"_erosion")
    
    kernel_cl = np.ones((5,5),np.uint8)
    closing_e = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel_cl)
    #using a larger kernel (kernel_cl to connect the embolism part)
    #enlarge the boundaries of foreground (bright) regions
    if plot_interm == True:
        plot_gray_img(closing_e,str(img_idx)+"_closing_e")
    
    contours, hierarchy = cv2.findContours(closing_e,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    #find contours with simple approximation
    
    areas = [] #list to hold all areas
    
    for contour in contours:
      ar = cv2.contourArea(contour)
      areas.append(ar)
    
    #remove particles with area < area_th
    area_index = [i for i,val in enumerate(areas) if val>=area_th]#extract the index in areas with areas>=area_th
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
        #    --> keep contours with area>area_th2 and fill in the contour by polygons [contour_mask2]
        ################################################################################################
        kernel_e2 = np.ones((e2_sz,e2_sz),np.uint8) #square image kernel used for erosion
        erosion2 = cv2.erode(bin_stack[img_idx,:,:].astype(np.uint8), kernel_e2,iterations = 1) #refines all edges in the binary image
        if plot_interm == True:
            plot_gray_img(erosion2,str(img_idx)+"_erosion2")
        
        kernel_o2 = np.ones((o2_sz,o2_sz),np.uint8)
        #https://homepages.inf.ed.ac.uk/rbf/HIPR2/open.htm
        opening = cv2.morphologyEx(erosion2.astype(np.uint8), cv2.MORPH_OPEN, kernel_o2)
        #tends to remove some of the foreground (bright) pixels from the edges of regions of foreground pixels.
        # However it is less destructive than erosion in general.
        
        if plot_interm == True:
            plot_gray_img(opening,str(img_idx)+"_opening")
        
        #the closing here is needed for cases like ALCLAT1_leaf Subset: img_idx = 57
        #where embolism is break into small chunks and disappear mostly after opening
        kernel_cl2 = np.ones((cl2_sz,cl2_sz),np.uint8)
        closing2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_cl2)
        if plot_interm == True:
            plot_gray_img(closing2,str(img_idx)+"_closing2")
        
        contours2, hierarchy2 = cv2.findContours(closing2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find contours with simple approximation
        
        areas2 = [] #list to hold all areas
        
        for contour2 in contours2:
          ar = cv2.contourArea(contour2)
          areas2.append(ar)
         
        #remove particles with area < area_th2
        area_index2 = [i for i,val in enumerate(areas2) if val>=area_th2]#extract the index in areas with areas>=area_th
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
                    #print(str(img_idx)+":"+str(np.sum(add_part)/area_cm2))
                    if(np.sum(add_part)/area_cm2 < ratio_th):
                        #avoid the case that a small noise in contour_mask leads to a really big "add_part"
                        #that are essentially just noises
                        #case: ALCLAT1_leaf Subset: img_idx = 190
                        final_mask = final_mask + add_part
                    
            final_img = final_mask
            if plot_interm == True:
                plot_gray_img(final_mask,str(img_idx)+"_final_mask")
                print(img_idx,np.sum(final_img)/stem_area,np.sum(final_img))
            
            ######################### ############  step 4: reduce false positive ########################## 
            #    if the percentage of embolism in the [stem_area] is too large (probably it's because of shifting of images) 
            #        OR the embolism area is too small
            #    --> treat as if there's no embolism in the entire image
            ################################################################################################
            if np.sum(final_img)/stem_area>shift_th or np.sum(final_img)<final_area_th:
                #percentage of embolism in the stem_area is too large
                #probably is false positive due to shifting of img (case: stem img_idx=66,67)
                #remove false positive with really small sizes
                final_img = final_img*0   
            if plot_interm == True:
                plot_gray_img(final_img,str(img_idx)+"_final_img")
            return(final_img*255)


final_stack = np.ndarray(bin_stack.shape, dtype=np.float32)
has_embolism = np.zeros(bin_stack.shape[0])#1: that img is predicted to have embolism

if is_stem == False:
    is_stem_mat2 = np.ones(bin_stack.shape)
    area_th = 30
    area_th2 = 30
    final_area_th = 0
    shift_th = 1
else:
    area_th = 1
    area_th2 = 10
    final_area_th = 20000/255
    shift_th = 0.05
    
bin_stem_stack = bin_stack*is_stem_mat2

for img_idx in range(0,bin_stack.shape[0]):
    stem_area = np.sum(is_stem_mat2[img_idx,:,:])
    final_stack[img_idx,:,:] = find_emoblism_by_contour(bin_stem_stack,img_idx,stem_area=stem_area,final_area_th = final_area_th,area_th=area_th, area_th2=area_th2,ratio_th=35,e2_sz=1,o2_sz=3,cl2_sz=2,plot_interm=False,shift_th=shift_th)
    #if np.any(final_stack[img_idx,:,:]):
    #    has_embolism[img_idx] = 1
        
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
print(con_img_list[0])#confusion matrix at img-level

#Tuned/Train using leaf ALC, testing by leaf CAS

#Leaf ALC
#        Predict 0  Predict 1
#True 0      137.0       24.0
#True 1        2.0       28.0

#Leaf CAS
#        Predict 0  Predict 1
#True 0      302.0        4.0
#True 1       15.0        3.0

#Stem
#             Predict 0  Predict 1
#True 0      295.0       17.0
#True 1        2.0       35.0

print("false positive img index",con_img_list[1])
print("false negative img index",con_img_list[2])

#create/empty folder
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

#######ALCLAT1_leaf Subset
#          Predict 0  Predict 1
#True 0  235408544.0    75285.0
#True 1      16434.0    55066.0

####### Leaf CAS
#         Predict 0  Predict 1
#True 0  399552512.0     7427.0
#True 1      14448.0     6373.0

#######Stem
#final_area_th
#          Predict 0  Predict 1
#True 0  215905376.0    14504.0
#True 1      14121.0    50923.0
