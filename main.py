# -*- coding: utf-8 -*-
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff #for numpy array in tiff formate; read bioimage
import cv2
import os,shutil#for creating/emptying folders
import re
import sys
from func import plot_gray_img, to_binary,plot_img_sum, plot_overlap_sum
from func import add_img_info_to_stack, extract_foregroundRGB,foreground_B,mat_reshape, corr_image
from func import detect_bubble, calc_bubble_area_prop, calc_bubble_cc_max_area_p, subset_vec_set, remove_cc_by_geo, rescue_weak_emb_by_dens
from func import img_contain_emb, extract_foreground, find_emoblism_by_contour, find_emoblism_by_filter_contour
from func import confusion_mat_img, confusion_mat_pixel,confusion_mat_cluster,calc_metric,print_used_time
from density import density_of_a_rect
from detect_by_filter_fx import median_filter_stack
import math
import datetime
start_time = datetime.datetime.now()

'''
user-specified arguments
'''
folder_idx_arg = 2
#disk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
disk_path = 'E:/Diane/Col/research/code/'
has_processed = True#Working on Processed data or Unprocessed data
chunk_idx = 0#starts from 0
chunk_size = 200#the number of imgs to process at a time #try to be a multiple of window_size(200), or else last stage of rolling window doesn't work well
#don't use 4,5, or else tif would be saved as rgb colored : https://stackoverflow.com/questions/48911162/python-tifffile-imsave-to-save-3-images-as-16bit-image-stack
is_save = True
plot_interm = False
version_num = 11.4
resize = True
folder_list = []
has_tif = []
no_tif =[]

if has_processed==True:
    img_folder_rel = os.path.join(disk_path,"Done", "Processed")
else:
    img_folder_rel = os.path.join(disk_path,"ImagesNotYetProcessed")

all_folders_name = sorted(os.listdir(img_folder_rel), key=lambda s: s.lower())
all_folders_dir = [os.path.join(img_folder_rel,folder) for folder in all_folders_name]

#for img_folder in all_folders_dir[1:7]:#1:7
img_folder = all_folders_dir[folder_idx_arg]

#Need to process c folder
#img_folder_c = all_folders_dir[14:]
#img_folder = img_folder_c[2]
print(f'img folder name : {img_folder}')

#read in either png or jpg files
img_paths = sorted(glob.glob(img_folder + '/*.png'))
if img_paths==[]:
    img_paths = sorted(glob.glob(img_folder + '/*.jpg'))
    

tiff_paths = sorted(glob.glob(img_folder + '/*.tif'))
match = False

#make sure have tif file in there for the processed ones
if tiff_paths and has_processed:
    #To get the most recent modified tiff file
    #choose the most recent modified one if there are multiple of them
    
    #first order the files by modified time. 
    #And start from the most recently modified one, see if the filename includes "mask of result of substack".
    #if there's one match, break the loop.
    tiff_paths.sort(key=lambda x: os.path.getmtime(x))
    for up_2_date_tiff in tiff_paths[::-1]:
        #get which folder its processing now
        #want to automate the index as well, so put into group
        #we can just use start_img_idx and end_img_idx
        #find files with following regex
        idx_pat = r'mask of result of substack \((?P<start_img_idx>\d{1,4})\-(?P<end_img_idx>\d{1,4})\)\.?'
        cand_match = re.search(idx_pat,up_2_date_tiff,re.I)
        if cand_match:
            match = True
            break
if match==False and has_processed:
    print("no match")
else:
#    has_tif.append(i)##commented out for-loop for img_folder
    if match==True:
        print("has tiff")    
        real_start_img_idx = int(cand_match.group('start_img_idx'))
        real_end_img_idx = int(cand_match.group('end_img_idx')) + 1#+1 cuz tiff has the same size as diff_stack, which has a smaller size (by 1 img) than img_stack
        #ex: c5 stem
    #get which folder its processing now
    img_folder_name = os.path.split(img_folder)[1]
    print(f'Image Folder Name: {img_folder_name}')
    
    is_flip=False
    
    if "inglau2_stem" in img_folder_name.lower():#is_stem==True and img_nrow < img_ncol: (can't use this for T1,T2_stem)
        print("[CAUTION] flipped img!")
        is_flip=True


    if "stem" in img_folder.lower():
        is_stem = True
    elif "leaf" in img_folder.lower():
        is_stem = False
    else:
        sys.exit("Error: image folder name doesn't contain strings like stem or leaf")
    
    start_img_idx = 1+chunk_idx*(chunk_size-1)#real_start_img_idx+chunk_idx*(chunk_size-1)
    if match==True:
        end_img_idx = min(min(start_img_idx+chunk_size-1,len(img_paths)),real_end_img_idx-(real_start_img_idx-1))
        #real_end_img_idx for in3_stem, or else run into OSError("image file is truncated") because last 2 imgs are truncated & corrupted
    else:
        end_img_idx = min(start_img_idx+chunk_size-1,len(img_paths))
  
    if is_save==True:
        #create a "folder" for saving resulting tif files such that next time when re-run this program,
        #the resulting tif file won't be viewed as the most recent modified tiff file
        #chunk_folder = os.path.join(img_folder,img_folder_name,'v'+str(version_num)+'_'+str(chunk_idx)+'_'+str(start_img_idx)+'_'+str(end_img_idx))
        chunk_folder = os.path.join(img_folder_rel,'version_output','v'+str(version_num),img_folder_name,'v'+str(version_num)+'_'+str(chunk_idx)+'_'+str(start_img_idx)+'_'+str(end_img_idx))
        if not os.path.exists(chunk_folder):#create new folder if not existed
            os.makedirs(chunk_folder)
        else:#empty the existing folder
            shutil.rmtree(chunk_folder)#delete
            os.makedirs(chunk_folder)#create
    else:
        chunk_folder=""#just a placeholder for fx like calc_bubble_area_prop
    
    
    img_num = end_img_idx-start_img_idx + 1#for initializing img_stack, might be too big if there's th.jpg or preview.png
    img_num_real = 0
    ignore_num = 0 
    img_re_idx = 0 #relative index for images in start_img_idx to end_img_idx
    
    #[resize](Dingyi)
    #get img size
    img = Image.open(img_paths[1]).convert('L') #.convert('L'): gray-scale # 646x958
    img_array = np.float32(img)
    if is_flip:
        img_width = min(img_array.shape[1], 800)
        img_height = math.ceil(min(img_width/img_array.shape[1] * img_array.shape[0] , img_array.shape[0]))  
    else:
        img_height = min(img_array.shape[0], 800)
        img_width = math.ceil(min(img_height/img_array.shape[0] * img_array.shape[1] , img_array.shape[1]))
    
    for filename in img_paths[(start_img_idx-1):]: #original img: 958 rowsx646 cols
        #can't just use end_img_idx for img_paths cuz of th.jpg
        if img_re_idx < chunk_size and img_re_idx < end_img_idx:#real_end_img_idx for in3_stem, or else run into OSError("image file is truncated")
            if img_re_idx%100==0:#for debugging on server
                print("img_re_idx:",img_re_idx)
            img=Image.open(filename).convert('L') #.convert('L'): gray-scale # 646x958
            img_array = np.float32(img) #convert from Image object to numpy array (black 0-255 white); 958x646
            img_array_resize = cv2.resize(img_array,(img_width,img_height))
            #put in the correct data structure
            if img_re_idx == 0:
                if resize:
                    img_nrow = img_array_resize.shape[0]
                    img_ncol = img_array_resize.shape[1]
                else:
                    img_nrow = img_array.shape[0]
                    img_ncol = img_array.shape[1]
                img_stack = np.ndarray((img_num,img_nrow,img_ncol), dtype=np.float32)
                print("img size:",img_nrow," x ", img_ncol)
            
            if resize:
                if img_array_resize.shape[0]==img_nrow and img_array_resize.shape[1]==img_ncol:
                    img_stack[img_re_idx] = img_array_resize
                    img_re_idx = img_re_idx + 1
                    img_num_real += 1
                else:
                    #to avoid error produced by th.jpg or preview.png
                    #ValueError: could not broadcast input array from shape (768,1024) into shape (1944,2592)
                    #ex: in ImagesNotYetProcessed\dacexc2.2_stem or Processed/a5.2_stem
                    print("Different img size",filename)
                    img_paths.remove(filename)#remove from the list -->won't cause problem in extract_foregroundRGB, add_img_info_to_stack
                    ignore_num += 1
            else:
                if img_array.shape[0]==img_nrow and img_array.shape[1]==img_ncol:
                    img_stack[img_re_idx] = img_array
                    img_re_idx = img_re_idx + 1
                    img_num_real += 1
                else:
                    #to avoid error produced by th.jpg or preview.png
                    #ValueError: could not broadcast input array from shape (768,1024) into shape (1944,2592)
                    #ex: in ImagesNotYetProcessed\dacexc2.2_stem or Processed/a5.2_stem
                    print("Different img size",filename)
                    img_paths.remove(filename)#remove from the list -->won't cause problem in extract_foregroundRGB, add_img_info_to_stack
                    ignore_num += 1
            
    
    if ignore_num >0:
        img_num = img_num_real
        img_stack = img_stack[:img_num,:,:]# so that there won't  be error with add_img_info_to_stack
        #end_img_idx = end_img_idx - ignore_num #s.t. no "index out of bounds" for extract_foregroundRGB c5_stem (chunk_idx=6,chunk_size=200) #not sure if this solves for a5.2
        end_img_idx = img_num +start_img_idx-1#for a5.2 (th.jpg) not sure if it works for others...
    
    print("finish loading img")
    print("img_num:",img_num)
    print("end_img_idx:",end_img_idx)
    #############################################################################
    #    Difference between consecutive images
    #    and Clip negative pixel values to 0 
    #############################################################################

    diff_stack = img_stack[1:,:,:] - img_stack[:-1,:,:] #difference btw consecutive img
    if is_stem==False:
        diff_stack= -diff_stack#!!! don't know why, but this is needed the leafs from the diff_sum graphs
    diff_pos_stack = (diff_stack >= 0)*diff_stack #discard the negative ones
    print("diff_pos_stack done")


    # Thresholding (clip pixel values smaller than threshold to 0)
    #  threshold is currently set to 3 (suggested by Chris)
    threshold = 3
    th_stack = (diff_stack >= threshold)*diff_stack
    
    #help to clip all positive px value to 1
    bin_stack = to_binary(th_stack)
    print("bin_stack done")
    
    #time
    print_used_time(start_time)
    
    '''
    Foreground Background Segmentation
    for stem only, cuz there are clearly parts that are background (not stem)
    want to avoid artifacts from shrinking of gels/ moving of plastic cover
    mask only the stem part to reduce false positive
    '''
    if is_stem==True:
        mean_img = np.mean(bin_stack,axis=0)
        if plot_interm==True:
            plot_gray_img(mean_img)
        if is_save==True:
            plt.imsave(chunk_folder + "/m1_mean_of_binary_img.jpg",mean_img,cmap='gray')
    
    if is_stem==True:
#        is_stem_mat = extract_foreground(mean_img,chunk_folder,expand_radius_ratio=8,is_save=True)
#        
#        the above is_stem_mat might be too big
#        (one matrix that determines whether it's stem for all images)
#        for each img, 
#            use the fact that stem is brighter than background from original img 
#            --> threshold by mean of each image to get another estimate of whether 
#            it's stem or not for each img
#            --> then intersect with is_stem_mat, to get final the final is_stem_mat2 
#        
#        mean_each_img = np.mean(np.mean(img_stack,axis=2),axis=1)#mean of each image
#        mean_each_stack = np.repeat(mean_each_img[:,np.newaxis],img_stack.shape[1],1)
#        mean_each_stack = np.repeat(mean_each_stack[:,:,np.newaxis],img_stack.shape[2],2) 
#        bigger_than_mean =  img_stack > mean_each_stack#thresholded by mean
#        is_stem_mat2 = bigger_than_mean[:-1,:,:]*(is_stem_mat*1)
#        #drop the last img s.t. size would be the same as diff_stack
#        #multiply by is_stem_mat to crudely remove the noises (false positive) outside of is_stem_mat2
        if version_num >= 10:
            use_max_area = False
        else:
            use_max_area = True
        
        is_stem_matG = np.ones(img_stack.shape)
        
        '''
        v10.1: Don't use is_stem_matG at all(inspired by unprocessed/Alclat3_stem,Alclat5_stemDoneBad) (fp might increase)
        '''
        if version_num < 10.1:
            img_re_idx = 0
            for filename in img_paths[(start_img_idx-1):end_img_idx]: #original img: 958 rowsx646 cols
                imgRGB_arr=np.float32(Image.open(filename))#RGB image to numpy array
                imgGarray = imgRGB_arr[:,:,1] #only look at G layer
                imgGarray_resize = cv2.resize(imgGarray,(img_width, img_height))
                #put in the correct data structure
                if img_re_idx==0 and is_save==True:
                    if resize:
                        is_stem_matG[img_re_idx] = extract_foregroundRGB(imgGarray_resize,img_re_idx, chunk_folder, blur_radius=10.0,expand_radius_ratio=2,is_save=True,use_max_area=use_max_area)
                    else:
                        is_stem_matG[img_re_idx] = extract_foregroundRGB(imgGarray,img_re_idx, chunk_folder, blur_radius=10.0,expand_radius_ratio=2,is_save=True,use_max_area=use_max_area)
    
                else:
                    if resize:
                        is_stem_matG[img_re_idx] = extract_foregroundRGB(imgGarray_resize,img_re_idx, chunk_folder, blur_radius=10.0,expand_radius_ratio=2,use_max_area=use_max_area)
                    else:
                        is_stem_matG[img_re_idx] = extract_foregroundRGB(imgGarray,img_re_idx, chunk_folder, blur_radius=10.0,expand_radius_ratio=2,use_max_area=use_max_area)
                img_re_idx = img_re_idx + 1
        
        
        #be more consservative about shifting, else error accumulation...
        shift_px_min = 0
        shift_ratio = 0.95
        
        stem_path = os.path.join(img_folder,"input", "stem.jpg")#input img (stem for 1st img)
        if version_num >=10 and not os.path.exists(stem_path):
            print("error : no input/stem.jpg")
            sys.exit("Error: no input/stem.jpg")
        elif version_num >= 10 and os.path.exists(stem_path):
            '''
            v10:
            use_max_area=False for is_stem_G
            read in input/stem.jpg as the 1st mask in is_stem_mat_cand (i.e. is_stem_mat_cand[0])
            use corr_image to detect shift from prev img, then shift accordingly and save into is_stem_mat_cand
            (v10)the final stem mask is the intersection of is_stem_matG and is_stem_mat_cand
            (v10.1)the final stem mask is the is_stem_mat_cand
            '''
            #read-in stem.jpg
            stem_img0=Image.open(stem_path).convert('L') #.convert('L'): gray-scale # 646x958
            stem_arr0 = np.float32(stem_img0)/255 #convert to array and {0,255} --> {0,1}
            if resize:
                stem_arr0 = cv2.resize(stem_arr0,(img_width,img_height))
            is_stem_mat_cand = np.zeros(img_stack.shape)#initialize by 0's (easier for padding left, right)
            is_stem_mat_cand[0,:,:] = stem_arr0#initialize 1st img's stem by stem.jpg
            for img_re_idx in range(1, img_stack.shape[0]):#1 cuz will look at (img_re_idx-1,img_re_idx) at once
                shift_down,shift_right = corr_image(img_stack[img_re_idx-1,:,:], img_stack[img_re_idx,:,:])
                if abs(shift_down) >= shift_px_min:
                    pad_row = int(abs(shift_down*shift_ratio))
                    pad_before_row = max(pad_row,0)#>0: down
                    pad_after_row = max(-pad_row,0)#>0: up
                else:
                    pad_row = 0
                    pad_before_row=0
                    pad_after_row=0
                if abs(shift_right) >= shift_px_min:
                    pad_col = int(abs(shift_right*shift_ratio))
                    pad_before_col = max(pad_col,0)#>0: right
                    pad_after_col = max(-pad_col,0)#>0: left
                else:
                    pad_col = 0
                    pad_before_col=0
                    pad_after_col=0
                is_stem_mat_pad = np.pad(is_stem_mat_cand[img_re_idx-1,:,:], ((pad_before_row*2, pad_after_row*2), (pad_before_col*2, pad_after_col*2)), 'edge')
                is_stem_mat_cand[img_re_idx,:,:]=is_stem_mat_pad[pad_row:img_nrow+pad_row,pad_col:img_ncol+pad_col]
            
            is_stem_mat = is_stem_matG*is_stem_mat_cand
        else:#version_num<10
            '''
            v9.7: to separate bark and stem (both very green --> is_stem_mat)
            assume stem is whiter(larger B value)
            '''
    
            quan_th = 0.8
            #quan_th=0.9 --> too strong for cas2.2_Stem --> is_stem_matB_before_max_area becomes not connected
            #quan_th: 0.9 --> 0.8 and expand_radius_ratio=9 --> 8
            is_stem_matB = np.ones(img_stack.shape)
            img_re_idx = 0
            if is_flip==False:
                for filename in img_paths[(start_img_idx-1):end_img_idx]:
                    imgRGB_arr = np.float32(Image.open(filename))#RGB image to numpy array
                    imgBarray = imgRGB_arr[:,:,2] #only look at B layer
                    imgBarray_resize = cv2.resize(imgBarray,(img_width, img_height))
                    #put in the correct data structure
                    if img_re_idx==0 and is_save==True:
                        #v9.85: add img_nrow to foreground_B
                        if resize:
                            is_stem_matB[img_re_idx] = foreground_B(imgBarray_resize,img_nrow,img_re_idx, chunk_folder,quan_th=quan_th,G_max = 160, blur_radius=10.0,expand_radius_ratio=9,is_save=True)
                        else:
                            is_stem_matB[img_re_idx] = foreground_B(imgBarray,img_nrow,img_re_idx, chunk_folder,quan_th=quan_th,G_max = 160, blur_radius=10.0,expand_radius_ratio=9,is_save=True)
                    else:
                        if resize:
                            is_stem_matB[img_re_idx] = foreground_B(imgBarray_resize,img_nrow,img_re_idx, chunk_folder,quan_th=quan_th,G_max = 160, blur_radius=10.0,expand_radius_ratio=9)
                        else:
                            is_stem_matB[img_re_idx] = foreground_B(imgBarray,img_nrow,img_re_idx, chunk_folder,quan_th=quan_th,G_max = 160, blur_radius=10.0,expand_radius_ratio=9)
                    img_re_idx = img_re_idx + 1
            else:
                for filename in img_paths[(start_img_idx-1):end_img_idx]: 
                    imgRGB_arr=np.float32(Image.open(filename))#RGB image to numpy array
                    imgBarray=imgRGB_arr[:,:,2] #only look at B layer
                    imgBarray_resize = cv2.resize(imgBarray,(img_width, img_height))
                    #put in the correct data structure
                    if img_re_idx==0 and is_save==True:
                        if resize:
                            is_stem_matB[img_re_idx] = foreground_B(imgBarray_resize,img_ncol,img_re_idx, chunk_folder,quan_th=quan_th,G_max = 160, blur_radius=10.0,expand_radius_ratio=9,is_save=True)
                        else:
                            is_stem_matB[img_re_idx] = foreground_B(imgBarray,img_ncol,img_re_idx, chunk_folder,quan_th=quan_th,G_max = 160, blur_radius=10.0,expand_radius_ratio=9,is_save=True)
                    else:
                        if resize:
                            is_stem_matB[img_re_idx] = foreground_B(imgBarray_resize,img_ncol,img_re_idx, chunk_folder,quan_th=quan_th,G_max = 160, blur_radius=10.0,expand_radius_ratio=9)
                        else:
                            is_stem_matB[img_re_idx] = foreground_B(imgBarray,img_ncol,img_re_idx, chunk_folder,quan_th=quan_th,G_max = 160, blur_radius=10.0,expand_radius_ratio=9)
                    img_re_idx = img_re_idx + 1
            is_stem_mat = is_stem_matG*is_stem_matB

        
        is_stem_mat2 = is_stem_mat[:-1,:,:]#drop the last img s.t. size would be the same as diff_stack
        if is_save==True:
            plt.imsave(chunk_folder + "/m_3_is_stem_mat2_0.jpg",is_stem_mat2[0,:,:],cmap='gray')
            plt.imsave(chunk_folder + "/m_3_is_stem_mat2_last.jpg",is_stem_mat2[-1,:,:],cmap='gray')
            plt.imsave(chunk_folder + "/m_3_stem_and_img_0.jpg",is_stem_mat2[0]*img_stack[0],cmap='gray')
            plt.imsave(chunk_folder + "/m_3_stem_and_img_last.jpg",is_stem_mat2[-1]*img_stack[-2],cmap='gray')
        print("finish is_stem_mat2")
        
    
    final_stack1 = np.zeros(bin_stack.shape)
    has_embolism = np.zeros(bin_stack.shape[0])#1: that img is predicted to have embolism

    if is_stem == False:
        #Leaf
        is_stem_mat2 = np.ones(bin_stack.shape)
        area_th = 30
        area_th2 = 3#30
        final_area_th = 0
        max_emb_prop = 1
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
        
        emb_pro_th_min = 0.0001
        rect_width_th = 0.05#100/1286
        rect_height_th = 0.02#200/959
        cc_rect_th = 0.35#ALCLAT1_leaf img_idx=167 true emb part:0.34, img_idx=73(false positive part:0.36)
    else:
        #is_stem_mat2 = np.ones(bin_stack.shape)
        area_th = 1
        area_th2 = 3#10
        if is_flip==True:#in case img is flipped: in2_stem
            c1_sz = max(round(25/646*img_ncol),1)
            d1_sz = max(round(10/646*img_ncol),1)
        else:#normal direction
            c1_sz = max(round(25/646*img_nrow),1)
            d1_sz = max(round(10/646*img_nrow),1)
         
        max_emb_prop = 0.3#(has to be > 0.05 for a2_stem img_idx=224; has to > 0.19 for c4_stem img_idx=39; has to <0.29 for a4_stem img_idx=5; but has to be <0.19 for a4_stem img_idx=1
        #TODO: don't use max_emb_prop, but use img_sum?
        density_th = 0.3#<0.32 for cas2.2 stem img_idx=184
        num_px_th = 50#[Dingyi] consider changing it to 20?
        ratio_th=35  
        
        emb_freq_th = 0.05#5/349=0.014#a2_stem #depends on which stages the photos are taken
        cc_th = 3
        window_size = 200
        minRadius = 5
        bubble_area_prop_max = 0.1 #0.2 (cuz hough_param2 increases from 10 to 15)
        bubble_cc_max_area_prop_max = 0.14 #(0.08:cas5_stem+0.2:Alclat2_stem)/2
        second_ero_kernel_sz = 3
        second_clo_kernel_sz = 10
        second_width_max = 0.85#cas2.2 img_idx=232, width=0.41/ img_idx=208, width=0.8
        second_rect_dens_max = 0.3
        second_area_max = 0.2
        hough_param2=15#10
        
        if resize:
            final_area_th = 55#[Dingyi]before resize is 78 change 55 as (alcat2 stem 376 img have )
            final_area_th2 = 55#[Dingyi]used to be 80
            median_kernel_sz = 2#[Dingyi suggested]C2.2_stem, filter_stack, img_idx: 122,146,148,152,159,185,188,190,194
        else:
            final_area_th = 78
            final_area_th2 = 80
            median_kernel_sz = 5
    
    bin_stem_stack = bin_stack*is_stem_mat2
    '''1st stage'''
    if is_stem==True:
        filter_stack = median_filter_stack(is_stem_mat2*th_stack,median_kernel_sz)#[resize Dingyi]might consider 2?#5 is better than max(round(5/646*img_ncol),1) for cas2.2_Stem
            
        print("median filter done")
        
        '''
        bubble detection
        '''
        bubble_stack,has_bubble_vec= detect_bubble(filter_stack, minRadius = minRadius, hough_param2=hough_param2)
        
        if is_save==True:
            max_filter_stack = np.max(np.max(filter_stack,2),1) + 1 #"+1" to avoid divide by 0. TODO: remove "+1" but replace 0 by 1
            filter_norm = filter_stack/np.repeat(np.repeat(max_filter_stack[:,np.newaxis],img_nrow,1)[:,:,np.newaxis],img_ncol,2)#normalize for displaying
            combined_list_bubble = ((bubble_stack*255).astype(np.uint8),(filter_norm*255).astype(np.uint8),(bin_stack*255).astype(np.uint8))
            bubble_combined = np.concatenate(combined_list_bubble,axis=2)
            bubble_combined_inv =  -bubble_combined+255#so that bubbles: white --> black, bgd: black-->white
            tiff.imsave(chunk_folder+'/bubble_stack.tif', bubble_combined_inv)
            print("saved bubble_stack.tif")
            
            has_bubble_idx = np.where(has_bubble_vec==1)[0]
            has_bubble_per = round(100*len(has_bubble_idx)/(img_num-1),2)
        else:
            print("finish bubble_stack")
        
        bubble_area_prop_vec = calc_bubble_area_prop(bubble_stack,is_stem_mat2,chunk_folder,is_save=is_save,plot_interm=plot_interm)
        poor_qual_set = np.where(bubble_area_prop_vec >= bubble_area_prop_max)[0]
        
        bubble_cc_max_area_prop_vec = calc_bubble_cc_max_area_p(bubble_stack,is_stem_mat2,chunk_folder,is_save=is_save,plot_interm=plot_interm)
        poor_qual_set_cc = np.where(bubble_cc_max_area_prop_vec>=bubble_cc_max_area_prop_max)[0]
        poor_qual_set1 = poor_qual_set_cc#for 1st stage
        '''
        shift detection
        '''
        plot_overlap_sum(is_stem_mat, img_folder_name ,chunk_folder, is_save = is_save)
        
        
        for img_idx in range(0, bin_stack.shape[0]):
            if img_idx not in poor_qual_set1:
                #the if condition above saves some time but doesn't change much results 
                #(cuz some predictions would be blocked by shift_th)
                #results effective on img_idx=5,8 of cas 2.2 stem
                #could introduce more false positive cuz in the stage "Don't count as embolism if it keeps appearing (probably is plastic cover)"
                #the not_emb_mask would become smaller 
                stem_area = np.sum(is_stem_mat2[img_idx,:,:])
                if stem_area==0:
                    print(img_idx," : stem_area=0")
                final_stack1[img_idx,:,:] = find_emoblism_by_filter_contour(bin_stem_stack,filter_stack,img_idx,stem_area = stem_area,final_area_th = final_area_th,
                                                            area_th=area_th, area_th2=area_th2,ratio_th=ratio_th,e2_sz=1,o2_sz=2,cl2_sz=2,c1_sz=c1_sz,d1_sz=d1_sz,
                                                            plot_interm=plot_interm,max_emb_prop=max_emb_prop,density_th=density_th,num_px_th=num_px_th,resize=resize)
            #TODO: closing/dilate param should depend on the width of stem (now it's depend on width of img)
    else:
        for img_idx in range(0, bin_stack.shape[0]):
            stem_area = np.sum(is_stem_mat2[img_idx,:,:])
            final_stack1[img_idx,:,:] = find_emoblism_by_contour(bin_stem_stack,img_idx,stem_area = stem_area,final_area_th = final_area_th,
                                                        area_th=area_th, area_th2=area_th2,ratio_th=ratio_th,e2_sz=1,o2_sz=2,cl2_sz=2,
                                                        plot_interm=plot_interm,max_emb_prop=max_emb_prop,density_th=density_th,num_px_th=num_px_th)
    print("1st stage done")
    '''2nd stage: reduce false positive'''
    if is_stem==True:
        
        final_stack21 = np.copy(final_stack1)
        '''
        if there's at least one cc too wide, big area, and dense in bounding box --> flag as poor quality (poor_qual_set2)
        and treat as if no emb
        TODO: maybe try using density seg. as in leafs?
        '''
        poor_qual_set2 =[]
        
        
        if is_flip==True:#flip for the special horizontal img:
            middle_row = np.where(is_stem_mat2[0,:,round(img_ncol/2)])[0]#take the middle row(in case top/bottom of stem isn't correctly detected cuz of bark)
        else:#normal direction
            middle_row = np.where(is_stem_mat2[0,round(img_nrow/2),:])[0]#take the middle row(in case top/bottom of stem isn't correctly detected cuz of bark)
        stem_est_width1 = middle_row[-1]-middle_row[0]+1#an estimate of stem_width based on the middle row of is_stem_mat2 1st img
        stem_est_area1 = np.sum(is_stem_mat2[0,:,:])#stem area of 1st img
        
        for img_idx in range(0, final_stack21.shape[0]):
            current_img = final_stack1[img_idx,:,:]
            if np.sum(current_img)>0:
        #        img_ero = cv2.erode(current_img.astype(np.uint8), np.ones((second_ero_kernel_sz,second_ero_kernel_sz),np.uint8),iterations = 1)#erose to seperate embolism from noises
        #        if plot_interm == True:
        #            plot_gray_img(img_ero,str(img_idx)+"_img_ero")
                #density_img_exp = cv2.closing(density_img_ero.astype(np.uint8), ,np.uint8),iterations = 1)#expand to connect
                img_clo = cv2.morphologyEx(current_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((second_clo_kernel_sz,second_clo_kernel_sz),np.uint8))
                if plot_interm == True:
                    plot_gray_img(img_clo,str(img_idx)+"_img_clo")
                num_cc, mat_cc, stats, centroids  = cv2.connectedComponentsWithStats(img_clo.astype(np.uint8), 8)#8-connectivity
                
                
                #TODO: not sure if this is the correct direction for the special horiz. imgs
                #first two arg: total number of cc, mat with same input size that labels each cc
                cc_width = stats[1:,cv2.CC_STAT_WIDTH]#"1:", ignore bgd:0
                cc_height = stats[1:,cv2.CC_STAT_HEIGHT]
                cc_area = stats[1:, cv2.CC_STAT_AREA]
                #largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                
                cc_area_too_big = cc_area/stem_est_area1 > second_area_max
                cc_too_wide = cc_width/stem_est_width1 > second_width_max
                cc_high_dens_in_rect = cc_area/(cc_width*cc_height) > second_rect_dens_max
                #cas2.2 img_idx=193 (true emb) the top bark is wide (0.93) and high dens, so need area
                #cas2.2 img_idx=6 (false pos) is wide prop (>1) and big area prop (0.239) and high dens (0.425)
                #cas2.2 img_idx=213 (true emb) is wide prop (>1) and big area proportion (0.22) but density of cc in bounding box is small (0.26)
                if np.any(cc_too_wide*cc_high_dens_in_rect*cc_area_too_big):
                    poor_qual_set2.append(img_idx)
                    final_stack21[img_idx,:,:]=mat_cc*0
                    if plot_interm==True:
                        print(img_idx," in poor_qual_set2")
                    

#        '''
#        remove imgs with bubble before rolling window
#        '''
#        no_bubble_stack = 1-bubble_stack
#        final_stack21=final_stack1*no_bubble_stack
        
        '''
        Don't count as embolism if it keeps appearing (probably is plastic cover) (rolling window)
        '''
        final_stack= np.zeros(bin_stack.shape)
        window_idx_max = math.ceil(bin_stack.shape[0]/window_size)
        for window_idx in range(0,window_idx_max):
            window_start_idx = window_idx*window_size
            window_end_idx = min((window_idx+1)*window_size,img_num-1)
            # "-start_img_idx": because start_img_idx might not start at 0
            current_window_size = window_end_idx-window_start_idx#img_num mod window_size might not be 0 
            
            substack = np.sum(final_stack21[window_start_idx:window_end_idx],axis=0)/255
            #the number of times embolism has occurred in a pixel
            #each pixel in final_stack is 0 or 255, hence we divide by 255 to make it more intuitive 
            #plot_gray_img(final_stack_sum,"final_stack_sum")
            not_emb_mask = (substack/current_window_size > emb_freq_th)*(substack>cc_th)
            #plot_gray_img(not_emb_mask,str(window_idx)+"_not_emb_mask")
            if is_save==True:
                plt.imsave(chunk_folder + "/"+str(window_idx)+"_not_emb_mask.jpg",not_emb_mask,cmap='gray')
            
            if resize:
                #[Dingyi] 9 instead of 10
            	not_emb_mask_exp = cv2.dilate(not_emb_mask.astype(np.uint8), np.ones((9,9),np.uint8),iterations = 1)#expand a bit
            else:
            	not_emb_mask_exp = cv2.dilate(not_emb_mask.astype(np.uint8), np.ones((10,10),np.uint8),iterations = 1)#expand a bit
            
            if plot_interm==True:
                plot_gray_img(not_emb_mask_exp,str(window_idx)+"_not_emb_mask_exp")
            if is_save==True:
                plt.imsave(chunk_folder + "/"+str(window_idx)+"_not_emb_mask_exp.jpg",not_emb_mask_exp,cmap='gray')
            emb_cand_mask = -(not_emb_mask_exp-1)#inverse, switch 0 and 1
            #plot_gray_img(emb_cand_mask,"emb_cand_mask")
            final_stack[window_start_idx:window_end_idx,:,:] = np.repeat(emb_cand_mask[np.newaxis,:,:],current_window_size,0)*final_stack21[window_start_idx:window_end_idx,:,:]
        
        #treat whole img as no emb, if the number of embolized pixels is too small in an img
        num_emb_each_img = np.sum(np.sum(final_stack/255,axis=2),axis=1)#the number of embolized pixels in each img
        emb_cand_each_img = (num_emb_each_img>final_area_th2)*1#a vector of length = (img_num-1), 0 means the img should be treated as no emb, 1 means to keep the same way as it is
        emb_cand_each_img1 = np.repeat(emb_cand_each_img[:,np.newaxis],final_stack.shape[1],1)
        emb_cand_stack = np.repeat(emb_cand_each_img1[:,:,np.newaxis],final_stack.shape[2],2)
        final_stack = emb_cand_stack*final_stack
        

#        '''
#        remove imgs with bubble after rolling window (v9.6)
#        '''
#        no_bubble_stack = 1-bubble_stack
#        final_stack=final_stack*no_bubble_stack
        '''
        remove short/too small/too big/too wide cc
        remove cc wide but not tall
        '''
        has_embolism1 = img_contain_emb(final_stack)
        if resize:
            if version_num >= 11.4:
                blur_radius = 3
                cc_dens_min = 1500#[resize] Diane(cas5.5 stem, 148.jpg (has emb): 1945)
                weak_emb_height_min = 40##[resize] Diane(cas5.5 stem, 148.jpg (has emb): 49)
                weak_emb_area_min = 700#[resize] Diane(cas5.5 stem, 148.jpg (has emb): 1158)
            else:
                blur_radius = 5 #[Dingyi] 6 is too big might try 5-4
                cc_dens_min = 1000#hasn't tuned yet(cas5.5 stem, 148.jpg (has emb): 1253)
                weak_emb_height_min = 25#hasn't tuned #maybe to 30?(cas5.5 stem, 148.jpg (has emb): 33)
                weak_emb_area_min = 500#hasn't tuned(cas5.5 stem, 148.jpg (has emb): 707) 
            cc_height_min = 49 #[Dingyi]alcalt 2 stem img:218 50
        else:
            blur_radius = 3
            cc_height_min = 70
        cc_area_min = 1000
        cc_area_max = 75000
        cc_width_min = 25	
        cc_width_max = 200#100#v9.82(100-->150):#v9.83(150-->200) c5_stem img_idx=28: cc_width=157#basically useless
        
        final_stack_prev_stage = np.copy(final_stack)
        input_stack = filter_stack*final_stack_prev_stage
        #before_rm_cc_geo_stack_small = mat_reshape(final_stack_prev_stage,round(img_nrow/3),round(img_ncol/3))#reshape to 256x256. can barely see the weak emb?
        final_stack,geo_invalid_emb_set,cleaned_but_not_all_geo_invalid_set,weak_emb_cand_set = remove_cc_by_geo(input_stack,final_stack_prev_stage,has_embolism1,blur_radius,cc_height_min,cc_area_min,cc_area_max,cc_width_min,cc_width_max,weak_emb_height_min,weak_emb_area_min)
        if version_num >= 11:
            '''
            v11: strong_emb_cand(stc)(combine v10.1 and v9.85 )
            Save output of v9.85 (weak_emb and small noises discarded) and use it as the input for the stage that separates strong emb. and big noises
            output: stc_predict.tif.tif, stc_true_positive_index.txt, stc_false_negative_index.txt, stc_false_positive_index.txt
            '''
            final_stack_strong_emb_cand = final_stack.copy()
            
        if version_num >= 9.9:
            weak_emb_stack,has_weak_emb_set = rescue_weak_emb_by_dens(input_stack,final_stack_prev_stage,weak_emb_cand_set,blur_radius,cc_height_min,cc_area_min,cc_area_max,cc_width_min,cc_width_max,weak_emb_height_min,weak_emb_area_min,cc_dens_min,plot_interm)
            final_stack = final_stack + weak_emb_stack
            #weak_emb_stack would be 0/1, and only have 1 in has_weak_emb_set.
            #TODO: add weak_emb_stack to the result from CNN
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
            density_img = density_of_a_rect(bin_stem_stack[img_idx,:,:],dens_rect_window_width)
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
        #for both stem and leaf:
        #if the proportion of embolised pixels are smaller than emb_pro_th_min, treat as no emb
        num_emb_each_img_after = np.sum(np.sum(final_stack/255,axis=2),axis=1)
        treat_as_no_emb_idx = np.nonzero(num_emb_each_img_after/(img_nrow*img_ncol)<emb_pro_th_min)[0]
        final_stack[treat_as_no_emb_idx,:,:] = np.zeros(final_stack[treat_as_no_emb_idx,:,:].shape)
        if version_num >= 11 and is_stem==True:
            #for final_stack_strong_emb_cand, if the proportion of embolised pixels are smaller than emb_pro_th_min, treat as no emb
            num_emb_each_img_strong_emb = np.sum(np.sum(final_stack_strong_emb_cand/255,axis=2),axis=1)
            treat_as_no_emb_idx_strong_emb = np.nonzero(num_emb_each_img_strong_emb/(img_nrow*img_ncol)<emb_pro_th_min)[0]
            final_stack_strong_emb_cand[treat_as_no_emb_idx_strong_emb,:,:] = np.zeros(final_stack_strong_emb_cand[treat_as_no_emb_idx_strong_emb,:,:].shape)
    
    print("2nd stage done")
    
    #time
    print_used_time(start_time)
    
    if match==True:
        #combined with true tif file
        true_mask  = tiff.imread(up_2_date_tiff)#tiff.imread(img_folder+'/4 Mask Substack ('+str(start_img_idx)+'-'+str(end_img_idx-1)+') clean.tif')
        tm_start_img_idx = chunk_idx*(chunk_size-1)
        tm_end_img_idx = tm_start_img_idx+chunk_size-1
        true_mask = true_mask[tm_start_img_idx:tm_end_img_idx,:,:]
        if resize:
        	true_mask = mat_reshape(true_mask, height = img_height, width = img_width)
        combined_list = (true_mask,final_stack.astype(np.uint8),(bin_stack*255).astype(np.uint8))
    else:
        combined_list = (final_stack.astype(np.uint8),(bin_stack*255).astype(np.uint8))
    
    final_combined = np.concatenate(combined_list,axis=2)
    final_combined_inv =  -final_combined+255 #invert 0 and 255 s.t. background becomes white

    final_combined_inv_info = add_img_info_to_stack(final_combined_inv,img_paths,start_img_idx)
        
    if is_save==True:
        tiff.imsave(chunk_folder+'/predict.tif',255-final_stack.astype(np.uint8))
        tiff.imsave(chunk_folder+'/bin_diff.tif',255-(bin_stack*255).astype(np.uint8))
        if is_stem == True:
            tiff.imsave(chunk_folder+'/predict_before_rm_cc_geo.tif',255-final_stack_prev_stage.astype(np.uint8))
            if version_num >= 11:
                tiff.imsave(chunk_folder+'/stc_predict.tif',255-final_stack_strong_emb_cand.astype(np.uint8))
            if version_num >= 9.9:
                tiff.imsave(chunk_folder+'/weak_emb_stack.tif',255-weak_emb_stack.astype(np.uint8))
            #tiff.imsave(chunk_folder+'/predict_before_rm_cc_geo_small.tif',255-before_rm_cc_geo_stack_small.astype(np.uint8))
        tiff.imsave(chunk_folder+'/combined_4.tif', final_combined_inv_info)
        print("saved tif files")
    
    diff_min_sec=print_used_time(start_time)
    has_embolism = img_contain_emb(final_stack)
    
    if match==True:    
        true_has_emb = img_contain_emb(true_mask)
        '''
        Confusion Matrix
        To see the performance compared to those processed manually using ImageJ
        '''
        con_img_list = confusion_mat_img(has_embolism,true_has_emb)
    
        #print(con_img_list[0])
        #print("false positive img index",con_img_list[1])
        #print("false negative img index",con_img_list[2])
    
        F_positive = os.path.join(chunk_folder,'false_positive')
        F_negative = os.path.join(chunk_folder,'false_negative')
        T_positive = os.path.join(chunk_folder,'true_positive')
        if is_save == True:
            #create/empty folder
            con_output_path = [F_positive,F_negative,T_positive]
            for foldername in con_output_path:
                if not os.path.exists(foldername):#create new folder if not existed
                    os.makedirs(foldername)
                else:#empty the existing folder
                    shutil.rmtree(foldername)#delete
                    os.makedirs(foldername)#create
            #save images into false_positive, false_negative, true_positive subfolders
            for i in con_img_list[1]:
                plt.imsave(chunk_folder + "/false_positive/"+str(i+(start_img_idx-1))+'.jpg',final_combined_inv_info[i,:,:],cmap='gray')
            
            for i in con_img_list[2]:
                plt.imsave(chunk_folder + "/false_negative/"+str(i+(start_img_idx-1))+'.jpg',final_combined_inv_info[i,:,:],cmap='gray')
            
            for i in con_img_list[3]:
                plt.imsave(chunk_folder + "/true_positive/"+str(i+(start_img_idx-1))+'.jpg',final_combined_inv_info[i,:,:],cmap='gray')

            np.savetxt(chunk_folder + '/false_positive_index.txt', con_img_list[1]+(start_img_idx-1),fmt='%i')#integer format
            np.savetxt(chunk_folder + '/false_negative_index.txt', con_img_list[2]+(start_img_idx-1),fmt='%i')
            np.savetxt(chunk_folder + '/true_positive_index.txt', con_img_list[3]+(start_img_idx-1),fmt='%i')
            #but there could still be cases where there are false positive pixels in true positive img
            if version_num >= 11 and is_stem==True:
                '''
                v11: save txt files for strong emb candidate(stc)
                '''
                has_embolism_stc = img_contain_emb(final_stack_strong_emb_cand)
                con_img_list_stc = confusion_mat_img(has_embolism_stc,true_has_emb)
                np.savetxt(chunk_folder + '/stc_false_positive_index.txt', con_img_list_stc[1]+(start_img_idx-1),fmt='%i')#integer format
                np.savetxt(chunk_folder + '/stc_false_negative_index.txt', con_img_list_stc[2]+(start_img_idx-1),fmt='%i')
                np.savetxt(chunk_folder + '/stc_true_positive_index.txt', con_img_list_stc[3]+(start_img_idx-1),fmt='%i')
        con_df_px = confusion_mat_pixel(final_stack,true_mask)
        #print(con_df_px)
        total_num_pixel = final_stack.shape[0]*final_stack.shape[1]*final_stack.shape[2]
        #print(con_df_px/total_num_pixel)
        metrix_img = calc_metric(con_img_list[0])
        metrix_px = calc_metric(con_df_px)
        
        con_df_cluster, tp_area, fp_area,tp_height,fp_height,tp_width,fp_width = confusion_mat_cluster(final_stack, true_mask, has_embolism, true_has_emb, blur_radius=10, chunk_folder=chunk_folder,is_save=is_save)    
            
        metrix_cluster = calc_metric(con_df_cluster)
        
        #make sure fn are in weak_emb_cand_set
        if is_stem:
            weak_emb_cand_arr = np.asarray(weak_emb_cand_set)#list to array
            fn_in_weak_cand_vec = np.isin(weak_emb_cand_set,con_img_list[2])
            fn_in_weak_cand_idx = weak_emb_cand_arr[fn_in_weak_cand_vec]
        
        if version_num >= 9.9 and is_stem:
            #after rescue_weak_emb_by_dens, how many fn are being rescued to tp?
            has_weak_emb_arr = np.asarray(has_weak_emb_set)#list to array
            tp_in_has_weak_emb_vec = np.isin(has_weak_emb_arr,con_img_list[3])
            tp_in_has_weak_emb_idx = has_weak_emb_arr[tp_in_has_weak_emb_vec]
        

        if is_stem==True:
            true_emb_bubble_cc_max_area = subset_vec_set(bubble_cc_max_area_prop_vec,start_img_idx,np.where(true_has_emb)[0],output_row_name='bubble_cc_max_area_prop')
            poor_qual_bubble_cc_max_area = subset_vec_set(bubble_cc_max_area_prop_vec,start_img_idx,poor_qual_set_cc,output_row_name='bubble_cc_max_area_prop')
        if is_save ==True:
            with open (chunk_folder + '/confusion_mat_file.txt',"w") as f:
                f.write(f'used time: {diff_min_sec[0]} min {diff_min_sec[1]} sec\n\n')
                f.write('img level metric:\n')
                f.write(str(metrix_img))
                f.write(str("\n\n"))
                f.write(str(con_img_list[0]))
                f.write(str("\n\n"))
                f.write(f'false positive img index: {con_img_list[1]+(start_img_idx-1)}')
                f.write(str("\n\n"))
                f.write(f'false negative img index: {con_img_list[2]+(start_img_idx-1)}')
                f.write(str("\n\n"))
                f.write('pixel level metric:\n')
                f.write(str(metrix_px))
                f.write(str("\n\n"))
                f.write(f'con_df_px: \n {con_df_px}')
                f.write(str("\n\n"))
                f.write(f'probability of pix: \n {(con_df_px/total_num_pixel)}')
                f.write(str("\n\n"))
                f.write('cluster level metric:\n')
                f.write(str(metrix_cluster))
                f.write(str("\n\n"))
                f.write(f'con_df_cluster: \n {con_df_cluster}')
                if is_stem==True:
                    f.write(str("\n\n"))
                    f.write(f'percentage of img w/ bubble: {len(has_bubble_idx)}/{(img_num-1)} = {has_bubble_per} %\n')
                    f.write('img index with bubble:\n')
                    f.write(str(has_bubble_idx+(start_img_idx-1)))
                    f.write(str("\n\n"))
                    f.write(f'geo_invalid_emb_set:{geo_invalid_emb_set}\n')
                    f.write(f'cleaned_but_not_all_geo_invalid_set:{cleaned_but_not_all_geo_invalid_set}\n\n')
                    f.write(f'weak_emb_cand_set:{weak_emb_cand_set}\n')
                    if version_num >= 9.9:
                        f.write(f'\nhas_weak_emb_set(rescue_weak_emb_by_dens):{has_weak_emb_set}\n')
                        f.write(f'tp_in_has_weak_emb_idx:{tp_in_has_weak_emb_idx}\n')
                    elif len(con_img_list[2])>0:
                        f.write(f'the number fn in weak_emb_cand_set/the number of fn: {len(fn_in_weak_cand_idx)}/{len(con_img_list[2])}\n')
                        f.write(f'fn_in_weak_cand_idx:{fn_in_weak_cand_idx}\n')
    else:#match ==False, no more confusion matrix
        if is_stem==True:
            poor_qual_bubble_cc_max_area = subset_vec_set(bubble_cc_max_area_prop_vec,start_img_idx,poor_qual_set_cc,output_row_name='bubble_cc_max_area_prop')
        has_emb_idx = np.where(has_embolism)[0]
        if is_save ==True:
            with open (chunk_folder + '/results_summary.txt',"w") as f:
                f.write(f'used time: {diff_min_sec[0]} min {diff_min_sec[1]} sec\n\n')
                f.write(f'percentage of img predicted to have emb: {len(has_emb_idx)}/{(img_num-1)} = {round(len(has_emb_idx)/(img_num-1)*100,2)} %\n')
                f.write('img idx of predicted to have emb:\n')
                f.write(str(has_emb_idx))
                if is_stem==True:
                    f.write(f'percentage of img w/ bubble: {len(has_bubble_idx)}/{(img_num-1)} = {has_bubble_per} %\n')
                    f.write('img index with bubble:\n')
                    f.write(str(has_bubble_idx+(start_img_idx-1)))
                    f.write(str("\n\n"))
                    f.write(f'geo_invalid_emb_set:{geo_invalid_emb_set}\n')
                    f.write(f'cleaned_but_not_all_geo_invalid_set:{cleaned_but_not_all_geo_invalid_set}\n\n')
                    f.write(f'weak_emb_cand_set:{weak_emb_cand_set}\n')
                    if version_num >= 9.9:
                        f.write(f'\nhas_weak_emb_set(rescue_weak_emb_by_dens):{has_weak_emb_set}\n')
            
