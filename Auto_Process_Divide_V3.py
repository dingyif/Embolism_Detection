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
from detect_by_contour_v4 import plot_gray_img, to_binary,plot_img_sum, plot_overlap_sum
from detect_by_contour_v4 import add_img_info_to_stack, extract_foregroundRGB, detect_bubble
from detect_by_contour_v4 import img_contain_emb, extract_foreground, find_emoblism_by_contour, find_emoblism_by_filter_contour
from detect_by_contour_v4 import confusion_mat_img, confusion_mat_pixel,confusion_mat_cluster,calc_metric
from density import density_of_a_rect
from detect_by_filter_fx import median_filter_stack
import math

folder_list = []
has_tif = []
no_tif =[]
#disk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
disk_path = 'F:/Diane/Col/research/code/'
img_folder_rel = os.path.join(disk_path,"Done", "Processed")
all_folders_name = sorted(os.listdir(img_folder_rel), key=lambda s: s.lower())
all_folders_dir = [os.path.join(img_folder_rel,folder) for folder in all_folders_name]
#for i, img_folder in enumerate(all_folders_dir):
img_folder = all_folders_dir[17]

#Need to process c folder
#img_folder_c = all_folders_dir[14:]
#img_folder = img_folder_c[2]
print(f'img folder name : {img_folder}')

img_paths = sorted(glob.glob(img_folder + '/*.png'))
tiff_paths = sorted(glob.glob(img_folder + '/*.tif'))
match = False
#make sure it have tif file in there
if tiff_paths:
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
if match:
#    has_tif.append(i)##commented out for-loop for img_folder
    print("has tiff")    
    real_start_img_idx = int(cand_match.group('start_img_idx'))
    real_end_img_idx = int(cand_match.group('end_img_idx')) + 1
    #get which folder its processing now
    img_folder_name = os.path.split(img_folder)[1]
    print(f'Image Folder Name: {img_folder_name}')

    chunk_idx = 0#starts from 0
    chunk_size = 400#process 600 images a time
    #print('index: {}'.format(i))
    is_save = True

    if "stem" in img_folder.lower():
        is_stem = True
    elif "leaf" in img_folder.lower():
        is_stem = False
    else:
        sys.exit("Error: image folder name doesn't contain strings like stem or leaf")
    
    start_img_idx = 1+chunk_idx*(chunk_size-1)#real_start_img_idx+chunk_idx*(chunk_size-1)
    end_img_idx = min(start_img_idx+chunk_size-1,len(img_paths))
 
    if is_save==True:
        #create a "folder" for saving resulting tif files such that next time when re-run this program,
        #the resulting tif file won't be viewed as the most recent modified tiff file
        chunk_folder = os.path.join(img_folder,img_folder_name,'v9.2_'+str(chunk_idx)+'_'+str(start_img_idx)+'_'+str(end_img_idx))
        if not os.path.exists(chunk_folder):#create new folder if not existed
            os.makedirs(chunk_folder)
        else:#empty the existing folder
            shutil.rmtree(chunk_folder)#delete
            os.makedirs(chunk_folder)#create 
    
    img_num = end_img_idx-start_img_idx + 1

    img_re_idx = 0 #relative index for images in start_img_idx to end_img_idx
    for filename in img_paths[start_img_idx-1:end_img_idx]: #original img: 958 rowsx646 cols
        img=Image.open(filename).convert('L') #.convert('L'): gray-scale # 646x958
        img_array=np.float32(img) #convert from Image object to numpy array (black 0-255 white); 958x646
        #put in the correct data structure
        if img_re_idx == 0:
            img_nrow = img_array.shape[0]
            img_ncol = img_array.shape[1]
            img_stack = np.ndarray((img_num,img_nrow,img_ncol), dtype=np.float32)
        img_stack[img_re_idx] = img_array
        img_re_idx = img_re_idx + 1
    print("finish loading img")
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
    
    '''
    Foreground Background Segmentation
    for stem only, cuz there are clearly parts that are background (not stem)
    want to avoid  artifacts from shrinking of gels/ moving of plastic cover
    mask only the stem part to reduce false positive
    '''
    if is_stem==True:
        mean_img = np.mean(bin_stack,axis=0)
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
        is_stem_mat = np.ones(img_stack.shape)
        img_re_idx = 0
        for filename in img_paths[(start_img_idx-1):end_img_idx]: #original img: 958 rowsx646 cols
            imgRGB_arr=np.float32(Image.open(filename))#RGB image to numpy array
            imgGarray=imgRGB_arr[:,:,1] #only look at G layer
            #put in the correct data structure
            if img_re_idx==0 and is_save==True:
                is_stem_mat[img_re_idx] = extract_foregroundRGB(imgGarray,chunk_folder, blur_radius=10.0,expand_radius_ratio=2,is_save=True)
            else:
                is_stem_mat[img_re_idx] = extract_foregroundRGB(imgGarray,chunk_folder="", blur_radius=10.0,expand_radius_ratio=2)
            img_re_idx = img_re_idx + 1
        is_stem_mat2 = is_stem_mat[:-1,:,:]#drop the last img s.t. size would be the same as diff_stack
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
        plot_interm = False
        
        emb_pro_th_min = 0.0001
        rect_width_th = 0.05#100/1286
        rect_height_th = 0.02#200/959
        cc_rect_th = 0.35#ALCLAT1_leaf img_idx=167 true emb part:0.34, img_idx=73(false positive part:0.36)
    else:
        #is_stem_mat2 = np.ones(bin_stack.shape)
        area_th = 1
        area_th2 = 3#10
        c1_sz = max(round(25/646*max(img_ncol,img_nrow)),1)#in case img is flipped: in2_stem
        d1_sz = max(round(10/646*max(img_ncol,img_nrow)),1)
        final_area_th = 78
        max_emb_prop = 0.3#(has to be > 0.05 for a2_stem img_idx=224; has to > 0.19 for c4_stem img_idx=39; has to <0.29 for a4_stem img_idx=5; but has to be <0.19 for a4_stem img_idx=1
        #TODO: don't use max_emb_prop, but use img_sum?
        density_th = 0.35#<0.395 for cas5_stem #<0.36 in4_stem img_idx=232
        num_px_th = 50
        ratio_th=35  
        final_area_th2 = 80
        emb_freq_th = 0.05#5/349=0.014#a2_stem #depends on which stages the photos are taken
        cc_th = 3
        window_size = 200
        minRadius = 5
        bubble_area_prop_max = 0.2
        second_ero_kernel_sz = 3
        second_clo_kernel_sz = 10
        second_width_max = 0.85#cas2.2 img_idx=232, width=0.41/ img_idx=208, width=0.8
        second_rect_dens_max = 0.3
        second_area_max = 0.2
        plot_interm=False
    
    bin_stem_stack = bin_stack*is_stem_mat2
    '''1st stage'''
    if is_stem==True:
        filter_stack = median_filter_stack(is_stem_mat2*th_stack,5)#5 is better than max(round(5/646*img_ncol),1) for cas2.2_Stem
        print("median filter done")
        
        '''
        bubble detection
        '''
        bubble_stack,has_bubble_vec= detect_bubble(filter_stack, minRadius = minRadius)
        
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
        
        '''
        '''
        bubble_area_prop_vec=np.sum(np.sum(bubble_stack,2),1)/np.sum(np.sum(is_stem_mat2,2),1)
#        '''
#        To decide bubble area th--> considered as poor quality -->no emb:
#        '''
#        num_bins=50
#        n_th, bins_th, patches_th= plt.hist(bubble_area_prop_vec,bins=num_bins)
#        
#        #don't show 0
#        plt.figure()
#        plt.plot(bins_th[2:],n_th[1:])
#        #plt.xlim(bins_th[1],bins_th[-1])
#        plt.ylabel("frequency")
#        plt.xlabel("bubble area")
#        plt.title("histogram of bubble area (ignoring the 1st bin)")
#        
#        plt.figure()
#        plt.plot(range(len(bubble_area_prop_vec)),bubble_area_prop_vec)
#        plt.ylabel("bubble area")
#        plt.xlabel("image index")
#        plt.title("bubble area in each image")
#        
        
        poor_qual_set = np.where(bubble_area_prop_vec >= bubble_area_prop_max)[0]

        '''
        shift detection
        '''
        plot_overlap_sum(is_stem_mat, img_folder_name ,chunk_folder, is_save = True)
        
        
        for img_idx in range(0, bin_stack.shape[0]):
            if img_idx not in poor_qual_set:
                #the if condition above saves some time but doesn't change much results 
                #(cuz some predictions would be blocked by shift_th)
                #results effective on img_idx=5,8 of cas 2.2 stem
                #could introduce more false positive cuz in the stage "Don't count as embolism if it keeps appearing (probably is plastic cover)"
                #the not_emb_mask would become smaller 
                stem_area = np.sum(is_stem_mat2[img_idx,:,:])
                final_stack1[img_idx,:,:] = find_emoblism_by_filter_contour(bin_stem_stack,filter_stack,img_idx,stem_area = stem_area,final_area_th = final_area_th,
                                                            area_th=area_th, area_th2=area_th2,ratio_th=ratio_th,e2_sz=1,o2_sz=2,cl2_sz=2,c1_sz=c1_sz,d1_sz=d1_sz,
                                                            plot_interm=False,max_emb_prop=max_emb_prop,density_th=density_th,num_px_th=num_px_th)
            #TODO: closing/dilate param should depend on the width of stem (now it's depend on width of img)
    else:
        for img_idx in range(0, bin_stack.shape[0]):
            stem_area = np.sum(is_stem_mat2[img_idx,:,:])
            final_stack1[img_idx,:,:] = find_emoblism_by_contour(bin_stem_stack,img_idx,stem_area = stem_area,final_area_th = final_area_th,
                                                        area_th=area_th, area_th2=area_th2,ratio_th=ratio_th,e2_sz=1,o2_sz=2,cl2_sz=2,
                                                        plot_interm=False,max_emb_prop=max_emb_prop,density_th=density_th,num_px_th=num_px_th)
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
        
        if img_nrow > img_ncol:#normal direction
            middle_row = np.where(is_stem_mat2[0,round(img_nrow/2),:])[0]#take the middle row(in case top/bottom of stem isn't correctly detected cuz of bark)
        else:#flip for the special horizontal img:
            middle_row = np.where(is_stem_mat2[0,:,round(img_ncol/2)])[0]#take the middle row(in case top/bottom of stem isn't correctly detected cuz of bark)
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
                #TODO: discard cc wide but not tall (probably bark)
#                else:
#                    
#                cc_area_too_small = cc_area
#                for cc_idx in range(1,num_cc):#1: ignore bgd(0)

#        '''
#        remove imgs with bubble
#        '''
#        no_bubble_stack = 1-bubble_stack
#        final_stack=final_stack1*no_bubble_stack
        
        '''
        Don't count as embolism if it keeps appearing (probably is plastic cover)
        '''
        final_stack= np.zeros(bin_stack.shape)
        window_idx_max = math.ceil(bin_stack.shape[0]/window_size)
        for window_idx in range(0,window_idx_max):
            window_start_idx = window_idx*window_size
            window_end_idx = min((window_idx+1)*window_size,(end_img_idx-start_img_idx))
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
            not_emb_mask_exp = cv2.dilate(not_emb_mask.astype(np.uint8), np.ones((10,10),np.uint8),iterations = 1)#expand a bit
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
        #if the proportion of embolised pixels are smaller than emb_pro_th_min, treat as no emb
        num_emb_each_img_after = np.sum(np.sum(final_stack/255,axis=2),axis=1)
        treat_as_no_emb_idx = np.nonzero(num_emb_each_img_after/(img_nrow*img_ncol)<emb_pro_th_min)[0]
        final_stack[treat_as_no_emb_idx,:,:] = np.zeros(final_stack[treat_as_no_emb_idx,:,:].shape)
    
    print("2nd stage done")
    #combined with true tif file
    true_mask  = tiff.imread(up_2_date_tiff)#tiff.imread(img_folder+'/4 Mask Substack ('+str(start_img_idx)+'-'+str(end_img_idx-1)+') clean.tif')
    tm_start_img_idx = chunk_idx*(chunk_size-1)
    tm_end_img_idx = tm_start_img_idx+chunk_size-1
    true_mask = true_mask[tm_start_img_idx:tm_end_img_idx,:,:]
    combined_list = (true_mask,final_stack.astype(np.uint8),(bin_stack*255).astype(np.uint8))
    final_combined = np.concatenate(combined_list,axis=2)
    final_combined_inv =  -final_combined+255 #invert 0 and 255 s.t. background becomes white

    final_combined_inv_info = add_img_info_to_stack(final_combined_inv,img_paths,start_img_idx)
    if is_save==True:
        tiff.imsave(chunk_folder+'/combined_4.tif', final_combined_inv_info)
        tiff.imsave(chunk_folder+'/predict.tif',255-final_stack.astype(np.uint8))
        tiff.imsave(chunk_folder+'/bin_diff.tif',255-(bin_stack*255).astype(np.uint8))
        print("saved tif files")
    
    '''
    Confusion Matrix
    To see the performance compared to those processed manually using ImageJ
    '''
    print("================results===============")
    
    true_has_emb = img_contain_emb(true_mask)
    has_embolism = img_contain_emb(final_stack)
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
        #but there could still be cases where there are false positive pixels in true positive img
    con_df_px = confusion_mat_pixel(final_stack,true_mask)
    #print(con_df_px)
    total_num_pixel = final_stack.shape[0]*final_stack.shape[1]*final_stack.shape[2]
    #print(con_df_px/total_num_pixel)
    metrix_img = calc_metric(con_img_list[0])
    metrix_px = calc_metric(con_df_px)
    
    con_df_cluster = confusion_mat_cluster(final_stack, true_mask, has_embolism, true_has_emb, blur_radius=10)
    metrix_cluster = calc_metric(con_df_cluster)
    if is_save ==True:
        with open (chunk_folder + '/confusion_mat_file.txt',"w") as f:
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
            f.write(str("\n\n"))
            f.write(f'percentage of img w/ bubble: {len(has_bubble_idx)}/{(img_num-1)} = {has_bubble_per} %\n')
            f.write('img index with bubble:\n')
            f.write(str(has_bubble_idx+(start_img_idx-1)))
            f.write(str("\n\n"))
            f.write(f'poor_qual_set:\n{poor_qual_set}')
            f.write(str("\n\n"))
            f.write(f'poor_qual_set2:\n{poor_qual_set2}')
    
        

else:
#    no_tif.append(i)##commented out for-loop for img_folder
    print("no match")
##commented out for-loop for img_folder
#print ("has tif: ", has_tif)
#print ("no tif: ", no_tif)
#
#print("notif")
#for j in no_tif:
#    print(all_folders_name[j])
            
