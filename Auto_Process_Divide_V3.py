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
from detect_by_contour_v4 import plot_gray_img, to_binary,plot_img_sum
from detect_by_contour_v4 import add_img_info_to_stack, extract_foregroundRGB
from detect_by_contour_v4 import img_contain_emb, extract_foreground, find_emoblism_by_contour, find_emoblism_by_filter_contour
from detect_by_contour_v4 import confusion_mat_img, confusion_mat_pixel,calc_metric
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
img_folder = all_folders_dir[34]

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

    chunk_idx = 1#starts from 0
    chunk_size = 200#process 600 images a time
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
        chunk_folder = os.path.join(img_folder,img_folder_name,'v8_'+str(chunk_idx)+'_'+str(start_img_idx)+'_'+str(end_img_idx))
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
        is_stem_mat2 = np.ones(bin_stack.shape)
        img_re_idx = 0
        for filename in img_paths[(start_img_idx-1):(end_img_idx-1)]: #original img: 958 rowsx646 cols
            imgRGB_arr=np.float32(Image.open(filename))#RGB image to numpy array
            imgGarray=imgRGB_arr[:,:,1] #only look at G layer
            #put in the correct data structure
            if img_re_idx==0:
                is_stem_mat2[img_re_idx] = extract_foregroundRGB(imgGarray,chunk_folder, blur_radius=10.0,expand_radius_ratio=2,is_save=True)
            else:
                is_stem_mat2[img_re_idx] = extract_foregroundRGB(imgGarray,chunk_folder, blur_radius=10.0,expand_radius_ratio=2)
            img_re_idx = img_re_idx + 1
        print("finish is_stem_mat2")
    
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
    else:
        #is_stem_mat2 = np.ones(bin_stack.shape)
        area_th = 1
        area_th2 = 3#10
        c1_sz = max(round(25/646*max(img_ncol,img_nrow)),1)#in case img is flipped: in2_stem
        d1_sz = max(round(10/646*max(img_ncol,img_nrow)),1)
        final_area_th = 78
        shift_th = 0.3#(has to be > 0.05 for a2_stem img_idx=224; has to > 0.19 for c4_stem img_idx=39; has to <0.29 for a4_stem img_idx=5; but has to be <0.19 for a4_stem img_idx=1
        #TODO: don't use shift_th, but use img_sum?
        density_th = 0.4#<0.395 for cas5_stem
        num_px_th = 50
        ratio_th=35  
        final_area_th2 = 80
        emb_freq_th = 0.05#5/349=0.014#a2_stem #depends on which stages the photos are taken
        cc_th = 3
        window_size = 200
    
    bin_stem_stack = bin_stack*is_stem_mat2
    '''1st stage'''
    if is_stem==True:
        filter_stack = median_filter_stack(is_stem_mat2*th_stack,5)#5 is better than max(round(5/646*img_ncol),1) for cas2.2_Stem
        print("median filter done")
        for img_idx in range(0, bin_stack.shape[0]):
            stem_area = np.sum(is_stem_mat2[img_idx,:,:])
            final_stack1[img_idx,:,:] = find_emoblism_by_filter_contour(bin_stem_stack,filter_stack,img_idx,stem_area = stem_area,final_area_th = final_area_th,
                                                        area_th=area_th, area_th2=area_th2,ratio_th=ratio_th,e2_sz=1,o2_sz=2,cl2_sz=2,c1_sz=c1_sz,d1_sz=d1_sz,
                                                        plot_interm=False,shift_th=shift_th,density_th=density_th,num_px_th=num_px_th)
            #TODO: closing/dilate param should depend on the width of stem (now it's depend on width of img)
    else:
        for img_idx in range(0, bin_stack.shape[0]):
            stem_area = np.sum(is_stem_mat2[img_idx,:,:])
            final_stack1[img_idx,:,:] = find_emoblism_by_contour(bin_stem_stack,img_idx,stem_area = stem_area,final_area_th = final_area_th,
                                                        area_th=area_th, area_th2=area_th2,ratio_th=ratio_th,e2_sz=1,o2_sz=2,cl2_sz=2,
                                                        plot_interm=False,shift_th=shift_th,density_th=density_th,num_px_th=num_px_th)
    print("1st stage done")
    '''2nd stage: reduce false positive'''
    if is_stem==True:
        '''
        Don't count as embolism if it keeps appearing (probably is plastic cover)
        '''
        final_stack= np.zeros(bin_stack.shape)
        window_idx_max = math.ceil(bin_stack.shape[0]/window_size)
        for window_idx in range(0,window_idx_max):
            window_start_idx = window_idx*window_size
            window_end_idx = min((window_idx+1)*window_size,(end_img_idx-start_img_idx-1))
            #"-1": because final_stack.shape[0] = img_stack.shape[0]-1, "-start_img_idx": because start_img_idx might not start at 0
            current_window_size = window_end_idx-window_start_idx#img_num mod window_size might not be 0 
            
            substack = np.sum(final_stack1[window_start_idx:window_end_idx],axis=0)/255
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
            final_stack[window_start_idx:window_end_idx,:,:] = np.repeat(emb_cand_mask[np.newaxis,:,:],current_window_size,0)*final_stack1[window_start_idx:window_end_idx,:,:]
        
        #treat whole img as no emb, if the number of embolized pixels is too small in an img
        num_emb_each_img = np.sum(np.sum(final_stack/255,axis=2),axis=1)#the number of embolized pixels in each img
        emb_cand_each_img = (num_emb_each_img>final_area_th2)*1#a vector of length = (img_num-1), 0 means the img should be treated as no emb, 1 means to keep the same way as it is
        emb_cand_each_img1 = np.repeat(emb_cand_each_img[:,np.newaxis],final_stack.shape[1],1)
        emb_cand_stack = np.repeat(emb_cand_each_img1[:,:,np.newaxis],final_stack.shape[2],2)
        final_stack = emb_cand_stack*final_stack
        #final_stack=final_stack1
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
        print("saved combined_4.tif")
    
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
            