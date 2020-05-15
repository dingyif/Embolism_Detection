# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:55:25 2020

@author: USER
"""
import os,glob,sys,re,shutil
import cv2 #connected components
import pandas as pd
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from plotnine import *

def get_input_tif_path(use_predict_tif,has_processed,dir_path,img_folder):
    '''
    get input_tif_path based on user-specified arguments
    '''
    if use_predict_tif or not has_processed:
        input_tif_name = 'predict.tif'
        input_tif_folder = dir_path
        input_tif_path = os.path.join(input_tif_folder,input_tif_name)
    else:
        #paths for true.tif
        tiff_paths = sorted(glob.glob(img_folder + '/*.tif'))
        #make sure have tif file in there for the processed ones
        if tiff_paths:
            #To get the most recent modified tiff file
            #choose the most recent modified one if there are multiple of them
            
            #first order the files by modified time. 
            #And start from the most recently modified one, see if the filename includes "mask of result of substack".
            #if there's one match, break the loop.
            tiff_paths.sort(key=lambda x: os.path.getmtime(x))
            for up_2_date_tiff in tiff_paths[::-1]:
                #find files with following regex
                idx_pat = r'mask of result of substack \((?P<start_img_idx>\d{1,4})\-(?P<end_img_idx>\d{1,4})\)\.?'
                cand_match = re.search(idx_pat,up_2_date_tiff,re.I)
                if cand_match:
                    match = True
                    break
        if match==False:
            sys.exit("Error: no match for true tif file")
        else:
            input_tif_path = up_2_date_tiff
    return(input_tif_path)

def get_img_idx_from_one_file(file_path):
    '''
    input: 
        file_path: path file to be read
    output:
        img_idx: a vector of image index in the txt file
    '''
    with open (file_path) as file:
        img_idx = file.read().split('\n')
        #get rid of null, tranform str to int
        img_idx = [int(idx) for idx in img_idx if idx]
    return(img_idx)


def get_img_idx_from_all_files(dir_path,plot_fn,plot_fp):
    '''
    input: 
        dir_path: the directory where files are stored at
        plot_fn: (T/F)to plot from images in 'false_negative_index.txt' or not
        plot_fp: (T/F)to plot from images in 'false_positive_index.txt' or not
    output:
        emb_img_idx_plot: a vectir if image index to be plotted for this folder
        emb_num_plot: the number of images to be plotted
    '''
    tp_img_idx = get_img_idx_from_one_file(os.path.join(dir_path,'true_positive_index.txt'))
    
    emb_img_idx_plot = tp_img_idx #image index to plot
    
    #concate emb_img_idx_plot with fn_img_idx and fp_img_idx if their arguments(plot_fn,plot_fp) are set to TRUE
    if plot_fn:
      fn_img_idx = get_img_idx_from_one_file(os.path.join(dir_path,'false_negative_index.txt'))
      emb_img_idx_plot = emb_img_idx_plot + fn_img_idx #concatenate list
    
    if plot_fp:
      fp_img_idx = get_img_idx_from_one_file(os.path.join(dir_path,'false_positive_index.txt'))
      emb_img_idx_plot  = emb_img_idx_plot + fp_img_idx
    
    
    emb_img_idx_plot = sorted(emb_img_idx_plot) #sort in increasing order
    emb_num_plot = len(emb_img_idx_plot) #total number of embolism to plot (img level)
    return emb_img_idx_plot,emb_num_plot

def get_pixel_pos_and_cc_from_imgs(input_tiff,emb_img_idx_plot,cc_min_area,cc_min_area2):
    '''
    input:
        input_tiff
        emb_img_idx_plot: img index considered to be plotted
        cc_min_area: first min area threshold for cc (larger than cc_min_area2)
        cc_min_area2: smaller min area threshold for cc, in case there's no cc with area > cc_min_area.
    output:
        plot_mat_all: dataframe with 9 columns including pixel positions of embolism, embolism index (in img and cc level), and basic shape info for cc
        haven't adjusted for tiltness yet
    '''
    num_emb = 1#num_emb: (embolism index at img level), starts at 1
    cc_num_emb = 1#cc_num_emb(embolism index at cc level), starts at 1
    plot_mat_all = pd.DataFrame(columns=['row','col','number_emb','cc_num_emb','cc_width','cc_height','cc_area','cc_centroid_row','cc_centroid_col'])
    
    #initialinum_embe df with 3 cols: row,col,num_emb(embolism index at img level),cc_num_emb(embolism index at cc level)
    for j in emb_img_idx_plot:
        img_j = input_tiff[j] #0: background. 255: embolism
        smooth_img_j = cv2.morphologyEx(img_j.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))#connect a bit by closing(remove small holes)
        num_cc, mat_cc, stats, centroids  = cv2.connectedComponentsWithStats(smooth_img_j.astype(np.uint8), 8)#8-connectivity
        #number of c.c., centroids: 2 cols(col,row) 
        
        #below ignores background
        cc_width = stats[:,cv2.CC_STAT_WIDTH]
        cc_height = stats[:,cv2.CC_STAT_HEIGHT]
        cc_area = stats[:, cv2.CC_STAT_AREA]
        
        cc_big_enough_labels = np.where(cc_area[1:]>cc_min_area)[0]#ignore bgd:0
        if cc_big_enough_labels.size > 0:#at least have one cc big enough
            for cc_idx in (cc_big_enough_labels+1):#+1 cuz ignore bgd before
                row_col_cc = np.transpose(np.nonzero(mat_cc==cc_idx))#matrix: number of pixels in that cc x 2. Two cols: row,col
                num_px_in_cc = row_col_cc.shape[0]
                cc_stat = np.array([num_emb,cc_num_emb,cc_width[cc_idx],cc_height[cc_idx],cc_area[cc_idx],centroids[cc_idx][1],centroids[cc_idx][0]], ndmin=2)
                 #-1 cuz some statistics for cc ignores background
                cc_stat_rep = np.repeat(cc_stat,num_px_in_cc,axis=0)#horizontally repeat num_px_in_cc times
                plot_mat_j =  pd.DataFrame(np.column_stack((row_col_cc, cc_stat_rep)))
                plot_mat_j.columns =  plot_mat_all.columns#set to the same column names as plot_mat_all, s.t. can concat correctly
                plot_mat_all = pd.concat([plot_mat_all, plot_mat_j], ignore_index=True)
                
                ## not used
                #contours, _ = cv2.findContours(((mat_cc==cc_idx)*1).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                ##img_w_contour = cv2.drawContours((mat_cc==cc_idx)*1, contours, contourIdx=-1, color = 2, thickness = 1)#contourIdx=-1: draw all contours
                ##plt.imshow(img_w_contour)#slightly expanded contour (expand to ensure w/thickness = 1) will be drawn by pixel value 2. inside contour = 1. background  =0
                    
                cc_num_emb = cc_num_emb+ 1
        else:
            #try a smaller cc min area threshold
            cc_big_enough_labels2 = np.where(cc_area[1:]>cc_min_area2)[0]#ignore bgd:0
            if cc_big_enough_labels2.size > 0:#at least have one cc big enough
                for cc_idx in (cc_big_enough_labels2+1):#+1 cuz ignore bgd before
                    row_col_cc = np.transpose(np.nonzero(mat_cc==cc_idx))#matrix: number of pixels in that cc x 2. Two cols: row,col
                    num_px_in_cc = row_col_cc.shape[0]
                    cc_stat = np.array([num_emb,cc_num_emb,cc_width[cc_idx],cc_height[cc_idx],cc_area[cc_idx],centroids[cc_idx][1],centroids[cc_idx][0]], ndmin=2)
                     #-1 cuz some statistics for cc ignores background
                    cc_stat_rep = np.repeat(cc_stat,num_px_in_cc,axis=0)#horizontally repeat num_px_in_cc times
                    plot_mat_j =  pd.DataFrame(np.column_stack((row_col_cc, cc_stat_rep)))
                    plot_mat_j.columns =  plot_mat_all.columns#set to the same column names as plot_mat_all, s.t. can concat correctly
                    plot_mat_all = pd.concat([plot_mat_all, plot_mat_j], ignore_index=True)
                    cc_num_emb = cc_num_emb+ 1
                print("No cc in img_idx = ",j," has area >",cc_min_area,". But some cc area >",cc_min_area2)
            else:
                print("[CAUTION] No cc in img_idx = ",j," has area >",cc_min_area2)
        
        num_emb = num_emb + 1
    
    plot_mat_all["number_emb"] =plot_mat_all["number_emb"].astype(int)
    #plot_mat_all.dtypes
    #...
    #number_emb         float64
    ##convert the columns from float64 type to int type, s.t. it has the same type as embolism_table["number_emb"] , and thus can be merged later on
    return(plot_mat_all)

def extract_by_index_list(data,index_list,to_array):
    '''
    extract elements from data given a list of indices (index_list)
    then convert the output list to array, if to_array is True
    '''
    output = [data[i] for i in index_list]
    if to_array:#convert list to array
        output = np.asarray(output)
    return(output)

def get_img_filenames(img_folder):
    '''
    get a list of filenames under img_folder with filenames ending with ".png" or ".jpg"
    also return the file extension
    '''
    #get the embolism time data. (i.e. image file names)
    all_files = []
    img_extension = ""
    for file in sorted(os.listdir(img_folder)):
        if file.endswith(".png"):
            all_files.append(file)
    
    if all_files==[]:
        for file in sorted(os.listdir(img_folder)):
            if file.endswith(".jpg"):
                all_files.append(file)
    else:#all_files isn't empty
        img_extension = ".png"
    
    if all_files==[]:
        sys.exit("Error: Image files aren't png or jpg")
    elif len(img_extension)==0:#doesn't have img_extension yet, but all_files is not empty anymore
        img_extension = ".jpg"
    return(all_files,img_extension)

def get_date_time_from_filenames(all_files,img_extension):
    '''
    get the photo taken time (as a list)
    i.e. convert '20190725-103910.png' to '2019-07-25 10:39:10' 
    given a list of all filenames(all_files) and file extension(img_extension)
    '''
    #date_time_list = list()
    
    #for file in all_files:
    #    time_str = file.split(img_extension)[0]
        #put all year month day in formate
    #    Year = time_str[0:4]
    #    Month = time_str[4:6]
    #    Day =  time_str[6:8]
    #    YMD = '-'.join([Year,Month,Day])
    #    Hour = time_str[9:11]
    #    Minute = time_str[11:13]
    #    Secs =  time_str[13:15]
    #    HMS = ':'.join([Hour,Minute,Secs])
    #    date_time = ' '.join([YMD,HMS])
    #    date_time_list.append(date_time)
    time_str_list = [file.split(img_extension)[0] for file in all_files]
    date_time_list = [datetime.datetime.strptime(dt_str,'%Y%m%d-%H%M%S') for dt_str in time_str_list]
    return(date_time_list)

def get_time_wrt_start_time(date_time_list_ori,num_imgs,emb_num_plot,emb_img_idx_plot):
    '''
    get time difference w.r.t to starting time(1st img)
    and extract those information only for images with embolism (emb_num_plot,emb_img_idx_plot)
    '''
    #compute the relative time wrt to starting time(1st img)
    second_img_date_time = date_time_list_ori[1:(num_imgs+1)]#[Diane] one image later than Dingyi's
    start_time = date_time_list_ori[0]
    #change the different
    diff_time_list = [(end_dt - start_time).total_seconds()/60 for end_dt in second_img_date_time]
    #for date_time in second_img_date_time:
    #  diff_time = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    #  diff_time_in_min = diff_time.total_seconds() / 60#can be float number
    #  diff_time_list.append(diff_time_in_min)
    
    diff_time_list_int = [math.floor(float(x)) for x in diff_time_list]
    #put in the data frame to show table
    number_emb = np.arange(emb_num_plot)+1 #starts from 1
    embolism_time = extract_by_index_list(second_img_date_time,emb_img_idx_plot,to_array=True)
    diff_time_list_plot = extract_by_index_list(diff_time_list_int,emb_img_idx_plot,to_array=True)
    embolism_table = pd.DataFrame(np.column_stack((embolism_time,diff_time_list_plot,number_emb)))
    embolism_table.columns = ['embolism_time','time_since_start(mins)','number_emb']
    embolism_table["number_emb"] =embolism_table["number_emb"].astype(int)
    embolism_table["time_since_start(mins)"] =embolism_table["time_since_start(mins)"].astype(int)
    ##embolism_table.dtypes
    #...
    #number_emb                object
    #convert the columns from OBJECT type to INT type, s.t. it has the same type as plot_mat_all["number_emb"] , and thus can be merged later on
    return(second_img_date_time, diff_time_list_int, embolism_table)

def create_or_empty_folder(target_folder_path, to_empty):
  if not os.path.exists(target_folder_path):#create new folder if not existed
    os.makedirs(target_folder_path)
  elif to_empty:#empty the existing folder
    shutil.rmtree(target_folder_path)#delete
    os.makedirs(target_folder_path)#create

def get_sum_bin_tiff(input_tiff,row_num,col_num,emb_num_plot,emb_img_idx_plot):
    '''
    create binary tiff that records whether an embolism has occured at the same pixel position across time
    (binary in the sense that each image in the tiff neglects the "frequency" of embolism on the same pixel position)
    This will be used in shiny app: "Image"
    '''
    #data to feed image to plot
    sum_tiff = np.zeros((row_num,col_num))
    plot_tiff = np.zeros((emb_num_plot,row_num,col_num))
    i = 0
    for number in emb_img_idx_plot:
      sum_tiff = sum_tiff + input_tiff[number]/255
      sum_tiff = 1*(sum_tiff>0)#compress to 0 and 1 only
      #make the mat sparse to easy plot
      plot_tiff[i] = sum_tiff
      i = i+1
    return(plot_tiff)

def plot_point_one_emb_or_one_img(points_df,row_num,col_num,cc_num_emb=-1,num_emb=-1,fig_size=5):
    fig, ax = plt.subplots(figsize = (fig_size,fig_size))
    ax.set_xlim(0, col_num)
    ax.set_ylim(0, row_num)
    #look at a specific (based on cc_num_emb or num_emb) embolism event
    if cc_num_emb >0:
        sample_point_Z = points_df.loc[points_df.cc_num_emb == cc_num_emb,]
        num_emb = set(sample_point_Z.number_emb)
        ax.set_title('emb_num: ' + str(num_emb)+ ' cc_num_emb: ' + str(cc_num_emb))
    elif num_emb >0:
        sample_point_Z = points_df.loc[points_df.number_emb == num_emb,]
        ax.set_title('emb_num: ' + str(num_emb))
    #lets see how points look like 
    _ = plt.plot(sample_point_Z.col, sample_point_Z.row,'o',color='#20639B',alpha = 0.1, markersize = 4)

def plot_point(points_df,xlim,ylim,fig_size=5):
    fig, ax = plt.subplots(figsize = (fig_size,fig_size))
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    #lets see how points look like 
    _ = plt.plot(points_df.col, points_df.row,'o',color='#20639B',alpha = 0.1, markersize = 4)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def do_pca(X, row_num, col_num, to_plot=False):
    '''
    do PCA on an pixel position array (X)
    X: shape = number of points x 2 (x:col,y:row)
    '''
    pca = PCA(n_components=2)
    pca.fit(X)
    
    if to_plot:
        
        fig, ax = plt.subplots(figsize = (5,5))
        ax.set_xlim(0, col_num)
        ax.set_ylim(0, row_num)
        plt.scatter(X[:,0], X[:,1], alpha=0.2)#if there's TypeError: '(slice(None, None, None), 0)' is an invalid key, change to "iloc" i.e. plt.scatter(X.iloc[:,0], X.iloc[:,1], alpha=0.2)
        for length, vector in zip(pca.explained_variance_, pca.components_):
            scale_magnitude_tmp = np.sqrt(length)
            if scale_magnitude_tmp < min(row_num,col_num)/20:
                scale_magnitude =min(row_num,col_num)/20
            elif scale_magnitude_tmp > min(row_num,col_num)/2:
                scale_magnitude = min(row_num,col_num)/2
            else:
                scale_magnitude = scale_magnitude_tmp
            v = vector *scale_magnitude
            draw_vector(pca.mean_, pca.mean_ + v, ax)
    
    return pca

def get_upright_mat(plot_mat_time, row_num,col_num,inv_c):
    '''
    apply PCA to get tilted direction
    then get the new cc_width, cc_height, and cc_centroid after rotated back to the upright direction
    (area is rotation invariant)
    '''
    pt_mat_upright = pd.DataFrame(columns=['row','col','number_emb', 'cc_num_emb', 'cc_width', 
                                           'cc_height','cc_area', 'cc_centroid_row', 'cc_centroid_col', 
                                           'embolism_time','time_since_start(mins)', 
                                           'pca1_x', 'pca1_y', 'pca2_x', 'pca2_y','pca_explained_var1','pca_explained_var2'])
    
    for cc_i in np.unique(plot_mat_time.cc_num_emb):
        sample_point_Z = plot_mat_time.loc[plot_mat_time.cc_num_emb == cc_i,]
        X = sample_point_Z[['col','row']]#extract only pixel position columns. have to be this order ['col','row'] so that when doing PCA the order would be the same as (x,y)
        pca = do_pca(X, row_num, col_num, to_plot=False)
        
        dir_1 = pca.components_[0]#direction of 1st pca component
        cos_theta = dir_1[0]
        sin_theta = dir_1[1]
        
        if abs(cos_theta) < abs(sin_theta):
            #direction isn't horizontal
            #create rotation matrix .
            if sin_theta>=0: # quadrant I & II. want to rotate 90-theta degree
                rotation_mat = np.array(((sin_theta, -cos_theta), (cos_theta, sin_theta)))
            else:#quadrant III & IV. want to rotate 270-theta degree
                rotation_mat = np.array(((-sin_theta, cos_theta), (-cos_theta, -sin_theta)))
        
            #might be floated number after rotation
            #rotate not by origin but by the pivot
            #Not sure....
            #pivot_center = np.array((np.mean(plot_mat_time['col']),np.mean(plot_mat_time['row']))) #stem center
            #pivot_center = pca.mean_ #(which is the center of c.c.)
            if inv_c:#stem curved like inverse c
                if sin_theta*cos_theta>=0: # quadrant I & III
                    pivot_center = np.array((max(X['col']),max(X['row'])))#top right 
                else: # quadrant II&IV
                    pivot_center = np.array((max(X['col']),min(X['row'])))#bottom right
            else:#stem curved like c
                if sin_theta*cos_theta>=0: # quadrant I & III
                    pivot_center = np.array((min(X['col']),min(X['row'])))#bottom left
                else: # quadrant II&IV
                    pivot_center = np.array((min(X['col']),max(X['row'])))#top left
                
                
            X_rotated = (np.dot(rotation_mat,(X-pivot_center).T)).T + pivot_center#rotated by "90-theta" degree         
            cc_width = max(X_rotated[:,0])-min(X_rotated[:,0])
            cc_height = max(X_rotated[:,1])-min(X_rotated[:,1])
            prev_center = np.array((np.unique(sample_point_Z['cc_centroid_col'])[0],np.unique(sample_point_Z['cc_centroid_row'])[0]))
            center_rotated = (np.dot(rotation_mat,(prev_center-pivot_center).T)).T+pivot_center
            cc_centroid_col = center_rotated[0]
            cc_centroid_row = center_rotated[1]
            
            num_px_in_cc = sample_point_Z.shape[0]
            rotated_mat_Z = sample_point_Z.copy()
            rotated_mat_Z[['col','row']] = X_rotated
            rotated_mat_Z['cc_width'] = cc_width
            rotated_mat_Z['cc_height'] = cc_height
            rotated_mat_Z['cc_centroid_row'] = cc_centroid_row
            rotated_mat_Z['cc_centroid_col'] = cc_centroid_col
            pca_info = np.concatenate((pca.components_.flatten(),pca.explained_variance_), axis=None)
            pca_info_rep = pd.DataFrame(np.tile(pca_info,(num_px_in_cc,1)))#horizontally repeat num_px_in_cc times
            #num_px_in_cc x 6 (4 pca components, 2 pca explained variances)
            pt_mat_upright_Z = pd.concat([rotated_mat_Z.reset_index(drop=True),pca_info_rep.reset_index(drop=True)],axis=1) 
            pt_mat_upright_Z.columns =  pt_mat_upright.columns#set to the same column names as plot_mat_all, s.t. can concat correctly
            #else run into warning: FutureWarning: Sorting because non-concatenation axis is not aligned.
            pt_mat_upright = pd.concat([pt_mat_upright,pt_mat_upright_Z ], ignore_index=True)
        else:#no rotation
            print("cc_num_emb",cc_i,"didn't rotate, because pca direction is horizontal direction")
            rotated_mat_Z = sample_point_Z.copy()
            pca_info = np.array((0,1,1,0,-1,-1))
            pca_info_rep = pd.DataFrame(np.tile(pca_info,(num_px_in_cc,1)))#horizontally repeat num_px_in_cc times
            #num_px_in_cc x 6 (4 pca components, 2 pca explained variances)
            pt_mat_upright_Z = pd.concat([rotated_mat_Z.reset_index(drop=True),pca_info_rep.reset_index(drop=True)],axis=1) 
            pt_mat_upright_Z.columns =  pt_mat_upright.columns#set to the same column names as plot_mat_all, s.t. can concat correctly
            #else run into warning: FutureWarning: Sorting because non-concatenation axis is not aligned.
            pt_mat_upright = pd.concat([pt_mat_upright,pt_mat_upright_Z ], ignore_index=True)
            
    
    #convert to correct type s.t. plot axis looks nice later on
    pt_mat_upright["number_emb"] =pt_mat_upright["number_emb"].astype(int)
    pt_mat_upright["time_since_start(mins)"] =pt_mat_upright["time_since_start(mins)"].astype(int)
    return pt_mat_upright

def scatter_ggplot(df,y_colname,x_colname,title_str):
    print(ggplot(df, aes(y=y_colname, x=x_colname)) + ggtitle(title_str) + geom_point())

def scatter_plt(df,y,x,title_str,fig_size=5):
    fig, ax = plt.subplots(figsize = (fig_size,fig_size))
    plt.scatter(df[x], df[y])
    plt.title(title_str)
    plt.xlabel(x)
    plt.ylabel(y)

    plt.show()
