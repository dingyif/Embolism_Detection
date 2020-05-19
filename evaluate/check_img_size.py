# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:03:44 2020

@author: USER
"""

from PIL import Image
import glob
import numpy as np
import os#for creating/emptying folders
import re
import sys
import tifffile as tiff

import datetime
start_time = datetime.datetime.now()

img_folder_rel = 'F:/processed_0221_from_thumb_drive'#'E:/Diane/Col/research/code/Done/Processed'
#top directory that stores the folders w/raw image
has_processed = True#Working on Processed data or Unprocessed data
to_get_tif_size = True


all_folders_name = sorted(os.listdir(img_folder_rel), key=lambda s: s.lower())

if has_processed==False:
    all_folders_name_cleaned = []
    for filename in all_folders_name:
        if "processed" not in filename[0:10].lower():#filename doesn't start with processed
            all_folders_name_cleaned.append(filename)
else:
    all_folders_name_cleaned = all_folders_name
all_folders_dir = [os.path.join(img_folder_rel,folder) for folder in all_folders_name_cleaned]

def get_tif_size(tiff_path):
    true_mask  = tiff.imread(tiff_path)
    return(true_mask.shape)

print(f'folder_name:img_size:tif_size:is_large_img:tif_img_same_sz')
for img_folder in all_folders_dir:
    
    
    #read in either png or jpg files
    img_paths = sorted(glob.glob(img_folder + '/*.png'))
    if img_paths==[]:
        img_paths = sorted(glob.glob(img_folder + '/*.jpg'))
        
    if img_paths:
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
            print("no tiff match")
        else:
        #    has_tif.append(i)##commented out for-loop for img_folder
            if match==True:
                #print("has tiff")    
                real_start_img_idx = int(cand_match.group('start_img_idx'))
                real_end_img_idx = int(cand_match.group('end_img_idx')) + 1#+1 cuz tiff has the same size as diff_stack, which has a smaller size (by 1 img) than img_stack
                if to_get_tif_size==True:
                    tif_size = get_tif_size(up_2_date_tiff)
                else:
                    tif_size = tuple([0,0,0])#place holder for printing
                #ex: c5 stem
            #get which folder its processing now
            img_folder_name = os.path.split(img_folder)[1]
        
        
            if "stem" in img_folder.lower():
                is_stem = True
            elif "leaf" in img_folder.lower():
                is_stem = False
            else:
                sys.exit("Error: image folder name doesn't contain strings like stem or leaf")
            
            filename = img_paths[0]
            img=Image.open(filename).convert('L') #.convert('L'): gray-scale # 646x958
            img_array = np.float32(img)
            img_size = img_array.shape
            img_nrow = img_size[0]
            img_ncol = img_size[1]
            if img_nrow>2000 or img_ncol > 1900:
                is_large_img=True
            else:
                is_large_img=False
            
            tif_img_same_sz = ""#in case match == False (no tif)
            
            if match==True and to_get_tif_size==True:
                if img_nrow!= tif_size[1] or img_ncol != tif_size[2]:
                    tif_img_same_sz = False
                else:
                    tif_img_same_sz=True
            print(f'{img_folder_name}:{img_size}:{tif_size}:{is_large_img}:{tif_img_same_sz}')
    else:
        img_folder_name = os.path.split(img_folder)[1]
        print(f'{img_folder_name}: img_paths is empty')
        
