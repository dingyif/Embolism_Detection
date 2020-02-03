# -*- coding: utf-8 -*-
import glob
import os
import re

folder_list = []
has_tif = []
no_tif =[]
disk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#img_folder_rel = os.path.join(disk_path,"Done", "Processed")
img_folder_rel = 'F:/Diane/Col/research/code/Done/Processed'
all_folders_name = os.listdir(img_folder_rel)
all_folders_dir = [os.path.join(img_folder_rel,folder) for folder in all_folders_name]
for i,img_folder in enumerate(all_folders_dir):
    #img_folder = all_folders_dir[0]
    img_paths = sorted(glob.glob(img_folder + '/*.png'))
    tiff_paths = sorted(glob.glob(img_folder + '/*.tif'))
    match=False
    #make sure it have tif file in there
    if tiff_paths:
        #To get tif with filename "mask of result of substack"
        #choose the most recent modified one if there are multiple of them
        tiff_paths.sort(key=lambda x: os.path.getmtime(x))
        for up_2_date_tiff in tiff_paths[::-1]:
            #find files with the following regex
            idx_pat = r'mask of result of substack \((?P<start_img_idx>\d{1,4})\-(?P<end_img_idx>\d{1,4})\)\.?'
            cand_match = re.search(idx_pat,up_2_date_tiff,re.I)
            if cand_match:
                match=True
                break
        
        tiff_name = os.path.split(up_2_date_tiff)[1]
        #get which folder its processing now
        img_folder_name = os.path.split(img_folder)[1]
        print("Image Folder Name:",img_folder_name)
        #want to automate the index as well, so put into group
        #we can just use start_img_idx and end_img_idx
    if match:
        has_tif.append(i)
        #print("has tiff")
    else:
        no_tif.append(i)
        #print("No match")

print("has_tif",has_tif)
print("no_tif",no_tif)

print("no_tif")
for j in no_tif:
    print(all_folders_name[j])