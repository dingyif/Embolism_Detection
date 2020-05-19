# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:57:23 2020

@author: USER
"""

import os

'''
user-specified input arguments
'''
dir1 = 'F:/processed_0221_from_thumb_drive/leaf'
dir2 = 'F:/ProcessedImages_0221_from_disk/leaf'

def get_all_foldername_short_lower(directory_path):
    all_folders_name = sorted(os.listdir(directory_path), key=lambda s: s.lower())
    
    all_folders_dir = [os.path.join(directory_path,folder) for folder in all_folders_name]
    all_folders_name_short_lower = []
    
    for folder_name in all_folders_name:#dacexc5.2_stem.XXXXXXXXXXX
        #in case there are floating number in experiment number
        #ignore[processed]
        split_by_underscore = folder_name.split("_")
        name_before_stem_leaf =  split_by_underscore[0]#dacexc5.2
        if len(split_by_underscore)>=2:
            stem_leaf_tag = split_by_underscore[1].split(".")[0]#"stem" or "leaf"
            folder_name_short = name_before_stem_leaf+"_"+stem_leaf_tag
            all_folders_name_short_lower.append(folder_name_short.lower())
        else:
            print("ignore ", folder_name)
    return all_folders_name_short_lower,all_folders_dir

folder_abbrev1 =  get_all_foldername_short_lower(dir1)
folder_abbrev2 =  get_all_foldername_short_lower(dir2)

match_foldername = [foldername for idx, foldername in enumerate(folder_abbrev1[0]) if foldername in set(folder_abbrev2[0])] 
if len(match_foldername)>0:
    print(match_foldername)
else:
    print("no folders with the same name in these 2 directories")

    