# -*- coding: utf-8 -*-
from PIL import Image
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
import re

folder_list = []
disk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
img_folder_rel = os.path.join(disk_path,"Done", "Processed")
all_folders_name = os.listdir(img_folder_rel)
all_folders_dir = [os.path.join(img_folder_rel,folder) for folder in all_folders_name]
for i, folder_dir_path in enumerate(all_folders_dir):
    img_paths = sorted(glob.glob(folder_dir_path + '/*.png'))
    tiff_paths = sorted(glob.glob(folder_dir_path + '/*.tif'))
    #make sure it have tif file in there
    if tiff_paths:
        #To get the most recent modified tiff file
        tiff_paths.sort(key=lambda x: os.path.getmtime(x))
        up_2_date_tiff = tiff_paths[::-1][0]
        tiff_name = os.path.split(up_2_date_tiff)[1]
        #want to automate the index as well, so put into group
        #we can just use start_img_idx and end_img_idx
        idx_pat = r'mask of result of substack \((?P<start_img_idx>\d{1,4})\-(?P<end_img_idx>\d{1,4})\)\.?'
        match = re.search(idx_pat,tiff_name,re.I)
        if match:
            start_img_idx = match.group('start_img_idx')
            end_img_idx = match.group('end_img_idx')
            print(start_img_idx)
            print(end_img_idx)
            print('index: {}'.format(i))
