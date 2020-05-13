# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:10:09 2020

@author: USER
"""

def get_each_stage_arg(version_num):
    '''
    the third digit after decimal point of "version_num" would be used to determine the run argument for each stage
    '''
    if (version_num*1000) % 10==1:#version num = XX.001
        #only runs detect embolism main stage (1st stage) w/o foreground seg and poor qual
        run_foreground_seg = False #to run the foreground segmentation (extracting stem part) or not
        run_poor_qual = False
        run_rm_big_emb = False
        run_rolling_window = False
        run_sep_weak_strong_emb = False
        run_rm_small_emb = False
    elif (version_num*1000) % 10==2:#version num = XX.002
        #runs detect embolism main stage (1st stage) w/foreground seg
        run_foreground_seg = True 
        run_poor_qual = False
        run_rm_big_emb = False
        run_rolling_window = False
        run_sep_weak_strong_emb = False
        run_rm_small_emb = False
    elif (version_num*1000) % 10==3:
        #runs detect embolism main stage (1st stage) w/foreground seg and poor_qual
        run_foreground_seg = True 
        run_poor_qual = True
        run_rm_big_emb = False
        run_rolling_window = False
        run_sep_weak_strong_emb = False
        run_rm_small_emb = False
    elif (version_num*1000) % 10==4:
        run_foreground_seg = True
        run_poor_qual = True
        run_rm_big_emb = True
        run_rolling_window = False
        run_sep_weak_strong_emb = False
        run_rm_small_emb = False
    elif (version_num*1000) % 10==5:
        run_foreground_seg = True
        run_poor_qual = True
        run_rm_big_emb = True
        run_rolling_window = True
        run_sep_weak_strong_emb = False
        run_rm_small_emb = False
    elif (version_num*1000) % 10==6:
        run_foreground_seg = True 
        run_poor_qual = True
        run_rm_big_emb = True
        run_rolling_window = True
        run_sep_weak_strong_emb = True
        run_rm_small_emb = False
    elif (version_num*1000) % 10==7:#7 is actually the same as else, but use 7 s.t. 
        #comparison with others are in the case where there's only one changed at a time
        run_foreground_seg = True 
        run_poor_qual = True
        run_rm_big_emb = True
        run_rolling_window = True
        run_sep_weak_strong_emb = True
        run_rm_small_emb = True
    elif (version_num*1000) % 10==8:#not 7 because has 2 differences compared with 6
        run_foreground_seg = True 
        run_poor_qual = True
        run_rm_big_emb = True
        run_rolling_window = False
        run_sep_weak_strong_emb = True
        run_rm_small_emb = True
    elif (version_num*1000) % 10==9:
        run_foreground_seg = True 
        run_poor_qual = True
        run_rm_big_emb = False
        run_rolling_window = False
        run_sep_weak_strong_emb = True
        run_rm_small_emb = True
    else:
        run_foreground_seg = True 
        run_poor_qual = True
        run_rm_big_emb = True
        run_rolling_window = True
        run_sep_weak_strong_emb = True
        run_rm_small_emb = True
    return run_foreground_seg,run_poor_qual,run_rm_big_emb,run_rolling_window,run_sep_weak_strong_emb,run_rm_small_emb