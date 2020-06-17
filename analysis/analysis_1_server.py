#!/usr/bin/env python
# coding: utf-8

# # load modules

# In[1]:

import os,sys
import pandas as pd
import numpy as np 
import argparse
import pickle#save multiple variables into a pickle file

from tqdm import tqdm_notebook as tqdm
import datetime



#add folder to the system-path at runtime, for importing file in another folder
if '../shiny_files' not in sys.path:
    sys.path.insert(0, '../shiny_files')
import func_get_data as fx

'''
only include the parts to output all_inter_event_dist_time.csv and total_summary_stats.csv
'''

start_time = datetime.datetime.now()

# # load the data 

# In[2]:
parser  = argparse.ArgumentParser(description='Server version for get_data.py')
parser.add_argument('--input_dir', type = str, default="/rigel/stats/projects/emb_proj/output_get_data")
args = parser.parse_args()

version_name = 'true_tif' #'v11'
input_dir = args.input_dir
#input_dir = 'F:\output_get_data'

#use the below arguments and save different csv file
use_predict_tif = False #(False):use true labels tif . (True or has_processed=False): use predict.tif 
#default is to plot tp emb.
plot_fn = True #(TRUE):plot fn emb. from false_positive_index.txt 
plot_fp = False 

'''
Paths
'''
disk_path = os.path.join(input_dir,version_name)
all_folders_name = np.sort(os.listdir(disk_path))

all_folders_dir = [os.path.join(disk_path,folder) for folder in all_folders_name]

n_folders = len(all_folders_name)#[Diane 0522]

#output folder name based on user-specified arguments
if use_predict_tif:
    folder_tag1 = 'pred_tif'
else:
    folder_tag1= 'true_tif'

if plot_fn:
    folder_tag2 = 'has_fn'
else:
    folder_tag2 = 'no_fn'

if plot_fp:
    folder_tag3 = 'has_fp'
else:
    folder_tag3 = 'no_fp'


# # Define the functions

# In[3]:


def compute_cc_emb_info(folder_idx):
    folder_name_short = all_folders_name[folder_idx]
    dir_path = all_folders_dir[folder_idx]
    input_folder_tag = '_'.join((folder_tag1,folder_tag2,folder_tag3))
    input_folder = os.path.join(dir_path,input_folder_tag)
    pickle_dict = pickle.load(open(os.path.join(input_folder,'shinydata.pkl'),"rb"))#dictionary
    plot_mat_time = pickle_dict['plot_mat_time']
    cc_emb_info = plot_mat_time.drop(['row','col'], axis = 1)#drop two columns that are of pixel level (smaller than cc level)
    cc_emb_info = cc_emb_info.drop_duplicates()#keep unique rows
    x_mean = np.mean(plot_mat_time['col'])#col mean of one embolism event
    #[Diane 0522]returns plot_mat_time, else there'll be err in compute_col_dist
    return cc_emb_info, x_mean, folder_name_short, plot_mat_time 


# In[4]:


def compute_col_dist(cc_emb_info,x_mean,plot_mat_time):
    cc_centroid_col_dist_to_mean = cc_emb_info['cc_centroid_col'].apply(lambda x: abs(x - x_mean))
    cc_centroid_col_dist_to_mean = pd.concat([cc_centroid_col_dist_to_mean.reset_index(drop=True),cc_emb_info['time_since_start(mins)'].reset_index(drop=True)],axis=1)
    cc_centroid_col_dist_to_mean = cc_centroid_col_dist_to_mean.rename(columns = {'cc_centroid_col':'cc_centroid_col_dist_to_mean'})
    cc_col_dist_mean = cc_centroid_col_dist_to_mean.groupby(['time_since_start(mins)'])['cc_centroid_col_dist_to_mean'].mean().reset_index()
    cc_col_dist_mean = cc_col_dist_mean.rename(columns = {'cc_centroid_col_dist_to_mean':'cc_cen_col_dist_mean'})
    #summary_statistics = plot_mat_time.groupby('number_emb').mean().iloc[:,-6:]#img level, takes the "weighted" mean of other columns, "weight" would be porportional to the number of pixels in an embolism event
    summary_statistics = plot_mat_time.groupby('cc_num_emb').mean().groupby('number_emb').mean().drop(['row','col'], axis = 1)#[Diane 0522]
    summary_statistics = summary_statistics.merge(cc_col_dist_mean, on = 'time_since_start(mins)')
    summary_statistics['folder_name'] = folder_name_short.lower()
    return summary_statistics


# In[5]:


def summary_statistics_all(n_folders):
    total_summary_stats = pd.DataFrame()
    for i in tqdm(range(n_folders)):
        cc_emb_info, x_mean, folder_name_short , plot_mat_time = compute_cc_emb_info(i)#[Diane 0522]
        summary_statistics = compute_col_dist(cc_emb_info,x_mean, plot_mat_time)
        summary_statistics['folder_name'] = folder_name_short.lower()#[Diane 0522]
        total_summary_stats = total_summary_stats.append(summary_statistics)
    return total_summary_stats


# # Cumulative embolized area vs time

# In[6]:


cc_area_summary = pd.DataFrame()
for i in tqdm(range(n_folders)):
    cc_emb_info, x_mean, folder_name_short, _ = compute_cc_emb_info(i)#[Diane 0522]
    summary_statistics = cc_emb_info.groupby('time_since_start(mins)').agg({'cc_area':'sum','number_emb':'median'}).reset_index()
    summary_statistics['folder_name'] = folder_name_short.lower()
    summary_statistics['cumsum_cc_area'] = summary_statistics.cc_area.cumsum()/summary_statistics.cc_area.cumsum().max()
    cc_area_summary = cc_area_summary.append(summary_statistics)




# In[10]:


total_summary_stats = summary_statistics_all(n_folders)#[Diane 0522]
all_folders_name_short = np.unique(total_summary_stats.folder_name)#[Diane 0522]



# # Potential Problem of the above total_summary_stats:
# 
# - <span style="color:red">  Didn't consider shrinkage/stem curved yet </span>
# - <span style="color:red">  Didn't consider multiple embolism events in one image </span> As we grouped it by 'time_since_start(mins)' or 'number_emb'(at img level) and took the mean. We assumed the emb. in one img are all from the same embolism events, which might not be true.
#     - cc_num_emb might not be the same as emb event index (cuz same emb. event might be broken into small pieces)
#     - number_emb might not be the same as emb event index (cuz multiple embolism events can take place in one img)


# In[14]:


#input:total_summary_stats
all_inter_event_dist_time = pd.DataFrame()
all_folders_name_short = np.unique(total_summary_stats.folder_name)#[Diane 0522]
for folder_name_short in all_folders_name_short:#folder_idx = 0
    ss = total_summary_stats[total_summary_stats.folder_name == folder_name_short]
    #print(ss)#8 cols: cc_width,cc_height, cc_area, cc_centroid_row, cc_centroid_col, time_since_start(mins), cc_cen_col_dist_mean, folder_name
    
    inter_event_col_dist = abs(ss['cc_centroid_col'].diff())[1:]#the first row (index 0) would be NaN -> drop it. 
    #Now index is starting with 1. if row index = i, it means it's the distance btw (i+1)-th embolism and i-th embolism event (i starts with 1)
    inter_event_col_dist = inter_event_col_dist.rename('inter_event_col_dist')#a series, not pd --> a different rename method
    
    inter_event_time= abs(ss['time_since_start(mins)'].diff())[1:]
    inter_event_time = inter_event_time.rename('inter_event_time(min)')
    
    inter_event_dist_time = inter_event_col_dist.to_frame().join(inter_event_time.to_frame())
    inter_event_dist_time
    inter_event_dist_time['folder_name'] = folder_name_short#add a col for foldername, s.t. when append tp all_inter_evennt_dist_time, we can know inter_event_dist_time is for which folder 
    
    #get species name and relative index
    species_and_num =  folder_name_short.split("_")#ex: ['dacexc5.2', 'stem']
    species_name = species_and_num[0][0:6]#ex: dacexc
    inter_event_dist_time['species'] = species_name
    if all_inter_event_dist_time.size ==0 :#first folder
        rel_exp_idx = 1
    else:
        if prev_species != species_name:#not the same species as the ones that have been added to all_inter_event_dist_time
            rel_exp_idx = 1
        else:
            rel_exp_idx += 1
    prev_species = species_name#for next round
    inter_event_dist_time['rel_exp_idx'] = rel_exp_idx#realtive experiment index, used for plotting
    #fig = px.scatter(inter_event_dist_time, x="inter_event_col_dist", y="inter_event_time(min)", trendline="ols")
    #fig.update_layout(
    #    height=300,
    #    width = 300,
    #    title_text=folder_name_short
    #)

    #fig.show()
    
    all_inter_event_dist_time = all_inter_event_dist_time.append(inter_event_dist_time)
    
#return all_inter_event_dist_time





# # Save into csv files

# In[23]:


all_inter_event_dist_time.to_csv(os.path.join(input_dir,'all_inter_event_dist_time.csv'),index=True)


# In[24]:


total_summary_stats.to_csv(os.path.join(input_dir,'total_summary_stats.csv'),index=True)


fx.print_used_time(start_time)
