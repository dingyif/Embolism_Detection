import math
import argparse
import os
import numpy as np
import pickle
import func_get_data as fx
import tifffile as tiff
import datetime
start_time = datetime.datetime.now()

parser  = argparse.ArgumentParser(description='Server version for get_data.py')
parser.add_argument('--disk_path', type = str, default="/rigel/stats/projects/emb_proj/Processed0618")
parser.add_argument('--output_root_folder', type = str, default="/rigel/stats/projects/emb_proj/output_get_data")
parser.add_argument('--folder_idx_start', type = int, help = 'Start folder index: 0')
parser.add_argument('--folder_idx_end', type = int, help = 'End folder index: 52')
parser.add_argument('--has_processed', type = bool, help = 'Whether the folder is processed or not')
args = parser.parse_args()

'''
User-specific input arguments from parser
'''
#in server we have total 53 folders, only 22 folders will contain stem, so the idx range should be 0-22
folder_idx_start = args.folder_idx_start
folder_idx_end = args.folder_idx_end
output_dir = args.output_root_folder
#directory where the folder located.
output_version_name = 'true_tif'
disk_path = args.disk_path
#arguments for preprocessed folder
has_processed = args.has_processed
#use the below arguments and save into different folder for outputs
use_predict_tif = False #(False):use true labels tif . (True or has_processed=False): use predict.tif 
use_txt = False #use true_positive,fp,fn txt files for img index
#default is to plot tp emb.
plot_fn = True #(TRUE):plot fn emb. from false_positive_index.txt 
plot_fp = False 
sl_tag = 'stem'#tag for stem or leaf

'''
Paths
'''
#disk_path = os.path.join(tif_top_dir,version_name)
all_species_folder = os.listdir(disk_path)
#species name[Alchornea_latifolia,...]
all_folders_name = sorted(os.listdir(disk_path))
# list of all ['/rigel/stats/projects/emb_proj/Processed0522/Alchornea_latifolia',...]
all_folder_dir = [os.path.join(disk_path,folder_name) for folder_name in all_folders_name]
# '/rigel/stats/projects/emb_proj/Processed0522/Alchornea_latifolia'
## used to store the information of the folder
all_subfolder_exp_dir = []
for folder_path in all_folder_dir:
# [ALCLAT7,ALCLAT8,...]
    sub_folders = os.listdir(folder_path)
    # ['/rigel/stats/projects/emb_proj/Processed0522/Alchornea_latifolia/ALCLAT7/',...]
    all_subfolder_dir = [os.path.join(folder_path,subfolder) for subfolder in sub_folders]
    #Go deep for each experiment
    inside_str = 'OpticalVulnerabilityCurve/PiImages/'
    #['/rigel/stats/projects/emb_proj/Processed0522/Alchornea_latifolia/ALCLAT7/OpticalVulnerabilityCurve/PiImages/',]
    all_subfolder_dir_deep = [os.path.join(sub_folder_dir,inside_str) for sub_folder_dir in all_subfolder_dir]
    all_folders_dir = [os.path.join(disk_path,folder) for folder in all_folders_name]
    #list the stem or leaf of each experiment
    # (['/rigel/stats/projects/emb_proj/Processed0522/Alchornea_latifolia/ALCLAT7/OpticalVulnerabilityCurve/PiImages/500_alclat7_leaf_done', 
    # '/rigel/stats/projects/emb_proj/Processed0522/Alchornea_latifolia/ALCLAT7/OpticalVulnerabilityCurve/PiImages/500_alclat7_stem_done_processed', 
    # '/rigel/stats/projects/emb_proj/Processed0522/Alchornea_latifolia/ALCLAT8/OpticalVulnerabilityCurve/PiImages/alclat8_leaf_done_processed', 
    # '/rigel/stats/projects/emb_proj/Processed0522/Alchornea_latifolia/ALCLAT8/OpticalVulnerabilityCurve/PiImages/alclat8_stem_done_processed'])
    all_subfolder_exp_dir.extend([os.path.join(sub_folder_dir_deep,stem_or_leaf) for sub_folder_dir_deep in all_subfolder_dir_deep for stem_or_leaf in os.listdir(sub_folder_dir_deep)])
#subset the only stem folder out
all_subfolder_stem_exp_dir = []
for subfolder_exp_dir in all_subfolder_exp_dir:
    #check whether stem inside the string or not
    if sl_tag in subfolder_exp_dir.lower():
        all_subfolder_stem_exp_dir.append(subfolder_exp_dir)



for folder_idx in range(folder_idx_start,folder_idx_end+1):
    # '/rigel/stats/projects/emb_proj/Processed0522/Alchornea_latifolia'
    dir_path = all_subfolder_stem_exp_dir[folder_idx]
    # folder_name: alclat7 or tabhet6 (good - orlando)
    folder_name = dir_path.split(os.sep)[-4].lower()
    folder_name_short = folder_name.replace(" ", "").split("(")[0] +'_'+sl_tag
    #remove spaces and split by left bracket to get 'tabhet6', then add '_stem'
    
    folder_name
    
    #used true label, imgs are acutally located in folder_path
    img_folder = dir_path

    input_tif_path = fx.get_input_tif_path_server(use_predict_tif,has_processed,dir_path,img_folder)
    print("tif name:",input_tif_path)
    '''
    Read Data from TIFF file
    '''
    input_tiff = tiff.imread(input_tif_path)#num_imgs x row_num x col_num
    #0: background. 255:embolism
    num_imgs = input_tiff.shape[0]
    row_num  = input_tiff.shape[1]
    col_num  = input_tiff.shape[2]
    '''
    get emb_img_idx_plot (the image index that have to be plotted) and emb_num_plot (the total number of images to be plotted)
    img_idx starts from 0
    '''
    if use_txt==True:
        emb_img_idx_plot,emb_num_plot = fx.get_img_idx_from_txt(dir_path=dir_path,plot_fn=plot_fn,plot_fp=plot_fp)
    else:#use true_tif to get img_idx with emb
        emb_img_idx_plot,emb_num_plot = fx.get_img_idx_from_tiff(input_tiff)

    print("num of imgs with emb that'll be plotted",emb_num_plot)
    
    '''
    Processing the tiff into matrix pixel value
    '''
    
    cc_min_area = 50#for better visualization (you can barely see cc with area < cc_min_area even if you plot them out)
    cc_min_area2 = 10#is used in case there's no cc with area > cc_min_area
    
    plot_mat_all = fx.get_pixel_pos_and_cc_from_imgs(input_tiff,emb_img_idx_plot,cc_min_area,cc_min_area2)
    #print(plot_mat_all.head())
    #print("finished plot_mat_all")
    '''
    Processing the time list
    Get the embolism time data. (i.e. image file names)
    '''
    #get a list of filenames under img_folder with filenames ending with ".png" or ".jpg"
    #also return the file extension
    all_files,img_extension = fx.get_img_filenames(img_folder)
    
    #get the photo taken time
    #convert '20190725-103910.png' to '2019-07-25 10:39:10'
    date_time_list_ori = fx.get_date_time_from_filenames(all_files,img_extension)
    
    #get time difference w.r.t to starting time(1st img)
    #and extract those information only for images with embolism (emb_num_plot,emb_img_idx_plot)
    date_time_list, diff_time_list_int, embolism_table = fx.get_time_wrt_start_time(date_time_list_ori,num_imgs,emb_num_plot,emb_img_idx_plot)
    #print(embolism_table.head())
    #[Diane] date_time_list would be 1 image later than that in Dingyi's fx.get_data.Rmd
    #embolism_table: emb_num_plot x 3 (3 cols: 'embolism_time','time_since_start(mins)','number_emb')
    
    #merge the two dataframes togther
    plot_mat_time = plot_mat_all.merge(embolism_table, on='number_emb', how='left')#merge by'number_emb'
    print(plot_mat_time.head())
    #11 cols:['row', 'col', 'number_emb', 'cc_num_emb', 'cc_width', 'cc_height',
    #       'cc_area', 'cc_centroid_row', 'cc_centroid_col', 'embolism_time',
    #       'time_since_start(mins)']
    
    '''
    save plot_mat_time into csv file ('emb_points_time.csv')
    '''
    #create output directory if not existed
    fx.create_or_empty_folder(output_dir, to_empty=False)
    fx.create_or_empty_folder(os.path.join(output_dir,output_version_name), to_empty=False)
    fx.create_or_empty_folder(os.path.join(output_dir,output_version_name,folder_name_short), to_empty=False)
    
    #output folder name based on user-specified arguments
    if use_predict_tif:
        out_tag1 = 'pred_tif'
    else:
        out_tag1= 'true_tif'
    
    if plot_fn:
        out_tag2 = 'has_fn'
    else:
        out_tag2 = 'no_fn'
    
    if plot_fp:
        out_tag3 = 'has_fp'
    else:
        out_tag3 = 'no_fp'
    
    output_folder_tag = '_'.join((out_tag1,out_tag2,out_tag3))
    output_folder = os.path.join(output_dir,output_version_name,folder_name_short,output_folder_tag)
    fx.create_or_empty_folder(output_folder, to_empty=False)
    
    #Transform into csv format
    plot_mat_time.to_csv(os.path.join(output_folder,'emb_points_time.csv'),index=True)
    
    '''
    data to feed image to plot
    create binary tiff that records whether an embolism has occured at the same pixel position across time
        (binary in the sense that each image in the tiff neglects the "frequency" of embolism on the same pixel position)
        This will be used in shiny app: "Image"
    '''
    plot_tiff = fx.get_sum_bin_tiff(input_tiff,row_num,col_num,emb_num_plot,emb_img_idx_plot)
    
    
    #from rpy2.robjects import pandas2ri
    #pandas2ri.activate()
    #emb_t_df = pandas2ri.py2ri(embolism_table)
    
    # Saving the outputs to pickle file for shiny app later
    with open(os.path.join(output_folder,'shinydata.pkl'), 'wb') as f:  
        pickle.dump({'num_imgs':num_imgs,'row_num':row_num,'col_num':col_num, 'date_time_list':date_time_list, 
                     'diff_time_list_int':diff_time_list_int, 'embolism_table':embolism_table,  
                     'plot_mat_all':plot_mat_all, 'plot_tiff':plot_tiff, 'plot_mat_time': plot_mat_time}, f)#'emb_table_df':emb_t_df,
    
    fx.print_used_time(start_time)