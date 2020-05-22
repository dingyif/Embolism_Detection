Files that'll help 
+ evaluate the performance for image classification/detection/segmentation.
  + check_stem_extraction.ipynb: plot out stem extraction for 1st and last image in each folder. Useful as a sanity check to see if stem extraction in the algo works well.
  + output_summary.ipynb: compare the confusion matrix, sensitivity, precision, accuracy for 2 versions.
  + output_summary_multiversion.ipynb: similar to output_summary, but can be used for *multiple* versions.
+ get basic information of the dataset
  + check_img_size.py: get the below information for each folder inside the dataset
  
    {img_folder_name}:{img_size}:{tif_size}:{is_large_img}:{tif_img_same_sz}
  
  + compare_folders.py: check if under 2 root directories, there are folders with the same leading folder names. Useful when we receive new data. 
