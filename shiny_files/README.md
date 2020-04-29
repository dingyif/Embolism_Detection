All shiny app used files

get_data --> points_boundary --> Shiny Plot
get_data --> visualize_cc_info

get_data.py: outputs pickle file ('shinydata.pkl') and 'emb_points_time.csv'. The 'emb_points_time.csv' has the following columns: 'row', 'col', 'number_emb', 'cc_num_emb', 'cc_width', 'cc_height', 'cc_area', 'cc_centroid_row', 'cc_centroid_col', 'embolism_time', 'time_since_start(mins)'

points_boundary2.ipynb: is the same as points_boundary.ipynb before merging get_data to it, except that it replace the column name "Z" in 'emb_points_time.csv' by 'number_emb'. The output file 'polygon_vertices_df.csv' still only has 3 columns: 'col', 'row', 'Z'.
