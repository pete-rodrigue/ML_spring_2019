# README

This folder contains files for exercise five.&nbsp;&nbsp;

  
The main results and analysis are in run_pipeline_for_donors_choose_data.ipynb.&nbsp;&nbsp;  


The pipeline (with functions usable for applying machine learning algorithms to data other than the donors choose dataset) is availible in pipeline.py.&nbsp;&nbsp;  


The raw donors choose data is in the folder "raw data."&nbsp;&nbsp;  


The figures created by the run_pipeline_for_donors_choose_data.ipynb file are stored in the "figures" folder. You can find histograms and other distribution plots for the continuous variables in the donors choose dataset there. There's also a summary.csv file that has summary statistics for the dataset variables. Finally, there are several graphs plotting the false positive rate against the true positive rate for the pipeline ML models for different temporal splits of the donors choose data. The first split is found in fpr_vs_tprsplit_1_max_depth_10.png (with max tree depth of 10), the second split graph is found in fpr_vs_tprsplit_2.png, and the curve for the third temporal split is in fpr_vs_tprsplit_3.png.&nbsp;&nbsp;


Finally, there are several csv files in the main exercise five folder:&nbsp;&nbsp;  


* split_1.csv
* split_1_max_depth_10.csv
* split_1_max_depth_50.csv
* split_2.csv
* split_3.csv&nbsp;&nbsp;  


These each contain detailed performance metrics for each of our models, using the different temporal splits, at different threshold values.&nbsp;&nbsp;  
You can also find all the confusion matrices for every permutation of model, split, and threshold in the output of the cells in run_pipeline_for_donors_choose_data.ipynb.
