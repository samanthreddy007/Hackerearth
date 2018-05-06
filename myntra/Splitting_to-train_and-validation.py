# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 20:34:51 2018

@author: Karra's
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 22:27:03 2018

@author: Karra's
"""
import os
import shutil
import numpy as np

all_data_dir='D:/hackerearth/myntra_final/Train/'
training_data_dir='D:/hackerearth/myntra_final/Train_new/'
testing_data_dir='D:/hackerearth/myntra_final/Test_new/'

def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct):
    # Recreate testing and training directories
    
    num_training_files = 0
    num_testing_files = 0

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + category_name
        testing_data_category_dir = testing_data_dir + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < .3:
                shutil.copy(input_file, testing_data_dir  + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")
    
split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, .3)    