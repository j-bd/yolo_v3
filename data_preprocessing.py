#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:01:49 2019

@author: j-bd
"""

import os
import pandas as pd
import shutil


# =============================================================================
# Loading target data from training directory
# =============================================================================

# Root directory of the project
IMAGE_DIR = "/home/latitude/Documents/Kaggle/rsna-pneumonia/data/"
PROJECT_DIR = "/home/latitude/Documents/Kaggle/rsna-pneumonia/yolo_v3/"
FILE_TRAIN = "stage_2_train_labels.csv"
FILE_TEST = "stage_2_sample_submission.csv"
TRAIN_DATA_DIR = PROJECT_DIR + "train_data"
TRAIN_CSV = TRAIN_DATA_DIR + "/train_pneumonia.csv"
VAL_CSV = TRAIN_DATA_DIR + "/val_pneumonia.csv"

dataset_train = pd.read_csv(IMAGE_DIR + FILE_TRAIN)
dataset_test = pd.read_csv(IMAGE_DIR + FILE_TEST)

#Keep only images with pneumonia
dataset_train = dataset_train.dropna()
dataset_train = dataset_train.reset_index(drop=True)


# =============================================================================
# Choosing the right amount of data in accordance with the computer power used.
# Save the choice under two files : 'train' and 'test'
# =============================================================================
train = dataset_train.iloc[:200, :5]
val = dataset_train.iloc[200:220, :5]

os.makedirs(TRAIN_DATA_DIR, exist_ok=True)

'''TO DO : Compute the box center'''

train.to_csv(TRAIN_CSV, index=False)
val.to_csv(VAL_CSV, index=False)

#test = dataset_test.iloc[:20]
#test.to_csv(PROJECT_DIR + "test/test_ship.csv", index=False)


# =============================================================================
# Load the choosen images in the right directory
# =============================================================================
def sub_selection(dataset, origin_folder, target_folder):
    '''Copy the choosen images in the right directory'''
    filelist = list()

    for image_name in dataset.iloc[:,0].unique():
        filelist.append(IMAGE_DIR + origin_folder + image_name)

    for file in filelist:
        shutil.copy2(file, target_folder)


sub_selection(train, IMAGE_DIR, TRAIN_DATA_DIR)
sub_selection(val, IMAGE_DIR, TRAIN_DATA_DIR)
#sub_selection(test, "train_v2/", "test/")


