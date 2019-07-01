#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:01:49 2019

@author: j-bd

Before launching this algorithm make sure your data are organized as following:
    A directory with your images test named 'stage_2_test_images'
    A directory with your images train named 'stage_2_train_images'
    A detailled CSV file train labels named 'stage_2_train_labels.csv'
    A detailled CSV file for submission named 'stage_2_sample_submission.csv'
All this elements must be gather in the same directory. The path will be mention
in the following variable 'IMAGE_DIR'

Befor launching this algorithm clone darknet (yolov3 package) with following instructions on
this website: https://pjreddie.com/darknet/install/

To train yolo_v3 algorithm to detect our custom objects we need to follow this steps:
    Create a file named 'yolo-obj.cfg' as a configuration of the CNN (a custom copy of yolov3.cfg)
    Create a file named 'obj.names' with the names of our custom object. Here is 'pneumonia'
    Create a file named 'obj.data' with the numbers of class objects and path to differents files
    Gather all images ('jpg' format) in a same directory
    Create a '.txt' file with all labels in relative float value
    Create a '.txt' file with the path to all training images
    Create a '.txt' file with the path to all validation images
    Create a '.txt' file with the path to all test images
Source: https://github.com/AlexeyAB/darknet
"""

import os

import numpy as np
import pandas as pd
import pydicom
import cv2


IMAGE_DIR = "/home/latitude/Documents/Kaggle/rsna-pneumonia/data/"
INPUT_TRAIN_DATA_DIR = IMAGE_DIR + "stage_2_train_images"
INPUT_TEST_DATA_DIR = IMAGE_DIR + "stage_2_test_images"
PROJECT_DIR = "/home/latitude/Documents/Kaggle/rsna-pneumonia/yolo_v3/"
OUTPUT_TRAIN_DATA_DIR = PROJECT_DIR + "train_data"
OUTPUT_TEST_DATA_DIR = PROJECT_DIR + "test_data"
OUTPUT_METADATA_DIR = PROJECT_DIR + "metadata/"
FILE_TRAIN = "stage_2_train_labels.csv"
FILE_TEST = "stage_2_sample_submission.csv"
TRAIN_CSV = OUTPUT_TRAIN_DATA_DIR + "/train_pneumonia.csv"
VAL_CSV = OUTPUT_TRAIN_DATA_DIR + "/val_pneumonia.csv"
IMAGE_SIZE = 1024



def yolo_cfg_file(batch, subd, class_nbr):
    '''Generate the config file for yolo v3 training
    We copy an existing file 'darknet/cfg/yolov3.cfg'
    then we customize it regarding the context and save it under 'yolo-obj.cfg' '''
    input_cfg = PROJECT_DIR + "darknet/cfg/yolov3.cfg"
    with open(input_cfg, 'r') as cfg_in:
        metadata_file = cfg_in.read()

    max_batches = str(2000 * class_nbr)
    steps = str(max_batches * 0.8) + ',' + str(max_batches * 0.9)
    filter_yolo = str((class_nbr + 5) * 3)
    metadata_file = metadata_file.replace('batch=64', 'batch=' + str(batch))
    metadata_file = metadata_file.replace('subdivisions=16', 'subdivisions=' + str(subd))
    metadata_file = metadata_file.replace('max_batches = 500200', 'max_batches =' + max_batches)
    metadata_file = metadata_file.replace('steps=400000,450000', 'steps=' + steps)
    metadata_file = metadata_file.replace('classes=80', 'classes=' + str(class_nbr))
    metadata_file = metadata_file.replace('filters=255', 'filters=' + filter_yolo)

    output_cfg = OUTPUT_METADATA_DIR + "yolo-obj.cfg"
    with open(output_cfg, 'w') as cfg_out:
        cfg_out.write(metadata_file)


def dcm_to_array(image_path):
    '''Tranform dicom image format to a numpy array'''
    dcm_im = pydicom.read_file(image_path + ".dcm").pixel_array
    return np.stack([dcm_im]*3, -1)


def sub_selection(dataset, origin_folder, target_folder):
    '''Copy the choosen images in the right directory under jpg format'''
    for image_name in dataset.iloc[:, 0].unique():
        image = dcm_to_array(origin_folder + "/" + image_name)
        cv2.imwrite(target_folder + "/" + image_name + ".jpg", image)


def label_generation(dataset, target_folder):
    '''Generate label in the shape of yolo_v3 learning CNN:
    <object-class> <x_center> <y_center> <width> <height>
    relative value is required by yolo_v3 algorithm
    one txt file per image is required by yolo_v3 algorithm'''
    for name, groupe in dataset.groupby("patientId"):
        label_file = target_folder + "/" + name + ".txt"
        with open(label_file, "w+") as file:
            for x, y, w, h in groupe.iloc[:, 1:].values:
                rel_w = w / IMAGE_SIZE
                rel_h = h / IMAGE_SIZE
                rel_x_center = x / IMAGE_SIZE + rel_w / 2
                rel_y_center = y / IMAGE_SIZE + rel_h / 2
                line = "{} {} {} {} {}\n".format(0, rel_x_center, rel_y_center, rel_w, rel_h)
                file.write(line)


# =============================================================================
# Loading target data from training directory
# =============================================================================
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

os.makedirs(OUTPUT_TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_METADATA_DIR, exist_ok=True)

train.to_csv(TRAIN_CSV, index=False)
val.to_csv(VAL_CSV, index=False)
sub_selection(train, INPUT_TRAIN_DATA_DIR, OUTPUT_TRAIN_DATA_DIR)
sub_selection(val, INPUT_TRAIN_DATA_DIR, OUTPUT_TRAIN_DATA_DIR)
label_generation(train, OUTPUT_TRAIN_DATA_DIR)
label_generation(val, OUTPUT_TRAIN_DATA_DIR)

#test = dataset_test.iloc[:20]
#test.to_csv(PROJECT_DIR + "test/test_ship.csv", index=False)
#sub_selection(test, "train_v2/", OUTPUT_TEST_DATA_DIR)
