#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:01:49 2019

@author: j-bd

Before launching this algorithm make sure the Kaggle data are organized as following in a master
directory:
    A directory with your images test named 'stage_2_test_images'
    A directory with your images train named 'stage_2_train_images'
    A detailled CSV file train labels named 'stage_2_train_labels.csv'
    A detailled CSV file for submission named 'stage_2_sample_submission.csv'
All this elements must be gather in the same directory. The path will be mention in the following
variable 'IMAGE_DIR'.
Source: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

Before launching this algorithm, you need to clone darknet (yolov3 package) in your project
directory (variable: PROJECT_DIR). To do it, please follow the instructions on this website:
https://pjreddie.com/darknet/install/

To train yolo_v3 algorithm to detect our custom objects we need to follow this steps:
    Create a file named 'yolo-obj.cfg' as a configuration of the CNN (a custom copy of yolov3.cfg)
    Create a file named 'obj.names' with the names of our custom object. Here is 'pneumonia'
    Create a file named 'obj.data' with the numbers of class objects and path to differents files
    Gather all images ('.jpg' format) in a same directory
    Create a '.txt' file with all labels in relative float value
    Create a '.txt' file with the path to all training images
    Create a '.txt' file with the path to all validation images
    Download a pre-trained weights file
Source: https://github.com/AlexeyAB/darknet

To save model files more regularly during the training, we need to modify the code of this following
file "darknet/examples/detector.c" (around the line 138) :
    if(i%10000==0 || (i < 1000 && i%100 == 0)){ #old
    to:
    if(i%1000==0 || (i < 2000 && i%50 == 0)){ #new
    we now save in the backup folder every 50 iterations a '.weights' file till we reach 2000 and
    then we save after every 1000 iterations

@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
"""

import os
from distutils.dir_util import copy_tree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import cv2
import wget


IMAGE_DIR = "/home/latitude/Documents/Kaggle/rsna-pneumonia/data/"
INPUT_TRAIN_DATA_DIR = IMAGE_DIR + "stage_2_train_images/"
INPUT_TEST_DATA_DIR = IMAGE_DIR + "stage_2_test_images/"
PROJECT_DIR = "/home/latitude/Documents/Kaggle/rsna-pneumonia/yolo_v3/"
TRAIN_DATA_DIR = PROJECT_DIR + "data/"
TRAIN_IMAGES_DIR = TRAIN_DATA_DIR + "obj/"
TEST_DATA_DIR = PROJECT_DIR + "test_data/"
BACKUP = PROJECT_DIR + "backup_log/"
FILE_TRAIN = "stage_2_train_labels.csv"
FILE_TEST = "stage_2_sample_submission.csv"
TRAIN_CSV = TRAIN_IMAGES_DIR + "train_pneumonia.csv"
VAL_CSV = TRAIN_IMAGES_DIR + "val_pneumonia.csv"
IMAGE_SIZE = 1024
OBJ_NBR = 1
YOLO_LABEL = PROJECT_DIR + "darknet/data/labels/"



def structure():
    '''Create the structure for the project and downoald necessary file'''
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(BACKUP, exist_ok=True)

    copy_tree(YOLO_LABEL, TRAIN_DATA_DIR + "labels/")

    print("Initial weight downloading is ongoing")
    yolo_pre_trained_weights("https://pjreddie.com/media/files/darknet53.conv.74")


def data_preprocessing(dataset, split_rate):
    '''Produce different datasets'''
    #Dataframe with only images of pneumonia
    positive = dataset[dataset.iloc[:, -1] == 1]
    positive = positive.reset_index(drop=True)
    pos_size = len(positive)

    #Dataframe with only images of non pneumonia. As required by the authors of Yolov3 we need to
    #have DataFrames of same size
    negative = dataset[dataset.iloc[:, -1] == 0]
    neg_size = min(len(positive.iloc[:, 0].unique()), len(negative))
    negative = negative.iloc[: neg_size, :]
    negative = negative.reset_index(drop=True)

    #Train and validation dataframe
    pos_split = int(pos_size * split_rate)
    neg_split = int(neg_size * split_rate)
    train = pd.concat([positive.iloc[: pos_split, :], negative.iloc[: neg_split, :]], axis=0)
    train = train.reset_index(drop=True)
    val = pd.concat([positive.iloc[pos_split:, :], negative.iloc[neg_split:, :]], axis=0)
    val = val.reset_index(drop=True)

    return train, val, positive, negative


def yolo_cfg_file(batch, subd, class_nbr):
    '''Generate the config file for yolo v3 training
    We copy an existing file 'darknet/cfg/yolov3.cfg' then we customize it
    regarding the context and save it under 'yolo-obj.cfg' in metadata directory'''
    input_cfg = PROJECT_DIR + "darknet/cfg/yolov3.cfg"
    with open(input_cfg, 'r') as cfg_in:
        new_cfg = cfg_in.read()

    max_batches = 2000 * class_nbr
    steps = str(max_batches * 0.8) + ',' + str(max_batches * 0.9)
    filter_yolo = str((class_nbr + 5) * 3)
    new_cfg = new_cfg.replace('batch=64', 'batch=' + str(batch))
    new_cfg = new_cfg.replace('subdivisions=16', 'subdivisions=' + str(subd))
    new_cfg = new_cfg.replace('max_batches = 500200', 'max_batches =' + str(max_batches))
    new_cfg = new_cfg.replace('steps=400000,450000', 'steps=' + steps)
    new_cfg = new_cfg.replace('classes=80', 'classes=' + str(class_nbr))
    new_cfg = new_cfg.replace('filters=255', 'filters=' + filter_yolo)

    output_cfg = TRAIN_DATA_DIR + "yolo-obj.cfg"
    with open(output_cfg, 'w') as cfg_out:
        cfg_out.write(new_cfg)


def yolo_names_file(list_names):
    '''Generate the file gathering all object class names for yolo v3 training
    We except a list of string and save it under 'obj.names' in metadata directory'''
    names_file = TRAIN_DATA_DIR + "obj.names"
    with open(names_file, 'w') as names:
        for obj_name in list_names:
            line = "{}\n".format(obj_name)
            names.write(line)


def yolo_data_file(class_nbr):
    '''Generate the file with paths for yolo v3 training
    The file will be save under 'obj.data' in metadata directory'''
    data_file = TRAIN_DATA_DIR + "obj.data"
    with open(data_file, 'w') as data:
        line = f"classes = {class_nbr}\n"\
        f"train = {TRAIN_DATA_DIR + 'train.txt'}\n"\
        f"valid = {TRAIN_DATA_DIR + 'val.txt'}\n"\
        f"names = {TRAIN_DATA_DIR + 'obj.names'}\n"\
        f"backup = {BACKUP}"
        data.write(line)


def dcm_to_array(image_path):
    '''Tranform dicom image format to a numpy array'''
    dcm_im = pydicom.read_file(image_path + ".dcm").pixel_array
    return np.stack([dcm_im]*3, -1)


def yolo_jpg_file(dataset, origin_folder, target_folder):
    '''Copy the choosen images in the right directory under jpg format'''
    for image_name in dataset.iloc[:, 0].unique():
        image = dcm_to_array(origin_folder + image_name)
        cv2.imwrite(target_folder + image_name + ".jpg", image)


def yolo_label_generation(dataset, target_folder):
    '''Generate label in the shape of yolo_v3 learning CNN:
    <object-class> <x_center> <y_center> <width> <height>
    relative value is required by yolo_v3 algorithm
    one txt file per image is required by yolo_v3 algorithm'''
    for name, groupe in dataset.groupby("patientId"):
        label_file = target_folder + "/" + name + ".txt"
        with open(label_file, "w+") as file:
            for x, y, w, h, cl in groupe.iloc[:, 1:].values:
                rel_w = w / IMAGE_SIZE
                rel_h = h / IMAGE_SIZE
                rel_x_center = x / IMAGE_SIZE + rel_w / 2
                rel_y_center = y / IMAGE_SIZE + rel_h / 2
                line = f"{int(cl - 1)} "\
                f"{rel_x_center} "\
                f"{rel_y_center} "\
                f"{rel_w} "\
                f"{rel_h}\n"
                file.write(line)


def yolo_image_path_file(dataset, target_folder, file_name):
    '''Generate a 'txt' file with the path and the name of each image'''
    txt_file = target_folder + file_name
    with open(txt_file, "w+") as file:
        for image_name in dataset.iloc[:, 0].unique():
            line = "{}\n".format(TRAIN_IMAGES_DIR + image_name + ".jpg")
            file.write(line)


def yolo_pre_trained_weights(link):
    '''Download the pre-trained weights darknet53.conv.74 (162.5MB)'''
    url = link
    print("Pre-trained weights 'darknet53.conv.74' downloading in progress (162.5MB). Please wait")
    wget.download(url, out=PROJECT_DIR)


def visualisation(dataset, index_patient):
    '''Display pneumonia or not image with or without the box'''
    if dataset.iloc[index_patient, -1]:
        patient_box = dataset[dataset.iloc[:, 0] == dataset.iloc[index_patient, 0]]
        for x, y, w, h in patient_box.iloc[:, 1:5].values:
            plt.plot([x, x, x+w, x+w, x], [y, y+h, y+h, y, y], label="pneumonia")
        plt.imshow(cv2.imread(TRAIN_IMAGES_DIR + dataset.iloc[index_patient, 0] + '.jpg'))
        plt.title("Pneumonia")
        plt.legend()
    else:
        plt.imshow(cv2.imread(TRAIN_IMAGES_DIR + dataset.iloc[index_patient, 0] + '.jpg'))
        plt.title("No pneumonia")


def loss_function(file_path):
    '''Represent the loss trend line over the learning process
    The 'train_log.txt' file is required to go through this analysis'''
    with open(file_path, "r") as log:
        iter_nbr = list()
        loss_value = list()
        for line in log:
            if "rate" in line:
                iter_nbr.append(int(line.split()[0].split(":")[0]))
                loss_value.append(float(line.split()[2]))
        plt.plot(iter_nbr, loss_value, label="loss function")
        plt.title("Evolution of the loss function during the training")
        plt.legend()
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss value")


# =============================================================================
# Loading data from training directory
# =============================================================================
structure()

original_dataset = pd.read_csv(IMAGE_DIR + FILE_TRAIN)
test_dataset = pd.read_csv(IMAGE_DIR + FILE_TEST)
train_df, val_df, pneumonia_df, non_pneumonia_df = data_preprocessing(original_dataset, 0.8)


# =============================================================================
# Yolo v3 files and parameters preparation
# =============================================================================
yolo_cfg_file(64, 16, OBJ_NBR)
yolo_names_file(["pneumonia"])
yolo_data_file(OBJ_NBR)

yolo_jpg_file(original_dataset, INPUT_TRAIN_DATA_DIR, TRAIN_IMAGES_DIR)

yolo_label_generation(pneumonia_df, TRAIN_IMAGES_DIR)

yolo_image_path_file(train_df, TRAIN_DATA_DIR, "train.txt")
yolo_image_path_file(val_df, TRAIN_DATA_DIR, "val.txt")

print('''To lauch the training, please enter the following command in your terminal :\n
./darknet/darknet detector train train_data/obj.data train_data/yolo-obj.cfg darknet53.conv.74\
-i 0 | tee train_log.txt\n
Be sure to be in your Master Directory: {}'''.format(PROJECT_DIR))


# =============================================================================
# Launching test
# =============================================================================
yolo_jpg_file(test_dataset, INPUT_TEST_DATA_DIR, TEST_DATA_DIR)
#./darknet/darknet detector test train_data/obj.data train_data/yolo-obj.cfg yolo-obj_350.weights
LAST_DIR = "/home/latitude/Documents/Kaggle/rsna-pneumonia/yolo_v3/last/"
df_last = test_dataset.iloc[:30, :]
yolo_jpg_file(df_last, INPUT_TEST_DATA_DIR, LAST_DIR)
yolo_image_path_file(df_last, TRAIN_DATA_DIR, "test.txt")
