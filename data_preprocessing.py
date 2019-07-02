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
All this elements must be gather in the same directory. The path will be mention
in the following variable 'IMAGE_DIR'.

Before launching this algorithm, you need to clone darknet (yolov3 package) with following
instructions on this website: https://pjreddie.com/darknet/install/

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
"""

import os

import numpy as np
import pandas as pd
import pydicom
import cv2
import wget


IMAGE_DIR = "/home/latitude/Documents/Kaggle/rsna-pneumonia/data/"
INPUT_TRAIN_DATA_DIR = IMAGE_DIR + "stage_2_train_images/"
INPUT_TEST_DATA_DIR = IMAGE_DIR + "stage_2_test_images/"
PROJECT_DIR = "/home/latitude/Documents/Kaggle/rsna-pneumonia/yolo_v3/"
TRAIN_DATA_DIR = PROJECT_DIR + "train_data/"
TRAIN_IMAGES_DIR = TRAIN_DATA_DIR + "obj/"
TEST_DATA_DIR = PROJECT_DIR + "test_data/"
BACKUP = PROJECT_DIR + "backup_log/"
FILE_TRAIN = "stage_2_train_labels.csv"
FILE_TEST = "stage_2_sample_submission.csv"
TRAIN_CSV = TRAIN_IMAGES_DIR + "train_pneumonia.csv"
VAL_CSV = TRAIN_IMAGES_DIR + "val_pneumonia.csv"
IMAGE_SIZE = 1024



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
		line = """classes = {}\n
		train = {}\n
		valid = {}\n
		names = {}\n
		backup = {}
		""".format(class_nbr,
				   TRAIN_DATA_DIR + "train.txt",
				   TRAIN_DATA_DIR + "val.txt",
				   TRAIN_DATA_DIR + "obj.names",
				   BACKUP)
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
			for x, y, w, h in groupe.iloc[:, 1:].values:
				rel_w = w / IMAGE_SIZE
				rel_h = h / IMAGE_SIZE
				rel_x_center = x / IMAGE_SIZE + rel_w / 2
				rel_y_center = y / IMAGE_SIZE + rel_h / 2
				line = "{} {} {} {} {}\n".format(0, rel_x_center, rel_y_center, rel_w, rel_h)
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
	wget.download(url, out=PROJECT_DIR)


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

os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(TEST_DATA_DIR, exist_ok=True)
os.makedirs(BACKUP, exist_ok=True)

yolo_cfg_file(64, 16, 1)
yolo_names_file(["pneumonia"])
yolo_data_file(1)

train = dataset_train.iloc[:200, :5]
val = dataset_train.iloc[200:220, :5]


train.to_csv(TRAIN_CSV, index=False)
val.to_csv(VAL_CSV, index=False)

yolo_jpg_file(train, INPUT_TRAIN_DATA_DIR, TRAIN_IMAGES_DIR)
yolo_jpg_file(val, INPUT_TRAIN_DATA_DIR, TRAIN_IMAGES_DIR)
yolo_label_generation(train, TRAIN_IMAGES_DIR)
yolo_label_generation(val, TRAIN_IMAGES_DIR)
yolo_image_path_file(train, TRAIN_DATA_DIR, "train.txt")
yolo_image_path_file(val, TRAIN_DATA_DIR, "val.txt")
yolo_pre_trained_weights("https://pjreddie.com/media/files/darknet53.conv.74")

#test = dataset_test.iloc[:20]
#test.to_csv(PROJECT_DIR + "test/test_ship.csv", index=False)
#yolo_jpg_file(test, "train_v2/", TEST_DATA_DIR)
