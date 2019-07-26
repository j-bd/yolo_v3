#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:01:49 2019

@author: j-bd
"""
import os
from distutils.dir_util import copy_tree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import cv2
import wget


def yolo_cfg_file(project_dir, train_data_dir, batch, subd, class_nbr):
    '''Generate the config file for yolo v3 training
    We copy an existing file 'darknet/cfg/yolov3.cfg' then we customize it
    regarding the context and save it under 'yolo-obj.cfg' in data directory'''
    input_cfg = project_dir + "darknet/cfg/yolov3.cfg"
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

    output_cfg = train_data_dir + "yolo-obj.cfg"
    with open(output_cfg, 'w') as cfg_out:
        cfg_out.write(new_cfg)


def yolo_names_file(train_data_dir, list_names):
    '''Generate the file gathering all object class names for yolo v3 training
    We except a list of string and save it under 'obj.names' in data directory'''
    names_file = train_data_dir + "obj.names"
    with open(names_file, 'w') as names:
        for obj_name in list_names:
            line = "{}\n".format(obj_name)
            names.write(line)


def yolo_data_file(train_data_dir, backup, class_nbr):
    '''Generate the file with paths for yolo v3 training
    The file will be save under 'obj.data' in data directory'''
    data_file = train_data_dir + "obj.data"
    with open(data_file, 'w') as data:
        line = f"classes = {class_nbr}\n"\
        f"train = {train_data_dir + 'train.txt'}\n"\
        f"valid = {train_data_dir + 'val.txt'}\n"\
        f"names = {train_data_dir + 'obj.names'}\n"\
        f"backup = {backup}"
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


def yolo_label_generation(dataset, target_folder, image_size):
    '''Generate label in the shape of yolo_v3 learning CNN:
    <object-class> <x_center> <y_center> <width> <height>
    relative value is required by yolo_v3 algorithm.
    one txt file per image is also required by yolo_v3 algorithm
    As mentionned in darknet repo, to improve results we need to add images without objects
    All created files will be saved in the same directory which contains jpg files'''
    for name, groupe in dataset.groupby("patientId"):
        label_file = target_folder + "/" + name + ".txt"
        with open(label_file, "w+") as file:
            for x, y, w, h, cl in groupe.iloc[:, 1:].values:
                if cl:
                    rel_w = w / image_size
                    rel_h = h / image_size
                    rel_x_center = x / image_size + rel_w / 2
                    rel_y_center = y / image_size + rel_h / 2
                    line = f"{int(cl - 1)} "\
                    f"{rel_x_center} "\
                    f"{rel_y_center} "\
                    f"{rel_w} "\
                    f"{rel_h}\n"
                    file.write(line)


def yolo_image_path_file(dataset, target_folder, train_images_dir, file_name):
    '''Generate a 'txt' file with the path and the name of each image'''
    txt_file = target_folder + file_name
    with open(txt_file, "w+") as file:
        for image_name in dataset.iloc[:, 0].unique():
            line = "{}\n".format(train_images_dir + image_name + ".jpg")
            file.write(line)


def yolo_pre_trained_weights(link, path):
    '''Download the pre-trained weights darknet53.conv.74 (162.5MB)'''
    url = link
    print("[INFO] Pre-trained weights 'darknet53.conv.74' downloading in progress (162.5MB)."\
          "Please wait")
    wget.download(url, out=path)


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


def yolo_parameters(project_dir, train_data_dir, backup, batch, subdivisions, obj, list_obj):
    '''Create all files wich will be used by Yolo v3 algorithm during the learning process'''
    yolo_cfg_file(project_dir, train_data_dir, batch, subdivisions, obj)
    yolo_names_file(train_data_dir, list_obj)
    yolo_data_file(train_data_dir, backup, obj)


def structure(train_data_dir, train_images_dir, test_images_dir, backup, yolo_label, proj_dir):
    '''Create the structure for the project and downoald necessary file'''
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(backup, exist_ok=True)

    copy_tree(yolo_label, train_data_dir + "labels/")

    print(f"[INFO] Please, clone yolov3 package in '{proj_dir}' if it's not already done.")

    yolo_pre_trained_weights("https://pjreddie.com/media/files/darknet53.conv.74", proj_dir)


def visualisation(image_dir_path, dataset, index_patient):
    '''Display pneumonia or not image with or without the box'''
    if dataset.iloc[index_patient, -1]:
        patient_box = dataset[dataset.iloc[:, 0] == dataset.iloc[index_patient, 0]]
        for x, y, w, h in patient_box.iloc[:, 1:5].values:
            plt.plot([x, x, x+w, x+w, x], [y, y+h, y+h, y, y], label="pneumonia")
        plt.imshow(cv2.imread(image_dir_path + dataset.iloc[index_patient, 0] + '.jpg'))
        plt.title("Pneumonia")
        plt.legend()
    else:
        plt.imshow(cv2.imread(image_dir_path + dataset.iloc[index_patient, 0] + '.jpg'))
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
