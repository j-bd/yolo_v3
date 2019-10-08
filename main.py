#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:33:29 2019

@author: j-bd
"""
import os

import pandas as pd

import pneumonia_functions
import pneumonia_detection
import constants


def training(dict_args):
    '''Lauch all necessary steps to set up Yolo v3 algorithm before objects
    trainning and lauch the training command'''
    pneumonia_functions.algorithm_structure_creation(dict_args)

    pneumonia_functions.yolo_params_files_creation(
        dict_args, constants.CHANNEL_NBR, constants.OBJ_NBR, constants.OBJ_NAME
    )

    train_df = pd.read_csv(dict_args["file_train"])

    pneumonia_functions.create_jpg_file(
        set(train_df.iloc[:, 0]), dict_args["input_train_data_dir"], dict_args["train_images_dir"]
    )
    pneumonia_functions.yolo_label_generation(
        train_df, dict_args["train_images_dir"], constants.IMAGE_SIZE
    )

    train_df, val_df = pneumonia_functions.data_selection(train_df, dict_args["split_rate"])
    pneumonia_functions.yolo_image_path_file(
        train_df, dict_args, "train.txt"
    )
    pneumonia_functions.yolo_image_path_file(
        val_df, dict_args, "val.txt"
    )

    os.system(
        "./darknet/darknet detector train data/obj.data data/yolo-obj.cfg darknet53.conv.74 -i 0 | tee train_log.txt"
    )


def detection(dict_args):
    '''Lauch all necessary steps to set up Yolo v3 algorithm before objects
    detection'''
    test_df = pd.read_csv(dict_args["file_test"])
    pneumonia_functions.create_jpg_file(
        set(test_df.iloc[:, 0]), dict_args["input_test_data_dir"], dict_args["test_images_dir"]
    )
    cfg_file = pneumonia_detection.test_cfg_file(
        dict_args, constants.BATCH, constants.SUB, constants.OBJ_DETEC
    )
    images_to_detect = list()
    for image_name in test_df.iloc[:20, 0].unique(): #suppr 20
        images_to_detect.append(
            os.path.join(dict_args["test_images_dir"], image_name + ".jpg")
        )

    pneumonia_detection.image_detection(
        cfg_file, images_to_detect, dict_args
    )


def main():
    '''Allow the selection between algorithm training or image detection'''
    args = pneumonia_functions.arguments_parser()
    pneumonia_functions.check_inputs(args)
    dict_args = pneumonia_functions.path_creator(args)

    if args.command == "train":
        training(dict_args)
    else:
        detection(dict_args)


if __name__ == "__main__":
    main()
