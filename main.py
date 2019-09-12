#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:33:29 2019

@author: j-bd
"""
import os
import logging

import pandas as pd

import pneumonia_functions
import pneumonia_detection
import constants


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def pre_trainning(args):
    '''Lauch all necessary steps to set up Yolo v3 algorithm before for objects
    trainning'''
    image_dir = args.origin_folder
    input_train_data_dir = os.path.join(image_dir, "stage_2_train_images")
    project_dir = args.project_folder
    train_data_dir = os.path.join(project_dir, "data")
    train_images_dir = os.path.join(project_dir, "data/obj")
    backup = os.path.join(project_dir, "backup_log")
    file_train = os.path.join(image_dir, "stage_2_train_labels.csv")
    yolo_label = os.path.join(project_dir, "darknet/data/labels")
    test_images_dir = os.path.join(project_dir, "detect_results/obj")

    pneumonia_functions.structure(
        [train_images_dir, test_images_dir, backup], train_data_dir, yolo_label,
        project_dir
    )

    pneumonia_functions.yolo_parameters(
        project_dir, train_data_dir, backup, args.batch, args.subdivisions,
        constants.OBJ_NBR, constants.OBJ_NAME
    )

    df = pd.read_csv(file_train)

    pneumonia_functions.yolo_jpg_file(
        df, input_train_data_dir, train_images_dir
    )
    pneumonia_functions.yolo_label_generation(
        df, train_images_dir, constants.IMAGE_SIZE
    )

    train_df, val_df = pneumonia_functions.data_selection(df, args.split_rate)
    pneumonia_functions.yolo_image_path_file(
        train_df, train_data_dir, train_images_dir, "train.txt"
    )
    pneumonia_functions.yolo_image_path_file(
        val_df, train_data_dir, train_images_dir, "val.txt"
    )

    logging.info(
        '''To lauch the training, please enter the following command
        in your terminal :\n
        ./darknet/darknet detector train data/obj.data data/yolo-obj.cfg darknet53.conv.74\
        -i 0 | tee train_log.txt\n
        Be sure to be in your Master Directory: {}'''.format(project_dir)
    )


def detection(args):
    '''Lauch all necessary steps to set up Yolo v3 algorithm before objects
    detection'''
    image_dir = args.origin_folder
    input_test_data_dir = os.path.join(args.origin_folder, "stage_2_test_images")
    project_dir = args.project_folder
    test_data_dir = os.path.join(project_dir, "detect_results")
    test_images_dir = os.path.join(project_dir, "detect_results/obj")
    file_test = os.path.join(image_dir, "stage_2_sample_submission.csv")

    test_dataset = pd.read_csv(file_test)
    pneumonia_functions.yolo_jpg_file(
        test_dataset, input_test_data_dir, test_images_dir
    )
    cfg_file = pneumonia_detection.test_cfg_file(
        project_dir, test_data_dir, constants.BATCH, constants.SUB,
        constants.OBJ_DETEC, args.detect_im_size
    )
    images_to_detect = list()
    for image_name in test_dataset.iloc[:, 0].unique():
        images_to_detect.append(os.path.join(test_images_dir, image_name + ".jpg"))

    pneumonia_detection.image_detection(
        cfg_file, images_to_detect, os.path.join(test_data_dir, "submission.csv"),
        args
    )


def main():
    '''Allow the selection between algorithm training or image detection'''
    args = pneumonia_functions.create_parser()
    pneumonia_functions.check_inputs(args)

    if args.command == "train":
        pre_trainning(args)
    else:
        detection(args)


if __name__ == "__main__":
    main()
