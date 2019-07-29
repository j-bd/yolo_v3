#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:33:29 2019

@author: j-bd

Before launching this algorithm make sure the Kaggle data are organized as following in a master
directory:
    A directory with your images test named 'stage_2_test_images'
    A directory with your images train named 'stage_2_train_images'
    A detailled CSV file train labels named 'stage_2_train_labels.csv'
    A detailled CSV file for submission named 'stage_2_sample_submission.csv'
All this elements must be gather in the same directory. The path will be mention when launching the
algorithm.
Source: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

Before launching this algorithm, you need to clone darknet (yolov3 package) in your project
directory (to be created by yourself and mentionned when launching the algorithm). To do it, please
follow the instructions on this website: https://pjreddie.com/darknet/install/
Check options available before enter 'make' command as GPU and so on.

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

import argparse

import pandas as pd

import pneumonia_functions
import pneumonia_detection


def create_parser():
    '''Get the informations from the operator'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocessing",
                        help="command to prepare data in order to lauch yolo training")
    parser.add_argument("-of", "--origin_folder", required=True,
                        help="path to the Kaggle folder containing all data")
    parser.add_argument("-pf", "--project_folder", required=True,
                        help="path to your project folder")
    parser.add_argument("-b", "--batch", type=int, default=64,
                        help="batch number for yolo config file used during yolo training")
    parser.add_argument("-s", "--subdivisions", type=int, default=16,
                        help="subdivisions number for yolo config file used during yolo training")
    parser.add_argument("-sr", "--split_rate", type=float, default=0.8,
                        help="split rate between train and validation dataset during yolo training")
    parser.add_argument("-d", "--detection",
                        help="command to detect pneumonia object on image")
    parser.add_argument("-w", "--weights_path",
                        help="Path to the weights file used by Yolo algorith to detect object")
    parser.add_argument("-c", "--confidence", type=float, default=0.7,
                        help="minimum probability to filter weak detections")
    parser.add_argument("-t", "--threshold", type=float, default=0.025,
                        help="threshold when applying non-maxima suppression")
    parser.add_argument("-dis", "--detect_im_size", type=int, default=640,
                        help="resize input image to improve the detection"\
                        "(must be a multiple of 32)")
    args = parser.parse_args()
    return args


def check_inputs(args):
    '''Check if inputs are right'''
    if args.preprocessing and args.detection is not None:
        raise ValueError("Choose between preprocessing and detection but not both")
    if args.preprocessing:
        if args.origin_folder is None:
            raise ValueError("Missing path to the origin folder (Kaggle data)")
        if args.project_folder is None:
            raise ValueError("Missing path to the your project folder")
        if not 0.7 <= args.split_rate <= 0.95:
            raise ValueError(f"Split rate must be between 0,7 and 0.95,"\
                             "currently {args.split_rate}")
    if args.detection:
        if args.weights_path is None:
            raise ValueError("Missing path to Yolo weights file used for detection")
        if not args.detect_im_size % 32 == 0:
            raise ValueError("Detection image size must be a multiple of 32")


def pre_trainning(args):
    '''Lauch all necessary steps to set up Yolo v3 algorithm before for objects trainning'''
    IMAGE_DIR = args.origin_folder
    INPUT_TRAIN_DATA_DIR = IMAGE_DIR + "stage_2_train_images/"
    PROJECT_DIR = args.project_folder
    TRAIN_DATA_DIR = PROJECT_DIR + "data/"
    TRAIN_IMAGES_DIR = PROJECT_DIR + "data/obj/"
    BACKUP = PROJECT_DIR + "backup_log/"
    FILE_TRAIN = "stage_2_train_labels.csv"
    IMAGE_SIZE = 1024
    OBJ_NBR = 1
    YOLO_LABEL = PROJECT_DIR + "darknet/data/labels/"
    TEST_IMAGES_DIR = PROJECT_DIR + "detect_results/obj/"

    pneumonia_functions.structure(TRAIN_DATA_DIR,
                                  TRAIN_IMAGES_DIR,
                                  TEST_IMAGES_DIR,
                                  BACKUP,
                                  YOLO_LABEL,
                                  PROJECT_DIR)

    pneumonia_functions.yolo_parameters(PROJECT_DIR,
                                        TRAIN_DATA_DIR,
                                        BACKUP,
                                        args.batch,
                                        args.subdivisions,
                                        OBJ_NBR,
                                        ["pneumonia"])

    original_dataset = pd.read_csv(IMAGE_DIR + FILE_TRAIN)

    train_df, val_df, pneumonia_df, non_pneumonia_df = pneumonia_functions.data_preprocessing(
        original_dataset,
        args.split_rate)

    pneumonia_functions.yolo_jpg_file(original_dataset, INPUT_TRAIN_DATA_DIR, TRAIN_IMAGES_DIR)

    pneumonia_functions.yolo_label_generation(original_dataset, TRAIN_IMAGES_DIR, IMAGE_SIZE)

    pneumonia_functions.yolo_image_path_file(train_df,
                                             TRAIN_DATA_DIR,
                                             TRAIN_IMAGES_DIR,
                                             "train.txt")
    pneumonia_functions.yolo_image_path_file(val_df,
                                             TRAIN_DATA_DIR,
                                             TRAIN_IMAGES_DIR,
                                             "val.txt")

    print('''[INFO] To lauch the training, please enter the following command in your terminal :\n
    ./darknet/darknet detector train data/obj.data data/yolo-obj.cfg darknet53.conv.74\
    -i 0 | tee train_log.txt\n
    Be sure to be in your Master Directory: {}'''.format(PROJECT_DIR))


def pre_detection(args):
    '''Lauch all necessary steps to set up Yolo v3 algorithm before objects detection'''
    IMAGE_DIR = args.origin_folder + "/"
    INPUT_TEST_DATA_DIR = args.origin_folder + "/stage_2_test_images/"
    PROJECT_DIR = args.project_folder + "/"
    TEST_DATA_DIR = PROJECT_DIR + "detect_results/"
    TEST_IMAGES_DIR = PROJECT_DIR + "detect_results/obj/"
    FILE_TEST = "stage_2_sample_submission.csv"

    test_dataset = pd.read_csv(IMAGE_DIR + FILE_TEST)
    pneumonia_functions.yolo_jpg_file(test_dataset, INPUT_TEST_DATA_DIR, TEST_IMAGES_DIR)
    cfg_file = pneumonia_detection.test_cfg_file(PROJECT_DIR, TEST_DATA_DIR, args.detect_im_size)

    images_to_detect = list()
    for image_name in test_dataset.iloc[:, 0].unique():
        images_to_detect.append(TEST_IMAGES_DIR + image_name + ".jpg")

    final_result = list()
    final_result.append("patientId,PredictionString")

    return cfg_file, images_to_detect, final_result, TEST_DATA_DIR + "submission.csv"


def main():
    '''Allow the selection between algorithm training or image detection'''
    args = create_parser()
    check_inputs(args)

    if args.preprocessing:
        pre_trainning(args)

    if args.detection:
        cfg_path, images, result, output_path = pre_detection(args)
        for image in images:
            print("[INFO] ", images.index(image)+ 1, "/", len(images))
            box = pneumonia_detection.detect(image,
                                             args.weights_path,
                                             cfg_path,
                                             args.confidence,
                                             args.threshold)
            result.append(box)
        pneumonia_detection.submission_file(output_path, result)

        print("[INFO] All images have been proceed")


if __name__ == "__main__":
    main()
