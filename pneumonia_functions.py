#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:01:49 2019

@author: j-bd
"""
import os
from distutils.dir_util import copy_tree
import argparse
import logging

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
import cv2
import wget

import constants


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def arguments_parser():
    '''Get the informations from the operator'''
    parser = argparse.ArgumentParser(
        prog='YOLOv3 new object training', usage='%(prog)s [Initially for RSNA pneumonia Kaggle challenge]',
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        To lauch the preprocessing execution:
        -------------------------------------
        python main.py --command train --origin_folder path/to/kaggle/data/folder
        --project_folder path/to/your/project/folder --batch 64 --subdivisions 16
        --split_rate 0.8

        The following arguments are mandatory: --command (preprocessing activation),
        --origin_folder (origin folder path) and --project_folder (project folder path)

        The following arguments are optionnals: --batch (batch number, default value 64),
        --subdivisions (subdivisions number, default value 16) and --split_rate
        (split rate wich is the percent of trainning and validation images. It must
        be between 0.7 and 0.95. The default value is 0.7)

        To lauch the detection execution:
        ---------------------------------
        python main.py --command detection --origin_folder path/to/kaggle/data/folder
        --project_folder path/to/your/project/folder --weights_path path/to/the/weight/file
        --confidence 0.7 --threshold 0.025 --detect_im_size 640

        The following arguments are mandatory: --command detection (detection activation),
        --origin_folder (origin folder path), --project_folder (project folder path) and
        --weights_path (weights file path)

        The following arguments are optionnals: --confidence (confidence, default value 0.7),
        --threshold (threshold, default value 0.025) and --detect_im_size
        (Size of images during detection. It must be a multiple of 32. The default
         value is 640)'''
    )
    parser.add_argument(
        "-cmd", "--command", required=True,
        help="choice between 'train' and 'detection'"
    )
    parser.add_argument(
        "-of", "--origin_folder", required=True,
        help="path to the Kaggle folder containing all data"
    )
    parser.add_argument(
        "-pf", "--project_folder", required=True,
        help="path to your project folder"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=64,
        help="batch number for yolo config file used during yolo training"
    )
    parser.add_argument(
        "-s", "--subdivisions", type=int, default=16,
        help="subdivisions number for yolo config file used during yolo training"
    )
    parser.add_argument(
        "-sr", "--split_rate", type=float, default=0.8,
        help="split rate between train and validation dataset during yolo training"
    )
    parser.add_argument(
        "-w", "--weights_path",
        help="Path to the weights file used by Yolo algorith to detect object"
    )
    parser.add_argument(
        "-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.2,
        help="threshold when applying non-maxima suppression"
    )
    parser.add_argument(
        "-dis", "--detect_im_size", type=int, default=640,
        help="resize input image to improve the detection(must be a multiple of 32)"
    )
    args = parser.parse_args()
    return args


def check_inputs(args):
    '''Check if inputs are right'''
    if args.command not in ["train", "detection"]:
        raise ValueError(
            "Your choice for '-c', '--command' must be 'train' or 'detection'."
        )
    if not os.path.isdir(args.origin_folder):
        raise FileNotFoundError(
            "Your choice for '-of', '--origin_folder' is not a valide directory."
        )
    if not os.path.isdir(args.project_folder):
        raise FileNotFoundError(
            "Your choice for '-pf', '--project_folder' is not a valide directory."
        )
    if not os.path.isdir(os.path.join(args.project_folder, "darknet")):
        raise FileNotFoundError(
            f"Please, clone darknet repository in {args.project_folder}."
        )
    if args.command == "train":
        if not 0.7 <= args.split_rate <= 0.95:
            raise ValueError(
                f"Split rate must be between 0,7 and 0.95, currently {args.split_rate}"
            )
    if args.command == "detection":
        if args.weights_path is None:
            raise ValueError("Missing path to Yolo weights file used for detection")
        if not args.detect_im_size % 32 == 0:
            raise ValueError("Detection image size must be a multiple of 32")


def path_creator(args):
    '''Create all necessary paths for the algorithm'''
    dict_args = vars(args)
    if args.command == "train":
        dict_args["image_dir"] = args.origin_folder
        dict_args["input_train_data_dir"] = os.path.join(
            dict_args["image_dir"], "stage_2_train_images"
        )
        dict_args["project_dir"] = args.project_folder
        dict_args["train_data_dir"] = os.path.join(
            dict_args["project_dir"], "data"
        )
        dict_args["train_images_dir"] = os.path.join(
            dict_args["project_dir"], "data/obj"
        )
        dict_args["backup"] = os.path.join(
            dict_args["project_dir"], "backup_log"
        )
        dict_args["file_train"] = os.path.join(
            dict_args["image_dir"], "stage_2_train_labels.csv"
        )
        dict_args["yolo_label"] = os.path.join(
            dict_args["project_dir"], "darknet/data/labels"
        )
        dict_args["test_images_dir"] = os.path.join(
            dict_args["project_dir"], "detect_results/obj"
        )
    else:
        dict_args["image_dir"] = args.origin_folder
        dict_args["input_test_data_dir"] = os.path.join(
            args.origin_folder, "stage_2_test_images"
        )
        dict_args["project_dir"] = args.project_folder
        dict_args["test_data_dir"] = os.path.join(
            dict_args["project_dir"], "detect_results"
        )
        dict_args["test_images_dir"] = os.path.join(
            dict_args["project_dir"], "detect_results/obj"
        )
        dict_args["file_test"] = os.path.join(
            dict_args["image_dir"], "stage_2_sample_submission.csv"
        )
    return dict_args


def cfg_file_creator(dict_args, channel_nbr, class_nbr):
    '''Generate the config file for yolo v3 training
    We copy an existing file 'darknet/cfg/yolov3.cfg' then we customize it
    regarding the context and save it under 'yolo-obj.cfg' in data directory'''
    input_cfg = os.path.join(dict_args["project_dir"], "darknet/cfg/yolov3.cfg")
    with open(input_cfg, 'r') as cfg_in:
        new_cfg = cfg_in.read()

    max_batches = 2000 * class_nbr
    steps = str(max_batches * 0.8) + ',' + str(max_batches * 0.9)
    filter_yolo = str((class_nbr + 5) * 3)
    new_cfg = new_cfg.replace('batch=64', 'batch=' + str(dict_args["batch"]))
    new_cfg = new_cfg.replace('subdivisions=16', 'subdivisions=' + str(dict_args["subdivisions"]))
    new_cfg = new_cfg.replace('channels=3', 'channels=' + str(channel_nbr))
    new_cfg = new_cfg.replace('max_batches = 500200', 'max_batches =' + str(max_batches))
    new_cfg = new_cfg.replace('steps=400000,450000', 'steps=' + steps)
    new_cfg = new_cfg.replace('classes=80', 'classes=' + str(class_nbr))
    new_cfg = new_cfg.replace('filters=255', 'filters=' + filter_yolo)

    output_cfg = os.path.join(dict_args["train_data_dir"], "yolo-obj.cfg")
    with open(output_cfg, 'w') as cfg_out:
        cfg_out.write(new_cfg)


def names_file_creator(train_data_dir, list_names):
    '''Generate the file gathering all object class names for yolo v3 training
    We except a list of string and save it under 'obj.names' in data directory'''
    names_file = os.path.join(train_data_dir, "obj.names")
    with open(names_file, 'w') as names:
        for obj_name in list_names:
            line = "{}\n".format(obj_name)
            names.write(line)


def data_file_creator(dict_args, class_nbr):
    '''Generate the file with paths for yolo v3 training
    The file will be save under 'obj.data' in data directory'''
    data_file = os.path.join(dict_args["train_data_dir"], "obj.data")
    with open(data_file, 'w') as data:
        line = f"classes = {class_nbr}\n"\
        f"train = {dict_args['train_data_dir'] + '/train.txt'}\n"\
        f"valid = {dict_args['train_data_dir'] + '/val.txt'}\n"\
        f"names = {dict_args['train_data_dir'] + '/obj.names'}\n"\
        f"backup = {dict_args['backup']}\n"
        data.write(line)


def dcm_to_array(image_path):
    '''Tranform dicom image format to a numpy array'''
    dcm_im = pydicom.read_file(image_path + ".dcm").pixel_array
    return dcm_im


def yolo_jpg_file(df, dict_args):
    '''Copy the choosen images in the right directory under jpg format'''
    for image_name in df.iloc[:, 0].unique():
        image = dcm_to_array(os.path.join(dict_args["input_train_data_dir"], image_name))
        cv2.imwrite(
            os.path.join(dict_args["train_images_dir"], image_name + ".jpg"), image
        )


def yolo_label_generation(df, target_folder, image_size):
    '''Generate label in the shape of yolo_v3 learning CNN:
    <object-class> <x_center> <y_center> <width> <height>
    relative value is required by yolo_v3 algorithm.
    one txt file per image is also required by yolo_v3 algorithm
    As mentionned in darknet repo, to improve results we need to add images without objects
    All created files will be saved in the same directory which contains jpg files'''
    for name, group in df.groupby("patientId"):
        label_file = os.path.join(target_folder, name + ".txt")
        with open(label_file, "w+") as file:
            for x, y, w, h, cl in group.iloc[:, 1:].values:
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


def yolo_image_path_file(df, dict_args, file_name):
    '''Generate a 'txt' file with the path and the name of each image'''
    txt_file = os.path.join(dict_args["train_data_dir"], file_name)
    with open(txt_file, "w+") as file:
        for image_name in df.iloc[:, 0].unique():
            line = "{}\n".format(os.path.join(dict_args["train_images_dir"], image_name + ".jpg"))
            file.write(line)


def pre_trained_weights_download(link, path):
    '''Download the pre-trained weights darknet53.conv.74 (162.5MB)'''
    url = link
    logging.info(
        '''Pre-trained weights 'darknet53.conv.74' downloading in progress (162.5MB).
        Please wait'''
    )
    wget.download(url, out=path)


def data_selection(df, split_rate):
    '''Produce different datasets. As required by the authors of Yolov3, we need
    to have DataFrames of same size
    '''
    # Dataframe with only images of pneumonia
    positive = df[df.iloc[:, -1] == 1]
    pos_size = len(positive)

    # Dataframe with only images of non pneumonia
    negative = df[df.iloc[:, -1] == 0]
    neg_size = min(pos_size, len(negative))
    negative = negative.iloc[: neg_size, :]

    df = pd.concat([positive, negative], axis=0)
    df = df.reset_index(drop=True)
    xs = df.iloc[:, :-1]
    ys = df.iloc[:, -1]

    x_train, x_val, y_train, y_val = train_test_split(
        xs, ys, test_size=1 - split_rate, random_state=42, stratify=ys
    )

    return x_train, x_val


def yolo_params_files_creation(dict_args, channel, obj, list_obj):
    '''Create all files wich will be used by Yolo v3 algorithm during the learning process'''
    cfg_file_creator(dict_args, channel, obj)
    names_file_creator(dict_args["train_data_dir"], list_obj)
    data_file_creator(dict_args, obj)


def algorithm_structure_creation(dict_args):
    '''Create the structure for the project and downoald necessary file'''
    folder_list = [
        dict_args["train_images_dir"], dict_args["test_images_dir"], dict_args["backup"]
    ]
    for name in folder_list:
        os.makedirs(name, exist_ok=True)

    copy_tree(
        dict_args["yolo_label"], os.path.join(dict_args["train_data_dir"], "labels")
    )

#    pre_trained_weights_download(constants.W_PATH, dict_args["project_dir"])


def visualisation(image_dir_path, df, index_patient):
    '''Display pneumonia or not image with or without the box'''
    if df.iloc[index_patient, -1]:
        patient_box = df[df.iloc[:, 0] == df.iloc[index_patient, 0]]
        for x, y, w, h in patient_box.iloc[:, 1:5].values:
            plt.plot([x, x, x + w, x + w, x], [y, y + h, y + h, y, y], label="pneumonia")
        plt.imshow(
            cv2.imread(os.path.join(image_dir_path, df.iloc[index_patient, 0] + '.jpg'))
        )
        plt.title("Pneumonia")
        plt.legend()
    else:
        plt.imshow(
            cv2.imread(os.path.join(image_dir_path, df.iloc[index_patient, 0] + '.jpg'))
        )
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
