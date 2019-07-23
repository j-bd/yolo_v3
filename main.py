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
All this elements must be gather in the same directory. The path will be mention in the following
variable 'IMAGE_DIR'.
Source: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

Before launching this algorithm, you need to clone darknet (yolov3 package) in your project
directory (variable: PROJECT_DIR). To do it, please follow the instructions on this website:
https://pjreddie.com/darknet/install/
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

import pneumonia_functions
import pneumonia_detection


def main():
    "Allow the selection between algorithm training or image detection"
    argp = argparse.ArgumentParser()



if __name__ == "__main__":
    main()