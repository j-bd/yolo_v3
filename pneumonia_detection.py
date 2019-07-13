#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:51:18 2019

@author: latitude
"""

import numpy as np
import time
import cv2
import os

project_dir_path = "/home/latitude/Documents/Kaggle/rsna-pneumonia/yolo_v3/"
image_path = project_dir_path + "test_data/obj/0a0f91dc-6015-4342-b809-d19610854a21.jpg"

def detect(image_path, project_dir_path, confidence=0.5, threshold=0.0025):
    '''Detection of pneumonia on images'''
    labels_path = project_dir_path + "data/obj.names"
    weights_path = project_dir_path + "data/p_1400.weights"
    config_path = project_dir_path + "data/yolo-obj.cfg"

    labels = open(labels_path).read().strip().split("\n")

    #Load YOLOv3 structure
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    #Load image
    image = cv2.imread(image_path)
    H, W = image.shape[:2]

    #Output layer names from YOLOv3
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print(f"[INFO] Processing time : {end - start}seconds")
