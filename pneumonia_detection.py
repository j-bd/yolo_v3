#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:51:18 2019

@author: latitude
"""
import time

import numpy as np
import cv2


def detect(image_path, weights_path, config_path, confidence=0.5, threshold=0.0025):
    '''Detection of pneumonia on images'''
    #Load YOLOv3 structure
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    #Load image
    image = cv2.imread(image_path)
    H, W = image.shape[:2]
    #Output layer names from YOLOv3
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # Image preprocessing and Yolov3 detection
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=False, crop=False)
    net.setInput(blob)

    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print(f"[INFO] Processing time : {end - start}seconds")

    # initialize our lists of detected bounding boxes, confidences
    boxes = []
    confidences = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the confidence of pneumonia detection if it does exist
            # returns the center (x, y)-coordinates of the box followed the width and height
            conf = detection[5]
            if conf > confidence:
                #transform coordinates from relative size to image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(conf))
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "Pneumonia: {:.4f}".format(confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # show the output image
    cv2.imshow("Image", image)


#project_dir_path = "/home/latitude/Documents/Kaggle/rsna-pneumonia/yolo_v3/"
#weights_path = project_dir_path + "test_data/p_1400.weights"
#config_path = project_dir_path + "test_data/yolo-obj_test.cfg"
#image_path = project_dir_path + "test_data/obj/0a8d486f-1aa6-4fcf-b7be-4bf04fc8628b.jpg"
##/home/latitude/Documents/Kaggle/rsna-pneumonia/yolo_v3/test_data/obj/0a9c4c04-5918-48da-9ba1-b2a3add66ae9.jpg
#detect(image_path, weights_path, config_path, 0.5, 0.0025)
