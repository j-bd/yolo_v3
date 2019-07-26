#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:51:18 2019

@author: j-bd
"""
import time

import numpy as np
import cv2


def submission_file(output_path, result):
    '''Create a '.csv' file with all analysis to be exported to Kaggle'''
    with open(output_path, 'w') as file:
        for line in result:
            line = line + "\n"
            file.write(line)


def test_cfg_file(project_dir, test_data_dir, size):
    '''Create a new '.cfg' file for yolo v3 detection as recommand by authors. We increase the
    network-resolution by changing the size of 'height' and 'width'. Note that we need to keep a
    value multiple of 32'''
    input_cfg = project_dir + "darknet/cfg/yolov3.cfg"
    with open(input_cfg, 'r') as cfg_in:
        new_cfg = cfg_in.read()

    new_cfg = new_cfg.replace('width=608', 'width=' + str(size))
    new_cfg = new_cfg.replace('height=608', 'height=' + str(size))

    output_cfg = test_data_dir + "yolo-obj_test.cfg"
    with open(output_cfg, 'w') as cfg_out:
        cfg_out.write(new_cfg)

    return output_cfg


def detect(image_path, weights_path, config_path, confidence=0.5, threshold=0.0025, show=False):
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
    boxes = list()
    confidences = list()
    final_boxes = list()
    final_boxes.append(image_path.split(sep="/")[-1].split(sep=".")[0] + ",")
    for output in layerOutputs:
        for detection in output:
            # if pneumonia have been detect, returns the center (x, y)-coordinates of the box,
            # followed by the width, the height and finally the confidence
            if detection[5] > confidence:
                #transform coordinates from relative size to image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(detection[5]))
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
            final_boxes.append(str(round(confidences[i], 2)) + " " + str(x) + " " + str(y) + " " +
                               str(w) + " " + str(h) + " ")
            if show:
                # draw a bounding box rectangle and label on the image
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "Pneumonia: {:.4f}".format(confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # show the output image
                cv2.imshow("Image", image)

    return ''.join(final_boxes)
