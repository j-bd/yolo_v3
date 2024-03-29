#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:51:18 2019

@author: j-bd
"""
import os
import time
import logging

import numpy as np
import cv2


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def submission_file(output_path, result):
    '''Create a '.csv' file with all analysis to be exported to Kaggle'''
    with open(output_path, 'w') as file:
        for line in result:
            line = line + "\n"
            file.write(line)

def test_cfg_file(dict_args, batch, subd, class_nbr):
    '''Create a new '.cfg' file for yolo v3 detection as recommand by authors.
    We increase the network-resolution by changing the size of 'height' and
    'width'. Note that we need to keep a value multiple of 32'''
    input_cfg = os.path.join(dict_args["project_dir"], "darknet/cfg/yolov3.cfg")
    with open(input_cfg, 'r') as cfg_in:
        new_cfg = cfg_in.read()

    max_batches = 2000 * class_nbr
    steps = str(max_batches * 0.8) + ',' + str(max_batches * 0.9)
    filter_yolo = str((class_nbr + 5) * 3)
    new_cfg = new_cfg.replace('batch=64', 'batch=' + str(batch))
    new_cfg = new_cfg.replace('subdivisions=16', 'subdivisions=' + str(subd))
    new_cfg = new_cfg.replace(
        'max_batches = 500200', 'max_batches =' + str(max_batches)
    )
    new_cfg = new_cfg.replace('steps=400000,450000', 'steps=' + steps)
    new_cfg = new_cfg.replace('classes=80', 'classes=' + str(class_nbr))
    new_cfg = new_cfg.replace('filters=255', 'filters=' + filter_yolo)
    new_cfg = new_cfg.replace('width=608', 'width=' + str(dict_args["detect_im_size"]))
    new_cfg = new_cfg.replace('height=608', 'height=' + str(dict_args["detect_im_size"]))

    output_cfg = os.path.join(dict_args["test_data_dir"], "yolo-obj_test.cfg")
    with open(output_cfg, 'w') as cfg_out:
        cfg_out.write(new_cfg)

    return output_cfg

def detect(image_path, weights_path, config_path, confidence, threshold, show=False):
    '''Detection of pneumonia on single image'''
    # Load YOLOv3 structure
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    # Load image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    # Output layer names from YOLOv3
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # Image preprocessing and Yolov3 detection
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=False, crop=False
    )
    net.setInput(blob)

    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()

    # Initialize our lists of detected bounding boxes, confidences
    boxes = list()
    confidences = list()
    final_boxes = list()
    final_boxes.append(image_path.split(sep="/")[-1].split(sep=".")[0] + ",")
    for output in layer_outputs:
        for detection in output:
            # If pneumonia have been detect, returns the center (x, y)-coordinates
            # of the box, followed by the width, the height and finally the confidence
            if detection[5] > confidence:
                # Transform coordinates from relative size to image size
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")
                # Use the center (x, y)-coordinates to derive the top
                # and left corner of the bounding box
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                # Update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(detection[5]))
    # Apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            x, y, w, h = boxes[i]
            final_boxes.append(f"{round(confidences[i], 2)} {x} {y} {w} {h}")
            if show:
                # Draw a bounding box rectangle and label on the image
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "Pneumonia: {:.4f}".format(confidences[i])
                cv2.putText(
                    image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                # Show the output image
                cv2.imshow("Image", image)

    logging.info(
        f"Processing time: {round(end - start, 2)}seconds "
        f"- {len(idxs)} object(s) detected"
    )

    return ' '.join(final_boxes)


def image_detection(cfg_path, images, dict_args):
    "Lauch detection for images"
    output_path = os.path.join(dict_args["test_data_dir"], "submission.csv")
    result = list()
    result.append("patientId,PredictionString")

    for image in images:
        logging.info(f"{images.index(image) + 1} / {len(images)}")
        box_coordinate = detect(
            image, dict_args["weights_path"], cfg_path, dict_args["confidence"],
            dict_args["threshold"]
        )
        result.append(box_coordinate)
    submission_file(output_path, result)

    logging.info("All images have been proceed")
