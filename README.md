# Implementation of Yolo v3

You will find here a test application of Yolo v3 deep learning method to an images detection challenge.


## Overview of Yolo v3

For more information about Yolo v3, you can download the [article](https://pjreddie.com/media/files/papers/YOLOv3.pdf) of Joseph Redmon and Ali Farhadi.

The biggest interest of this algorithm is the speed of process:
![Yolov3 inference time](https://pjreddie.com/media/image/map50blue.png)

You can find more information directly on the [website](https://pjreddie.com/darknet/yolo/)


## Data source

I used data made available by "Radiological Society of North America" on Kaggle. The challenge name is ["RSNA Pneumonia Detection Challenge"](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).

![Kaggle_rsna](https://storage.googleapis.com/kaggle-competitions/kaggle/10338/logos/header.png?t=2018-08-21-19-48-11)

The second version of data is organised as :
* A train folder with about 26 684 images,
* A test folder with about 3 000 images,
* A csv file named "stage_2_train_labels.csv" with the bounding boxes of all train images,
* A csv file named "sample_submission_v2.csv" to show the shape of the output expected.


## Before any code execution

Before launching this algorithm make sure the Kaggle data are organized as following in a master directory:
* A directory with your images test named 'stage_2_test_images'
* A directory with your images train named 'stage_2_train_images'
* A detailed CSV file train labels named 'stage_2_train_labels.csv'
* A detailed CSV file for submission named 'stage_2_sample_submission.csv'
All this elements must be gather in the same directory. The path will be mention when launching the algorithm.
[Source](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)

Before launching this algorithm, you need to clone darknet (yolov3 package) in your project directory (to be created by yourself and mentionned when launching the algorithm). To do it, please follow the instructions on this [website](https://pjreddie.com/darknet/install/).
Check options available before enter ```make``` command as GPU and so on.

To train yolo_v3 algorithm to detect our custom objects we need to follow this steps:
* Create a file named 'yolo-obj.cfg' as a configuration of the CNN (a custom copy of yolov3.cfg)
* Create a file named 'obj.names' with the names of our custom object. Here is 'pneumonia'
* Create a file named 'obj.data' with the numbers of class objects and path to different files
* Gather all images ('.jpg' format) in a same directory
* Create a '.txt' file with all labels in relative float value
* Create a '.txt' file with the path to all training images
* Create a '.txt' file with the path to all validation images
* Download the right [pre-trained weights file](https://github.com/AlexeyAB/darknet)

To save model files more regularly during the training, we need to modify the code of this following file ```darknet/examples/detector.c``` (around the line 138) :
```
    if(i%10000==0 || (i < 1000 && i%100 == 0)){
```
to:
```
    if(i%1000==0 || (i < 2000 && i%50 == 0)){
```
We now save in the backup folder every 50 iterations a '.weights' file till we reach 2000 and then we save after every 1000 iterations


## Global organization of the repository

The folder **data** gather few files automatically generated by my algorithms.

The file ```pneumonia_functions.py``` is mainly dedicated to prepare the algorithm to learn a new object. It offers the possibility to:
* create the structure of the project and download automatically a pre trained weight given by the author of Yolo v3 ```def_structure```
* prepare different config files used by Yolo v3 to learn new objects ```def yolo_parameters```:
  - ```def yolo_cfg_file```: generate the config file for yolo v3 training
  - ```def yolo_names_file```: name(s) of the new object(s)
  - ```def yolo_data_file```: paths of all files needed for yolo v3 training
* set up the train et val dataset with ```def data_preprocessing```
* transform medical images (.dcm) into ".jpg" files with ```def yolo_jpg_file``` and ```def dcm_to_array```
* generate label in the shape of yolo_v3 learning CNN with ```def yolo_label_generation```. This function indicate to the algorithm where is located the object in the image training or image validation
* generate a '.txt' file with the path and the name of each image ```def yolo_image_path_file```
* display pneumonia or not image with or without the box ```def visualisation```
* represent the loss trend line over the learning process ```def loss_function```

The file ```pneumonia_detection.py```is dedicated to the detection on the test images. It allows to:
* create a new '.cfg' file for yolo v3 detection as recommended by authors ```def test_cfg_file```
* launch detection on submitted image ```def detect```
* record all results under the right shape to be send to Kaggle ```def submission_file```

I offer the possibility to execute automatically all the algorithm trough ```main.py```

