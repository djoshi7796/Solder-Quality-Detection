# Intelligent Visual Inspection for Solder Defect Detection

## Overview

### Introduction
Quality analysis of printed circuit boards is a task requiring meticulous attention to detail. Manually inspecting circuit boards for defects is not only time-consuming but also prone to human-errors. This project presents an deep-learning based automatic inspection system for detection of solder defects on a printed circuit board. Upon receiving an image of an circuit board, the system partitions it into smaller sub-images and automatically assigns labels to each of them. It attempts to assign a positive label to a sub-image with at least one defective solder and a negative label otherwise. The system is broadly divided into two phases - data preprocessing and model training.  
### Data Preprocessing  

### Deep Learning architecture 
The neural network architecture is inspired from the VGGNet16 network [1], an award-winnning entry in the ILSRVC 2014 competition. This network, tested on a 224x224 image, was aimed at improving classification accuracy by increasing the depth. Following the same line of thought, we further increased the depth of VGGNet16 as shown in the figure below, starting with an input image of size 384x384 as shown below:  
[vggnet16-384](architecture.png)  
 The values of hyperparameters are as follows: 

epochs: 74  
batch_size: 32  
learning_rate: 0.01  
l1: 0  
l2: 0  
  
## Structure

The system is divided into three Python modules - preprocessing, training, report_generation - that are sequentially run from the project.sh script

### Run project.sh

(Should we mention any assumptions about data directory structure before running the code?)
The project can be run by invoking the following command:  
`*chmod +x project*'
`*./project </path /to/ circuit/ boards> -e(or --epochs) <epochs> -b(or --batch_size) <batch_size> --lr < learning_rate> --l1 <l1_regularizer> --l2 <l2_regularizer>*`.  
It takes the path to circuit board images as a required argument. The rest of the arguments - preceded by either '-' or '--' - are optional arguments followed by their respective values. The default values are the ones mentioned in [dl arch](### Deep Learning architecture). These  hyperparameters gave optimal results out of all the experiments that we conducted. However, the user is free to tune them directly through the command-line.   
#### Preprocessing 
The preprocessing script generates a CSV file with 'crop-path, label' pairs and saves it in *</path /to/ circuit/ boards>*.  
It also generates another CSV file 'CURRENT-DATA.csv' that records number of available circuit boards, positives and negatives for the current data. It saves this file in '../results/' folder, relative to the source folder

#### Training
This script takes in the data file, hyperparameters and trains the model. The default train:test split used is 80:20 (should we keep that as command-line argument?). It saves the loss and accuracy history in '../results/results.csv'. It also computes the precision, recall and stores it in 'METRICS.csv' in '../results/'. 

#### Report generation
This module generates an HTML report stored in '../results/' This report displays the number of positives and negatives available in the data set, number of positives and negatives used for training, model hyperparameters, accuracy & loss curves, and the evaluated results. It also shows a sample of positives that were detected by the system and also those that were missed.  
## System requirements
### Python modules
* tensorflow 1.13.1
* keras 2.2.4-tf
* cudnn 
* cv2 3.4.2
* pandas 0.24.2
* numpy 1.16.4
* matplotlib 3.1.0
* sklearn 0.21.2
* scipy 1.2.1

### Installation in conda virtual environment

This project was written in the conda virtual environment. 
1. Please follow the link to download the Anaconda distribution for Python 3.7 on 64-bit (x86) Ubuntu:
https://www.anaconda.com/distribution/
2. Create a virtual environment  
   `conda create -n yourenvname python=x.x anaconda`  
   where x.x is the Python version you wish to use
3. Install above libraries
   - First switch to the environment you want to install everything in  
   `conda activate yourenvname`
   - Then install above packages in the new virtual environment  
`conda install tensorflow-gpu`   
(The above command also installs compatible version of keras and cudnn)  
`conda install numpy`  
`conda install pandas`  
`conda install -c conda-forge matplotlib`  
`conda install scipy`  
`conda install sklearn`  
`conda install -c menpo opencv`
     
## Sample results
Please refer the sample report presenting results for 20 epochs of the network. [report](report_actual.html)
##References
[1] K. Simonyan and A. Zisserman, "*Very deep convolutional networks for large-scale image recognition*", In *ICLR*, 2015.
