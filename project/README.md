# Intelligent Visual Inspection for Solder Defect Detection

## Overview

### Introduction
Quality analysis of printed circuit boards is a task requiring meticulous attention to detail. Manually inspecting circuit boards for defects is not only time-consuming but also prone to human-errors. This project presents an deep-learning based automatic inspection system for detection of solder defects on a printed circuit board. The system generates a dataset with binary labels which is used to train a [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network).The network is trained so as to learn the differences between difective and non-defective solders and be able to accurately classify new data into the two groups. The system is broadly divided into two phases - data preprocessing and training the deep learning architecture.  
### Data Preprocessing 
As mentioned before, this module generates the dataset that will be used for model training. It accepts circuit board images as input, partitions each of them into smaller sub-images and assigns labels automatically. It assigns a positive label to a sub-image with at least one defective solder and a negative label otherwise.  
### Deep Learning architecture 
The neural network architecture is inspired from the VGGNet16 network [1], an award-winnning entry in the ILSRVC 2014 competition. This network, tested on a 224x224 image, was aimed at improving classification accuracy by increasing the depth. Following the same line of thought, we further increased the depth of VGGNet16 as shown in the figure below, starting with an input image of size 384x384 as shown below:  
[vggnet16-384](architecture.png)  
 The values of hyperparameters are as follows: 

epochs: 74  
batch_size: 32  
learning_rate: 0.01  
l1: 0  
l2: 0  
where l1 and l2 are [regularization parameters](https://en.wikipedia.org/wiki/Regularization_(mathematics))
## Structure

The system is divided into three Python modules - preprocessing, training, report_generation - that are sequentially run from the project script found in project/src/  

### Running the project

(Should we mention any assumptions about data directory structure before running the code?)
Switch to the project/src/ directory and run the project by invoking the following commands:  
`$ chmod +x project`  
`$ ./project </path /to/ circuit/ boards> -e(or --epochs) <epochs> -b(or --batch_size) <batch_size> --lr < learning_rate> --l1 <l1_regularizer> --l2 <l2_regularizer>`.  
It takes the path to circuit board images as a required argument. The rest of the arguments - preceded by either '-' or '--' - are optional arguments followed by their respective values. The default values are the ones mentioned in the Deep Learning architecture section above. These  hyperparameters gave optimal results out of all the experiments that we conducted. However, the user is free to tune them directly through the command-line.   
#### Preprocessing 
The preprocessing script takes as input, a path to all the circuit board images. Th script assumes that this path contains folders 'F' each of which has sub-folders 'OK' and 'NG'. 'OK' contains defect-free circuit board images while 'NG' has the rest. It crops and labels these images as mentioned in the Data Preprocessing section above. It then generates a CSV file with 'crop-path, label' pairs and saves it in *</path /to/ circuit/ boards>*.  

#### Training
This script takes in the data file, hyperparameters and trains the model. The default train:test split used is 80:20 (should we keep that as command-line argument?). It saves the loss and accuracy history in '../results/results.csv'. It also computes the precision, recall and stores it in 'METRICS.csv' in '../results/'.  
It also generates another CSV file 'CURRENT-DATA.csv' that records number of available circuit boards, positives and negatives for the current data. It saves this file in '../results/' folder, relative to the source folder


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
   `$ conda create -n yourenvname python=x.x anaconda`  
   where x.x is the Python version you wish to use
3. Install above libraries
   - First switch to the environment you want to install everything in  
   `$ conda activate yourenvname`
   - Then install above packages in the new virtual environment  
`$ conda install tensorflow-gpu`   
(The above command also installs compatible version of keras and cudnn)  
`$ conda install numpy`  
`$ conda install pandas`  
`$ conda install -c conda-forge matplotlib`  
`$ conda install scipy`  
`$ conda install sklearn`  
`$ conda install -c menpo opencv`
     
## Sample results
Please refer the sample report *project_report.html* found in the *project/results/* folder presenting results for 74 epochs of the network.

## References
[1] K. Simonyan and A. Zisserman, "*Very deep convolutional networks for large-scale image recognition*", In *ICLR*, 2015.
