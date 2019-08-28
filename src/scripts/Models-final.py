#!/usr/bin/env python
# coding: utf-8

# ## Instructions to run this code:
# 
# ##### This notebook is for EXP13 : i.e., using the VGGNet architecture on 384x384 grid-size with cross-validation and 74 epochs 
# 
# ##### This notebook can be run from start to end. Please check and change the data path before running if needed
# 
# ##### The results will be saved in the "../results/" folder  

# In[1]:


#REFERENCES:
#https://medium.com/@tifa2up/image-classification-using-deep-neural-networks-a-beginner-friendly-approach-using-tensorflow-94b0a090ccd4
#https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/


# In[1]:


import argparse
import numpy as np
import pandas as pd
import csv
import cv2
import glob
import shutil
import os
import subprocess
import random
import tensorflow
import itertools
from tensorflow.keras.models import model_from_json
import sklearn
import scipy
from collections import defaultdict
from scipy.ndimage import rotate
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
plt.figure(figsize=(12,6))
import pydot


def load_dataset(path, split=0.8):
    X = [[], []]
    #Y = []
    data = []
    with open(path,"r") as f:
        reader = csv.reader(f)
        for row in reader:
            (name, label) = tuple(row)
            img = name
            if label == "1":
                X[1].append(img)
            else:
                X[0].append(img)
        f.close()
        return (X)

def Model(learn_rate=0.01, L1=0, L2=0, fc=4096, mm = 0.9):
#    K.clear_session()
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',input_shape=(new_height, new_width, ch)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(fc, activation='relu', kernel_regularizer=regularizers.l1_l2(l1 = L1, l2 = L2)))
    model.add(Dense(fc, activation='relu', kernel_regularizer=regularizers.l1_l2(l1 = L1, l2 = L2)))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=SGD(lr = learn_rate, momentum=mm), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# In[2]:

if __name__ == "__main__":

#Change this to local path before running
    
    
    parser = argparse.ArgumentParser()
    
    results = "../results/"
    ext = ".bmp"
    # In[3]:
    #path, [Epochs, batch size, learning rate, l1, l2]
    parser.add_argument('path', help='Path to circuit boards')
    parser.add_argument('-e','--epochs',default=74, type=int, help="Number of epochs to train on")
    parser.add_argument('-b','--batch_size',default=32, type=int, help="Number of examples in a single batch")
    parser.add_argument('--lr',default=0.01, type=float, help="Learning rate")
    parser.add_argument('--l1',default=0, type=float, help="L1 regularizer")
    parser.add_argument('--l2',default=0, type=float, help="L2 regularizer")
    parser = parser.parse_args()
    #suffix for saving data"
    #Change this whenever running different experiments
    EXP = "_final"
    lr = parser.lr  #0.01
    l1 = parser.l1 #0
    l2 = parser.l2 #0
    epochs = parser.epochs #10
    batch_size = parser.batch_size #32
    data_parent = parser.path
    fModel = open('MODEL-PARAMS.csv','w')
    writer = csv.writer(fModel)
    writer.writerow(['Epochs','Batch size','Learning rate','L1 regularizer','L2 regularizer'])
    writer.writerow([epochs, batch_size, lr, l1, l2])
    fModel.flush()
    print("Params : ", epochs, batch_size, lr, l1, l2)
    # #TODO: check pros and cons for os, subprocess, shutil - MAKE A NOTE SOMEWHERE

    grid_size = 384
    print("Load CSV file...")
    X = load_dataset(data_parent+"labels-full-"+str(grid_size)+".csv")
    random.shuffle(X[0])
    random.shuffle(X[1])
    circuits_neg = [x.split("offset",1)[0] for x in X[0]]
    circuits_pos = [x.split("offset",1)[0] for x in X[1]]

    #DEBUG OUTPUT
    #for i in sorted(set(circuits_neg)):
    #    print(i)
    
    #print(" NEGATIVE DONE ")
    #for i in sorted(set(circuits_pos)):
    #    print(i)
    circuits =  len(set(circuits_neg).union(set(circuits_pos)))
    print("Circuits: ",circuits)
    print("Positives : ", len(X[1]))
    print("Negatives : ", len(X[0]))
    fDataOrig = open("DATA-orig.csv","w")
    writer = csv.writer(fDataOrig)
    writer.writerow(['Circuits','Positives','Negatives','Grid size'])
    writer.writerow([circuits, len(X[1]), len(X[0]), grid_size])

    # In[4]:


    t_o = 6 #Total image operations: identity, rot90, rot180, rot270, flipUpDown, flipLeftRight
    upper_limit = len(X[1])*t_o
    del X[0][len(X[1]):]
    P = len(X[1])
    N = len(X[0])
    print("number of samples: ", N)
    labels = []
    dummy_img = cv2.imread(X[1][0])
    plt.imshow(dummy_img)
    plt.show()
    plt.imshow(rotate(dummy_img, 30, reshape=True))
    plt.show()
    (height, width, ch) = dummy_img.shape
    Xdata = np.zeros((upper_limit*2, height, width, ch), dtype=np.int8)
    print("Augmenting data...")
    percent=0
    for i in range(N):
        img = cv2.imread(X[0][i])
        Xdata[t_o*i] = img
        Xdata[t_o*i+1] = rotate(img, 90, reshape=False)
        Xdata[t_o*i+2] = rotate(img, 180, reshape=False)
        Xdata[t_o*i+3] = rotate(img, 270, reshape=False)
        Xdata[t_o*i+4] = np.flipud(img)
        Xdata[t_o*i+5] = np.fliplr(img)
        labels.append((X[0][i],False))
        for j in range(5):
            labels.append((0, False))
        if i == 2*percent*N//100:
            print("{}% done ...".format(percent))
            percent += 10
    for i in range(P):
        
        img = cv2.imread(X[1][i])
        Xdata[upper_limit+t_o*i] = img
        Xdata[upper_limit+t_o*i+1] = rotate(img, 90, reshape=False)
        Xdata[upper_limit+t_o*i+2] = rotate(img, 180, reshape=False)
        Xdata[upper_limit+t_o*i+3] = rotate(img, 270, reshape=False)
        Xdata[upper_limit+t_o*i+4] = np.flipud(img)
        Xdata[upper_limit+t_o*i+5] = np.fliplr(img)
        labels.append((X[1][i], True))
        for j in range(5):
            labels.append((0, True))
        if i == 2*(percent-50)*P//100:
            print("{}% done ...".format(percent))
            percent += 10

    X = Xdata
    del Xdata

    
    Y = [0 for i in range(2*upper_limit)]
    for i in range(upper_limit):
        Y[upper_limit+i] = 1
    #Ydata = np.float32(Y)
    Ydata = to_categorical(Y)
    del Y
    Xdata, Y, names = sklearn.utils.shuffle(X, Ydata, labels)
    #THIS IS OPTIONAL
    np.save("../../Augmented-data/data"+EXP, Xdata)
    np.save("../../Augmented-data/labels"+EXP,Y)
    del X
    del Ydata
    del labels


    # In[7]:


    #print(Xdata.shape)

    #THIS IS OPTIONAL
    Xdata = np.load("../../Augmented-data/data"+EXP+".npy")
    Ydata = np.load("../../Augmented-data/labels"+EXP+".npy")
    (new_height, new_width, ch) = Xdata[0].shape
    print("Data loaded")

    # In[5]:


    Xtrain = Xdata[:int(0.8*len(Xdata))]#[:32]
    Xval = Xdata[int(0.8*len(Xdata)):]
    Ytrain = Ydata[:int(0.8*len(Ydata))]#[:32]
    Yval = Ydata[int(0.8*len(Ydata)):]
    names_train = names[:int(0.8*len(names))]#[:32]
    names_val = names[int(0.8*len(names)):]
    del Xdata
    del Ydata
    fData = open("DATA.csv","w")
    writer = csv.writer(fData)
    writer.writerow(['Original','Original','Augmented','Augmented','Total','Total'])
    writer.writerow(['Positive','Negative','Positive','Negative','Positive','Negative'])
    train_remain_pos = sum([1 for x in names_train if not x[0] and x[1]])
    test_remain_pos =  sum([1 for x in names_val if not x[0] and x[1]])
    train_remain_neg =  sum([1 for x in names_train if not x[0] and not x[1]])
    test_remain_neg =  sum([1 for x in names_val if not x[0] and not x[1]])
    len_names_train_pos = sum([1 for x in names_train if x[1]])
    len_names_train_neg = sum([1 for x in names_train if not x[1]])
    len_names_val_pos = sum([1 for x in names_val if x[1]])
    len_names_val_neg = sum([1 for x in names_val if not x[1]])

    writer.writerow([len_names_train_pos-train_remain_pos,len_names_train_neg-train_remain_neg,train_remain_pos,train_remain_neg,len_names_train_pos,len_names_train_neg])
    writer.writerow([len_names_val_pos-test_remain_pos,len_names_val_neg-test_remain_neg,test_remain_pos,test_remain_neg,len_names_val_pos,len_names_val_neg])
    fData.flush()
    # In[6]:

    #Reference: https://keras.io/applications/#vgg16
    #Reference: https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
    #Reference: https://stackoverflow.com/questions/44495698/keras-difference-between-kernel-and-activity-regularizers
    #For now we are using kernel (i.e. weights) regularization


    # model_json = model.to_json()
    # with open("../Models/model"+EXP+".json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("../Models/model"+EXP+".h5")


    # In[6]:


    #Reference : https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/

    #CROSSVAL WAS HERE
    model = Model(learn_rate=lr, L1=l1, L2=l2)
    history = model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, validation_data=(Xval, Yval))

    print(history.history)
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    fResults = open("../../results/results"+EXP+".csv","w")
    writer = csv.writer(fResults)
    for i in range(len(val_acc)):
        writer.writerow([acc[i], loss[i], val_acc[i], val_loss[i]])
        fResults.flush()

    # #Metrics
    # # predict probabilities for test set
    # 
    yhat_probs_class = model.predict(Xval, verbose=0)
    yhat_probs = yhat_probs_class[:, 0]
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(Xval, verbose=0)
    yhat_classes = np.argmax(yhat_probs_class, axis=1)
    Yval_ = np.argmax(Yval, axis=1)
    accuracy = accuracy_score(Yval_, yhat_classes)
    tp_index = []
    fn_index = []
    for i in range(len(Yval_)):
        if Yval_[i] and yhat_classes[i]:
            tp_index.append(i)
        if Yval_[i] and not yhat_classes[i]:
            fn_index.append(i)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    # precision = precision_score(Yval_, yhat_classes)
    # print('Precision: %f' % precision)
    # # recall: tp / (tp + fn)
    # recall = recall_score(yhat_classes, Yval_)
    # print('Recall: %f' % recall)
    # fpr, tpr, thresholds = metrics.roc_curve(Yval_, yhat_probs, pos_label=2)
    # auc = roc_auc_score(Yval_,yhat_probs)
    # print('ROC AUC: %f' % auc)
    # print(list(fpr))
    # confusion matrix
    matrix = confusion_matrix(Yval_, yhat_classes)
    TN = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TP = matrix[1][1]
    print(matrix)
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # #Fraction of predicted positives that are actually positive
    precision = TP/(TP+FP)
    print("Precision: ", precision)
    # #Fraction of positives predicted correctly
    recall = TP/(TP+FN)
    print("Recall: ",recall)
    f = open("METRICS.csv","w")
    writer = csv.writer(f)
    writer.writerow(['acc','TP','FP','TN','FN','precision','recall'])
    writer.writerow([accuracy,TP,FP,TN,FN,precision,recall])
    fTP = open("positives.txt","w")
    fFN = open("negatives.txt","w")
    for tp in tp_index:
        if names_val[tp][0]:
            fTP.write(names_val[tp][0]+'\n')
    for fn in fn_index:
        if names_val[fn][0]:
            fFN.write(names_val[fn][0]+'\n')




