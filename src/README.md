## Models-EXP13.ipynb

The code for training is present in Models-EXP13.ipynb. This document gives an overview of the code structure and how to run this Jupyter Notebook. This notebook should be run after the circuit board has been already preprocessed and labelled. This notebook runs 6-fold cross validation but to run just a single iteration, there are also a few changes that need to be made in CELL 10. These changes will be described in one of the following sections.

### Data path

CELL 2 is used to initialize the data-path, results path and image extension type respectively. The data-path should be something like `home/foo/base-path/`.  
This paths contains all DATE/ folders, each with OK/ and NG/. It is also assumed that this path contains the csv with labels, and has the name **labels-full-384.csv**

### Loading data

Cells 3 and 4 are used to load and augment data respectively. Labels are assigned to the images in CELL 6.  
Running CELL 7 is optional, it saves the numpy arrays to disk, but that is not necessary for the rest of the code.  
CELL 8 splits the data into training and testing. The default ratio is set to 80:20. (*Please comment or markdown CELL 8 if you want to run cross-validation*)

### Training and validation

CELLS 9 and 10 initialize the model and train it, respectively.   
CELL 11 evaluates predictions on the test data and prints accuracy and loss (This cell is optional since predictions are also done in CELL 10)

### Changes to be made in CELL 10 for single iteration

In order to run this CELL for a single training and validation session, following changes need to be made:
* Comment the for-loop statement while retaining the loop body
* Everywhere in the loop body, make following changes to variables:
  - `Xdata[train]` -> `Xtrain`
  - `Xdata[val]` -> `Xval`
  - `Ydata[train]` -> `Ytrain`
  - `Ydata[val]` -> `Yval`

### Display results

Results are saved in the file *../results/results_exp13.csv* file. CELLS 12-14 are used to save and display the loss and accuracy curves. Loss and accuracy curves are saved in *../results/Acc-loss-curves/* 
CELL 15 performs predictions on the test set and displays accuracy, precision, recall and confusion matrix. (This cell, too, is optional since these calculations are also done in CELL 10)

**Please Note** :  
The confusion matrix displayed by the library is a little different than usual. This is the format that it uses:  
TN | FP  
.............  
FN | TP  

### Running the notebook

Each cell of this notebook can be run individually one after the other. Please note that *CELL 12* needs to be run twice to save the acc and loss results in the csv file. This can be fixed by adding `f.flush()` statement at the end of CELL 12.  
