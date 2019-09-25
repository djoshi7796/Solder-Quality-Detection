## Models-EXP13.ipynb

The code for training is present in Models-EXP13.ipynb. This document gives an overview of the code structure and how to run this Jupyter Notebook.

### Data path

CELL 2 is used to initialize the data-path, results path and image extension type respectively. The data-path should be something like `home/foo/base-path/`.  
This paths contains all DATE/ folders, each with OK/ and NG/. It is also assumed that this path contains the csv with labels, and has the name **labels-full-384.csv**

### Loading data

Cells 3 and 4 are used to load and augment data respectively. Labels are assigned to the images in CELL 5.  
Running CELL 6 is optional, it saves the numpy arrays to disk, but that is not necessary for the rest of the code. 

### Training and validation

CELLS 9 and 10 initialize the model and train it, respectively.   
CELL 11 evaluates predictions on the test data and prints accuracy and loss (This cell is optional too, if you run CELLS 12-15 as discussed below)

### Display results

Results are saved in the file *../results/results_exp13.csv* file. CELLS 12-14 are used to save and display the loss and accuracy curves.  
CELL 15 performs predictions on the test set and displays accuracy, precision, recall and confusion matrix. 

**Please Note** :  
The confusion matrix displayed by the library is a little different than usual. This is the format that it uses:  
TN | FP  
.............  
FN | TP
