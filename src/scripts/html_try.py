import os
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import cv2
import numpy as np
#References:
# (A)
# (B) 
#Load DATA - Original

Original = pd.read_csv('DATA-orig.csv')
#(A)
Original.style.hide_index()
Original = Original.transpose()
Original = Original\
    .to_html()\
    .replace('<table border="1" class="dataframe">','<table border="1" class="table table-striped">') # use bootstrap styling

#Load DATA - Training
Training = pd.read_csv('DATA.csv',header=[0,1])
Training.rename(index={0:'Training',1:'Testing'}, inplace=True)
#(A)
Training = Training\
    .to_html()\
    .replace('<table border="1" class="dataframe">','<table border="1" class="table table-striped">') # use bootstrap styling

table1 = [['predicted positives','123','11'], ['predicted negatives','22','234']]
df = pd.DataFrame( table1 )
df.columns = [' ', 'actual positives','actual negatives']
df.head()
summary_table_1 = df
summary_table_1 = summary_table_1\
    .to_html()\
    .replace('<table border="1" class="dataframe">','<table class="table table-striped">') # use bootstrap styling


def format_fig(fig):
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue())
    return encoded

#Load MODEL PARAMETERS
Model_params = pd.read_csv('MODEL-PARAMS.csv')
#(A)
Model_params = Model_params\
    .to_html()\
    .replace('<table border="1" class="dataframe">','<table border="1" class="table table-striped">') # use bootstrap styling

#RESULTS 

#Loss curves
df = pd.read_csv("../../results/results"+'_exp11'+".csv",header=None, names=["acc", "loss", "val_acc","val_loss"])

fig_loss = plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.plot(df["acc"])
plt.plot(df["val_acc"])
plt.ylim(0, 1.5)
plt.xlim(0,20)
plt.title("VGGNet architecture, lr = 0.01")
plt.grid()
plt.legend(['Training acc','Validation acc'],loc="upper right")

#Accuracy curves
fig_acc = plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(df["loss"])
plt.plot(df["val_loss"])
plt.ylim(0, 2.5)
plt.xlim(0,20)
plt.title("VGGNet architecture, lr = 0.01")
plt.grid()
plt.legend(['Training loss','Validation loss'],loc="upper right")

#Confusion matrix
metric = pd.read_csv("METRICS.csv")
conf_mat = pd.DataFrame([[metric['TP'][0], metric['FP'][0]],[metric['FN'][0], metric['TN'][0]]])
conf_mat.columns = ['Actual positives', 'Actual negatives']
conf_mat.index = ['Predicted positives', 'Predicted negatives']
conf_mat = conf_mat\
    .to_html()\
    .replace('<table border="1" class="dataframe">','<table border="1" class="table table-striped">') # 

#PREDICTIONS
f = open('positives.txt','r')
l = f.read().splitlines()
small = l[:4]
fig_ims = plt.figure()
for i in range(len(small)):
    ax = plt.subplot(240+i+1)
    ax.set_title(os.path.basename(small[i]), fontsize="4")
    ax.imshow(np.uint8(cv2.cvtColor(cv2.imread(small[i]), cv2.COLOR_BGR2RGB)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
plt.tight_layout() #Experimental feature?
#fig_ims = format_fig(fig_ims)

#fig_mark = plt.figure()
for i in range(len(small)):
    ax = plt.subplot(240+len(small)+i+1)
    splitted = small[i].rsplit('/',2)
    index = splitted[1].index('cropped')
    new_dir = splitted[1][:index]+"checkmark_"+splitted[1][index:]
    index = splitted[2].index('offset')
    new_name = splitted[2][:index]+"checkmark_"+splitted[2][index:]
    full_marked_name = splitted[0]+"/"+new_dir+"/"+new_name
    ax.set_title(os.path.basename(full_marked_name), fontsize="4", wrap="True")
    ax.imshow(np.uint8(cv2.cvtColor(cv2.imread(full_marked_name), cv2.COLOR_BGR2RGB)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
plt.tight_layout() #Experimental feature?
fig_ims = format_fig(fig_ims)
html_string = '''
<html>
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{ margin:0 100; background:whitesmoke; }
            .column {
                display: block;
                float: left;
                width: 50%;
                padding: 5px
             }
             .row::after {
                content: "";
                clear: both;
                display: table;
            }
            p {
              text-align:center;
            }
        </style>
    </head>
	<body>
	    <h2> DATA </h2>
            <h3> Original </h3>
            <h5> Bad solder crops are labelled as positive </h5>'''+Original+'''
            <h3> Training </h3>'''+Training+'''
	    <h2> MODEL PARAMETERS </h2>'''+Model_params+'''
	    <h2> RESULTS </h2>
	        <h3> Loss curves </h3>'''+"<img src='data:image/png;base64,{}'>".format(format_fig(fig_loss).decode("utf-8"))+'''
	        <h3> Accuracy curves </h3>'''+"<img src='data:image/png;base64,{}'>".format(format_fig(fig_acc).decode("utf-8"))+'''
	        <h3> Confusion Matrix </h3>'''+conf_mat+'''
	        <h3> Accuracy: {} &percnt;'''.format(metric['acc'][0])+''' </h3>
	        <h3> Precision: {} &percnt;'''.format(metric['precision'][0])+''' </h3>
	        <h3> Recall: {} &percnt;'''.format(metric['recall'][0])+''' </h3>
	        <h2> PREDICTIONS </h2>
	        <div class="row">   
	            <div class="column">'''+"<img src='data:image/png;base64,{}'>".format(fig_ims.decode("utf-8"))+'''
                    <p>This is image 1</p>
                </div>
                <div class="column">'''+'''<!--"<img src='data:image/png;base64, {}'>".format(fig_mark.decode("utf-8"))-->'''+'''
                    <p>This is image 2</p>
                </div>
            </div>    
        </body>
</html>'''
f = open('report.html','w')
f.write(html_string)
f.close()
