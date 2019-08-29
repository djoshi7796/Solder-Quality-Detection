#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### Either store centroids while making grid or calculate from offset and image number
#### Setting grayscale background threshold : for original image or cropped image? try both
#### ==> Using threshold for the cropped image works well 
#### Also, test the detect red function on the transformed images/grids: IMPORTANT: Can marked full solders be detected ?? ://


# In[2]:


import sys
sys.executable


# In[3]:


import cv2
from math import gcd
import subprocess
import matplotlib.pyplot as plt
import os, shutil, glob
#from skimage import measure
import numpy as np
import csv
import itertools
import random
import argparse
#np.set_printoptions(threshold=sys.maxsize)


# In[18]:


# In[19]:


def read_RGB(file):
    return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

def get_threshold(gray):
    (thresh, output) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh


#Ref  https://henrydangprg.com/2016/06/26/color-detection-in-python-with-opencv/
def detect_color(values, im, delta=10, rgb_select = False, min_sat=100, min_value=100):
    color = np.uint8([[[values[0], values[1], values[2]]]])
    flag = cv2.COLOR_BGR2HSV
    if rgb_select:
            flag = cv2.COLOR_RGB2HSV_FULL
    hsv_color = cv2.cvtColor(color, flag)
    hue = hsv_color[0][0][0]
    #print("Hue: ",hue)
    lower = np.array([(hue-delta), min_sat, min_value])
    upper = np.array([(hue+delta), 255, 255])
    return cv2.inRange(im, lower, upper)


def resize(img, height=170):
    """ Resize image to given height """
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))

#Detecting corner points by finding contours
#Ref (1): https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
#Ref (2): https://bretahajek.com/2017/01/scanning-documents-photos-opencv/
#Algorithm: Use edge detection, get an edged image, find contours, sort by area, approximate the contours 
#and if the approx curve is a quadrilateral then store then 4 points

#Returns points in order of: TOP_LEFT, TOP_RIGHT, BT_LEFT, BT_RIGHT
#Original: resize, bilateral_filter, adaptive_threshold, median_blur (B, D, E, F)
#Green: (A, C, D, E)
def find_contours(input_image):
 
    #Original edge detection
    gray_ = cv2.cvtColor(resize(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)), cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray_, 9, 75, 75)
    img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    img = cv2.medianBlur(img, 11)
    
    #Green threshold edge detection
    '''bgr_green = [35,46,32]
    mask_green = detect_color(bgr_green, cv2.cvtColor(cv2.medianBlur(input_image,95), cv2.COLOR_BGR2HSV), delta=35, min_sat=10, min_value=10)
    gray_ = cv2.medianBlur(resize(mask_green, height=200),11)
    blur = cv2.bilateralFilter(gray_, 9, 75, 75)
    img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)'''
    
    #FOLLOWING IS FOR REFERENCE ONLY:
    
    #A:
    #mask_green = detect_color(bgr_green, cv2.cvtColor(cv2.medianBlur(input_image,95), cv2.COLOR_BGR2HSV), delta=35, min_sat=10, min_value=10)
    #blur_mask = cv2.bilateralFilter(mask_green, 9,105, 105)
     
    #A
    #plt.imshow(mask_green, cmap="gray")
    #plt.show()
    
    #B:
    #gray_ = cv2.cvtColor(resize(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)), cv2.COLOR_BGR2GRAY)
    
    #C:
    #gray_ = cv2.medianBlur(resize(mask_green, height=200),11)
    
    #D:
    #blur = cv2.bilateralFilter(gray_, 9, 75, 75)
    
    #LOGS: 06/25/2019, old threshold valus were 115,4
    #E:
    #img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    
    #LOGS: 06/25/2019, old median value was 11
    #F:
    #img = cv2.medianBlur(img, 11)
    
    
    #img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0]) (Use only if object touches image border)
    edges = cv2.Canny(img, 200, 250)
    #edges = cv2.Canny()
    edge_restore = cv2.resize(edges, (input_image.shape[1], input_image.shape[0]))
    
    #plt.imshow(edge_restore, cmap = 'gray')
    #plt.show()
    #print(edge_restore.shape)
    
    contours = cv2.findContours(edge_restore.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    #Above function returns img, contours, hierarchy(?) => We pick only contours
    #contours is a list of numpy array of coordinates
    
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3] #Pick only the top 3 contours with max area
    for c in contours[0:]:
        
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06*perimeter, True)
        #TODO: increase the second argument of above function by 0.1% until you get 4 points (TODO)
        #06/26/19: Changed perimeter weight. old was 0.06

        if len(approx) == 4:
            
            
            #TODO: make function for ordering
            
            approx = approx.reshape(4,2)
            
            #Lex sort : http://www.codehamster.com/2015/03/16/sort-2d-array-in-python/
            ind = np.lexsort((approx[:, 1], approx[:, 0]))
            approx = approx[ind]
     
            #following idea inspired by Ref(2)
            
            points_ = [0,0,0,0]
            #1) When we apply lexsort, we get top points first and then bottom points 
            #2) In order to differenciate between topleft and topright, we do the following
            #3) top left has min difference and top right max max among the top two points
            top = approx[:2]
            topdiff = abs(np.diff(top))
            points_[0] = top[np.argmin(topdiff)]
            points_[1] = top[np.argmax(topdiff)]

            #4) Similarly for Btleft and Btright
            bottom = approx[2:]
            btdiff = abs(np.diff(bottom))
            points_[2] = bottom[np.argmax(btdiff)]
            points_[3] = bottom[np.argmin(btdiff)]
            return np.float32(points_), c



# In[31]:


def make_cb_grids(file_original, crop_height, crop_width, crop_path_original, ismarked=False, file_marked=None, crop_path_marked=None):
    
    tf_original = cv2.imread(file_original)
    if ismarked:
        tf_marked = cv2.imread(file_marked)
    thresh_first = get_threshold(cv2.cvtColor(tf_original, cv2.COLOR_BGR2GRAY))
    #print(file_original)


    cropped = tf_original[:crop_height,:crop_width]
    cropped.shape

    #Test
    #cv2.imwrite(data_path+'cropped_0630.JPG',cropped)

    #subprocess.run(["mkdir", data_path+file_original[:-4]])
    #subprocess.run(["mkdir", data_path+file_marked[:-4]])

    #FOR ACTUAL DATASET
    subprocess.run(["rm","-rf", crop_path_original[:-1]+"_new/"])
    subprocess.run(["rm","-rf", crop_path_original])
    subprocess.run(["mkdir", crop_path_original])
    
    if ismarked:
        subprocess.run(["rm","-rf", crop_path_marked[:-1]+"_new/"])
        subprocess.run(["rm","-rf", crop_path_marked])
        subprocess.run(["mkdir", crop_path_marked])

    offset_list = []
    offset_list_xy = []


    def make_grid(crop_path, infile_, image_, crop_dims, annotated=False):

    
        offset_list_xy = []
        #TODO: MISNOMER ALERT !!! change the name num_squares to grid_size. 
        offset = 0
        crop_height_ = crop_dims[0]
        crop_width_ = crop_dims[1]
        
        #TODO: change?? 
        n_height = image_.shape[0]//crop_height_
        n_width = image_.shape[1]//crop_width_
        
        #print("OFFSETS:", offset_list)
        ht = crop_height_//4
        wd = crop_width_//4
        off_ht = ht
        off_wd = wd
        offset_list_xy.append((0,0))
        for i in range(3):
            off_ht  =ht
            for j in range(3):
                off = (off_ht, off_wd)
                offset_list_xy.append(off)
                off_ht += ht
            off_wd += wd

        #print("OFFSET XY  ", offset_list_xy)                                                                    

        for o in offset_list_xy:
            grid_count = 0
            for i in range(n_height):
                for j in range(n_width):
                    #cropped = image_[i*num_squares+o[0]:(i+1)*num_squares+o[0], j*num_squares+o[1]:(j+1)*num_squares+o[1]]
                    cropped = image_[i*crop_height_+o[0]:(i+1)*crop_height_+o[0], j*crop_width_+o[1]:(j+1)*crop_width_+o[1]]
                    
                    if cropped.shape == (crop_height_, crop_width_, 3):
                        date_index = crop_path.find('/', len(data_parent))
                        date = crop_path[len(data_parent):date_index]
                        cv2.imwrite(crop_path+date+'_'+infile_[:-len(ext)]+'_offset'+str(o[0])+'_'+str(o[1])+'_'+str(grid_count)+"_size"+str(crop_height_)+ext, cropped)
                    grid_count += 1
        #f.close()

    #annotated_list = make_grid(crop_path_test, marked, image_an, (crop_height, crop_width), annotated=True)
    original_name = os.path.basename(file_original.replace("_cropped",""))
    make_grid(crop_path_original, original_name, tf_original, (crop_height, crop_width))
    if ismarked:
        marked_name = os.path.basename(file_marked.replace("_cropped",""))
        make_grid(crop_path_marked, marked_name, tf_marked, (crop_height, crop_width))
    #make_grid(crop_path_original, original_name, tf_original, (crop_height, crop_width))
    #make_grid(crop_path_checked, checked_name, tf_marked, (crop_height, crop_width))

    


# In[62]:


#name = prefix+'0_'+str(981)
#red = cv2.imread(name+ext)


# In[32]:


def remove_noise(img):
    img = 255 - img 
    kernel = np.ones((3,3),np.float32)
    kernel[1][1] = 0
    dst =cv2.filter2D(img,-1,kernel)
    dst = 255 - dst
    return dst


#CODE TO FILTER FULL SOLDERS
#Blob detection
#Reference : https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
def isMarked(im_):
    bgr_red = [36,28,237]
    red_mask = detect_color(bgr_red, cv2.cvtColor(im_, cv2.COLOR_BGR2HSV), delta=10, min_sat = 150, min_value=150)
    red_mask = remove_noise(red_mask)
    return red_mask, cv2.countNonZero(red_mask)

#https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
# #Enhance contrast of color image
# #Reference : https://chrisalbon.com/machine_learning/preprocessing_images/enhance_contrast_of_color_image/
# #Reference: https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_equalization/histogram_equalization.html

# In[35]:

if __name__ =="__main__":
    
    #if len(sys.argv) < 2:
    #    print("Preprocessing takes one command-line argument: <path/ to/ circuit/ boards>")
    #    exit()

    ext = '.bmp'

    #FOR ACTUAL DATASET
#    data_parent = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Enter path to circuit boards")
    parser = parser.parse_args()
    data_parent = parser.path
    print("parent path: ", data_parent)
    #Crop circuit boards - NG, OK makes no difference
    print("Cropping circuit boards...")
    cropped = set(glob.glob(data_parent+"*/??_cropped/")) #Already cropped
    cb_names  = glob.glob(data_parent+"*/??/")
    make_crops = set([x[:-1]+"_cropped/" for x in cb_names]) 

    to_crop = list(make_crops-cropped)
    to_crop_original = [t[:t.rindex("_cropped")]+"/" for t in to_crop]
    for t in range(len(to_crop_original)):
        circuits = glob.glob(to_crop_original[t]+"*"+ext)#get all circuit boards inside to_crop_original[t]
        subprocess.run(["mkdir", to_crop[t]])
        for c in circuits:
            cropped_name = to_crop[t]+os.path.basename(c)[:-len(ext)]+"_cropped"+ext
            image = cv2.imread(c)
            ret_val = find_contours(image)
            if not ret_val:
                print(cropped_name)
                continue
            src, _ = ret_val
            dst = np.float32([[0,0], [0, 1500], [3000, 0], [3000, 1500]])
            matrix = cv2.getPerspectiveTransform(src,dst)
            #Get transformed images for original and marked inputs
            transformed = cv2.warpPerspective(image, matrix, (3000,1500))
            cv2.imwrite(cropped_name, transformed)

    #Make grids 
    print("Making grids of size 384...")
    crop_size = 384
    OK_files = set(glob.glob(data_parent+"*/OK_cropped/??_*[0-9]_cropped"+ext))
    ok_files = set(glob.glob(data_parent+"*/ok_cropped/??_*[0-9]_cropped"+ext))
    ok_files = sorted(list(OK_files.union(ok_files)))
    #ok_dirs = sorted(glob.glob(data_parent+"????/OK_cropped/ok_*/"))
    for o in range(len(ok_files)):
        make_cb_grids(ok_files[o], crop_size, crop_size, ok_files[o][:-len(ext)]+"_"+str(crop_size)+"/")
        
    NG_files = set(glob.glob(data_parent+"*/NG_cropped/??_*[0-9]_cropped"+ext))
    ng_files = set(glob.glob(data_parent+"*/ng_cropped/??_*[0-9]_cropped"+ext))
    ng_files = sorted(list(NG_files.union(ng_files)))
    #ng_dirs = sorted(glob.glob(data_parent+"????/NG_cropped/NG_*[0-9]/"))
    NG_marked = set(glob.glob(data_parent+"*/NG_cropped/??_*[0-9]_checkmark_cropped"+ext))
    ng_marked = set(glob.glob(data_parent+"*/ng_cropped/??_*[0-9]_checkmark_cropped"+ext))
    ng_marked = sorted(list(NG_marked.union(ng_marked)))
    #ng_marked_dirs = sorted(glob.glob(data_parent+"????/NG_cropped/NG_*[0-9]_checkmark/"))
    for o in range(len(ng_files)):
        make_cb_grids(ng_files[o], crop_size, crop_size, ng_files[o][:-len(ext)]+"_"+str(crop_size)+"/", True, ng_marked[o], ng_marked[o][:-len(ext)]+"_"+str(crop_size)+"/")

    
    #Create csv file for image and label. Assign randomly for now to just test the data stream
    #Keep a global csv and store entire image path in that, later save all filtered images to a separate global folder
    print("Saving cropped image names in CSV...")
    file_path = data_parent+"labels-full-"+str(crop_size)+".csv"
    flabel = open(file_path, "w")
    writer = csv.writer(flabel)
    OK_files = set(glob.glob(data_parent+"*/OK_cropped/??_*[0-9]_cropped_"+str(crop_size)+"/*"+ext))
    ok_files = set(glob.glob(data_parent+"*/ok_cropped/??_*[0-9]_cropped_"+str(crop_size)+"/*"+ext))
    files_ok = sorted(list(OK_files.union(ok_files)))
    print("Cropped files: ", len(files_ok)) 
    #copy ok names
    for f in range(len(files_ok)):
        writer.writerow([files_ok[f], str(0)])


    NG_files = set(glob.glob(data_parent+"*/NG_cropped/??_*[0-9]_cropped_"+str(crop_size)+"/*"+ext))
    ng_files = set(glob.glob(data_parent+"*/ng_cropped/??_*[0-9]_cropped_"+str(crop_size)+"/*"+ext))
    files_ng = sorted(list(NG_files.union(ng_files)))

    MARK_files = set(glob.glob(data_parent+"*/NG_cropped/??_*[0-9]_checkmark_cropped_"+str(crop_size)+"/*"+ext))
    mark_files = set(glob.glob(data_parent+"*/ng_cropped/??_*[0-9]_checkmark_cropped_"+str(crop_size)+"/*"+ext))
    files_marked = sorted(list(MARK_files.union(mark_files)))

    # files_ok = sorted(glob.glob(data_parent+"????/OK_cropped/ok_*[0-9]_cropped_"+str(crop_size)+"/*.bmp"))

    # files_ng = sorted(glob.glob(data_parent+"????/NG_cropped/NG_*[0-9]_cropped_"+str(crop_size)+"/*.bmp"))
    # files_marked = sorted(glob.glob(data_parent+"????/NG_cropped/NG_*[0-9]_checkmark_cropped_"+str(crop_size)+"/*.bmp"))

    #copy ng names
    for f in range(len(files_ng)):
        img = cv2.imread(files_marked[f])
        if isMarked(img)[1]:
            writer.writerow([files_ng[f], str(1)])
        else:
            writer.writerow([files_ng[f], str(0)])
    flabel.close()
    print("Preprocessing complete. CSV file for dataset found at: " + file_path)




