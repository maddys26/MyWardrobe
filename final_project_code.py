import cv2
import os
import csv
import glob
import numpy as np
import pandas as pd
import mahotas
import warnings
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor


#**************************************************************************************************************
#                                            FEATURE EXTRACTION
#**************************************************************************************************************

#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------

fixed_size       = tuple((500, 500))
train_path       = r'D:\SEM 4\PYTHON\innovative\dataset\final'
csv_data         = 'D:\SEM 4\PYTHON\innovative\output\data.csv'
csv_labels       = 'D:\SEM 4\PYTHON\innovative\output\labels.csv'
bins             = 8
train_labels = os.listdir(train_path)
train_labels.sort()
print(train_labels)
images_per_class = []
for category in train_labels:
    path = os.path.join(train_path,category)
    imgs = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')] #list of filenames for all jpg images in a directory.
    images_per_class.append(len(imgs))
print(images_per_class)

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# local feature-descriptor-4: SIFT
#def fd_sift(image):
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
#    return (keypoints, descriptors)


# feature-descriptor-5: Other:LBP,GLCM,Gabor filter
def fd_other_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # LBP
    feat_lbp = local_binary_pattern(image,5,2,'uniform')
    lbp_hist,_ = np.histogram(feat_lbp,8)
    lbp_hist = np.array(lbp_hist,dtype=float)
    lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
    lbp_energy = np.nansum(lbp_prob**2)
    lbp_entropy = -np.nansum(np.multiply(lbp_prob,np.log2(lbp_prob)))   
    # GLCM
    gCoMat = greycomatrix(image, [2], [0],256,symmetric=True, normed=True)
    contrast = greycoprops(gCoMat, prop='contrast')
    dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
    homogeneity = greycoprops(gCoMat, prop='homogeneity')    
    energy = greycoprops(gCoMat, prop='energy')
    correlation = greycoprops(gCoMat, prop='correlation')    
    feat_glcm = np.array([contrast[0][0],dissimilarity[0][0],homogeneity[0][0],energy[0][0],correlation[0][0]])
    # Gabor filter
    gaborFilt_real,gaborFilt_imag = gabor(image,frequency=0.6)
    gaborFilt = (gaborFilt_real**2+gaborFilt_imag**2)//2
    gabor_hist,_ = np.histogram(gaborFilt,8)
    gabor_hist = np.array(gabor_hist,dtype=float)
    gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
    gabor_energy = np.nansum(gabor_prob**2)
    gabor_entropy = -np.nansum(np.multiply(gabor_prob,np.log2(gabor_prob)))
    # Concatenating features(2+5+2)    
    concat_feat = np.concatenate(([lbp_energy,lbp_entropy],feat_glcm,[gabor_energy,gabor_entropy]),axis=0)
    return concat_feat


# empty lists to hold feature vectors and labels
global_features = []
labels          = []

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)
    imgs = [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith('.jpg')] 
    # get the current training label
    current_label = training_name

    #for x in range(1,images_per_class[imgpcl]+1):
    # loop over the images in each sub-folder
    #for imgct in images_per_class:
    
    for image in imgs:
            # read the image and resize it to a fixed-size
        # read the image and resize it to a fixed-size
        image = cv2.imread(image)
        image = cv2.resize(image, fixed_size)

            ####################################
            # Global Feature extraction
            ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        #fv_kp,fp_desc = fd_sift(image)
        fv_other      = fd_other_features(image)
        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments, fv_other])
                    # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")



import numpy as np
# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# scale features in the range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("rescaled_features type: {}".format(type(rescaled_features)))
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))
#print("target type: {}".format(type(target)))

print("rescaled_features type: {}".format(np.array(rescaled_features).dtype))
#print("target type: {}".format(np.array(target).dtype))

# save the feature vector using .csv file
with open(csv_data, 'w',newline='') as file:
    writer = csv.writer(file,delimiter = ',')
    writer.writerows(np.array(rescaled_features))
                      
with open(csv_labels, 'w',newline='') as file:
    writer = csv.writer(file,delimiter = ',')
    writer.writerows(map(lambda x: [x], target))

print("[STATUS] end of training..")


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#********************************************************************************
#                                   TEST-TRAIN
#********************************************************************************
num_trees = 100
test_size = 0.10
seed      = 9
warnings.filterwarnings('ignore')
scoring    = "accuracy"
test_path  = r'D:\SEM 4\PYTHON\innovative\dataset\test final all'
if not os.path.exists(test_path):
    os.makedirs(test_path)

# import the feature vector and trained labels

df=pd.read_csv(csv_data, sep=',',header=None)
global_features = df.values
df=pd.read_csv(csv_labels, sep=',',header=None)
global_labels = df.values

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")


(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                     np.array(global_labels),
                                                                               test_size=test_size,
                                                                                          random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))


#***************************************************************************************************************
#                                          TESTING OUR MODEL
#***************************************************************************************************************

# to visualize results
import matplotlib.pyplot as plt
import cv2

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
print('clf',clf)
# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)
print('test_path',test_path)
# loop through the test images
imgs = [os.path.join(test_path,f) for f in os.listdir(test_path) if f.endswith('.jpg')] 
    
for image in imgs:
        # read the image and resize it to a fixed-size
    image = cv2.imread(image)
        #print(image)
    image = cv2.resize(image, fixed_size)
    image = np.array(image, dtype=np.uint8)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    #fv_kp,fp_desc = fd_sift(image)
    fv_other      = fd_other_features(image)
    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments, fv_other])

    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform(global_feature.reshape(-1,1))

    # predict label of test image
    prediction = clf.predict(rescaled_feature.reshape(1,-1))[0]

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    
