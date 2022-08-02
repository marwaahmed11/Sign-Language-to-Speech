import numpy as np 
import pandas as pd 
import tensorflow
import matplotlib.pyplot as plt
import glob
from sklearn import svm
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
path_train='asl_alphabet_train'
train = list(glob.glob(path_train+'/*/*.jpg'))

def creatClf(train,color,clf):
    labels = list(map(lambda x:os.path.split(os.path.split(x)[0])[1], train))
    numeric_labels=[]
    for l in labels:
        if l =='del':
            numeric_labels.append(26)
        elif l=='nothing':
            numeric_labels.append(27)
        elif l=='space':
            numeric_labels.append(28)
        else:
            numeric_labels.append(ord(l)-ord('A'))
    train = pd.Series(train, name='FilePath').astype(str)
    labels = pd.Series(labels, name='Label')
    data = pd.concat([train, labels], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    Xtrain=[]
    Ytrain=[]
    Xtest=[]
    Ytest=[]
    
    print(len(train))
    for t in range(int(len(train)*.8)) :
        img=cv2.imread(data['FilePath'][t])
        img = cv2.cvtColor(img,color)
        img=cv2.resize(img,(100,100))
        img= cv2.GaussianBlur(img, (3, 3), 0)
        img=cv2.Canny(image=img, threshold1=3, threshold2=100)
        Xtrain.append(img)
        Ytrain.append(data['Label'][t])
    for t in range(int(len(train)*.8),int(len(train))):
        img=cv2.imread(data['FilePath'][t])
        img = cv2.cvtColor(img,color)
        img=cv2.resize(img,(100,100))
        img= cv2.GaussianBlur(img, (3, 3), 0)
        img=cv2.Canny(image=img, threshold1=3, threshold2=100)
        Xtest.append(img)
        Ytest.append(data['Label'][t])
        
    flatten_Xtrain= np.array(Xtrain).reshape(len(Xtrain),-1)
    flatten_Xtest= np.array(Xtest).reshape(len(Xtest),-1)
    clf.fit(flatten_Xtrain,Ytrain)
    Ypredict=clf.predict(flatten_Xtest)
    print("Accuracy:",accuracy_score(Ytest,Ypredict))
    print("Precision:",precision_score(Ytest,Ypredict,average=None))
    print("Recall:",recall_score(Ytest,Ypredict,average=None))


print("The color:Gray")
creatClf(train,cv2.COLOR_BGR2RGB,svm.SVC())
print("#############################################################################")
print("The color:BRG")
creatClf(train,cv2.COLOR_BGR2GRAY,svm.SVC())





