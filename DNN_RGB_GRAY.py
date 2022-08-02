import numpy as np 
import pandas as pd 
import tensorflow
import matplotlib.pyplot as plt
import glob


from sklearn import svm
import cv2
from keras import layers, models
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import keras, os


path_train='asl_alphabet_train'
train = list(glob.glob(path_train+'/*/*.jpg'))
def creatClf(train,color,dim):
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
            numeric_labels.append(ord(l.upper())-ord('A'))
    train = pd.Series(train, name='FilePath').astype(str)
    labels = pd.Series(numeric_labels, name='Label')
    data = pd.concat([train, labels], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    Xtrain=[]
    Ytrain=[]
    Xtest=[]
    Ytest=[]
    print(len(train))

    for t in range(int(len(train)*.7)) :
        img=cv2.imread(data['FilePath'][t])
        img = cv2.cvtColor(img,color)
        img=cv2.resize(img,(100,100))
        img=img/255-0.5
        Xtrain.append(img)
        Ytrain.append(data['Label'][t])
    for t in range(int(len(train)*.7),len(train)):
        img=cv2.imread(data['FilePath'][t])
        img = cv2.cvtColor(img,color)
        img=cv2.resize(img,(100,100))
        img=img/255-0.5
        Xtest.append(img)
        Ytest.append(data['Label'][t])
    
    
    if(dim==1):
        Xtrain=np.array(Xtrain)
        Xtest=np.array(Xtest)
        Xtrain=np.expand_dims(Xtrain, axis=3)
        Xtest=np.expand_dims(Xtest, axis=3)
        model = models.Sequential()
        model.add(layers.Conv2D(input_shape=(100,100,dim),filters=50, kernel_size=(3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(100, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(100, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(29,activation='softmax'))
    elif(dim==3):
        Xtrain=np.array(Xtrain)
        Xtest=np.array(Xtest)
        model = models.Sequential()
        model.add(layers.Conv2D(100, (3, 3), activation='relu', input_shape=(100, 100, 3)))
        model.add(layers.Conv2D(50, (3, 3), activation='relu', input_shape=(100, 100, 3)))
        model.add(layers.MaxPooling2D((3,3 )))
        model.add(layers.Conv2D(100, (3, 3), activation='relu'))
        model.add(layers.Conv2D(120, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(75, (3, 3), activation='relu'))
        model.add(layers.Conv2D(50, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Flatten())
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(29,activation='softmax'))
        
    model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    model.summary()

    history=model.fit(
      Xtrain,
      to_categorical(Ytrain),
      epochs=10,
    )

    model.save("DNN.h5", include_optimizer=True)
    Ypred=model.predict(Xtest)
    ypred=[]
    for i in Ypred:
        for c in range(len(i)):
            if i[c]==max(i):
                ypred.append(c)
    
    print("Accuracy:",accuracy_score(Ytest,ypred))
    print("Precision:",precision_score(Ytest,ypred,average=None))
    print("Recall:",recall_score(Ytest,ypred,average=None))
    

print("The color:Gray")
creatClf(train,cv2.COLOR_BGR2GRAY,1)
print("#############################################################################")
print("The color:BRG")
creatClf(train,cv2.COLOR_BGR2RGB,3)

