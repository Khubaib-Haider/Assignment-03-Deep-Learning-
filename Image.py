import numpy as np
import cv2
import os.path

Training_Data = 'F:\Masters RIME\Third Semester\Deep learning\Assignemnts\Test_Images'
Test_Data = 'F:\Masters RIME\Third Semester\Deep learning\Assignments\Train_Images'
Output = ["Balls", "Football"]

def pre_processing(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pred = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    
    img_pred = np.asarray(img_pred)
    
    img_pred = img_pred / 255
    return img_pred

def train(path):
    X = []
    for file in os.listdir(path):
            if (os.path.isfile(path + "/" + file)):
                image = pre_processing(path + "/" + file)
                image = np.reshape(image,(image.shape[0]*image.shape[1]))
                X.append(image)
    
    X_train = np.array(X)
    y_duck =  np.zeros((10,1))
    y_horse = np.ones((10,1))
    Y_train = np.concatenate((y_duck,y_horse))
    
    return X_train,Y_train

def test(path):
    X = []
    for file in os.listdir(path):
            if (os.path.isfile(path + "/" + file)):
                image = pre_processing(path + "/" + file)
                image = np.reshape(image,(image.shape[0]*image.shape[1]))
                X.append(image)
    
    X_test = np.array(X)
    y_duck =  np.zeros((5,1))
    y_horse = np.ones((5,1))
    Y_test = np.concatenate((y_duck,y_horse))
    return X_test,Y_test

X_train,Y_train = train(Dataset_path)
print(X_train.shape)
print(Y_train.shape)
            