import tensorflow as tf
from tensorflow.keras.utils import to_categorical #for categorizing into arrays with 0 and 1 values
from tensorflow.keras.datasets import mnist #for dataset
import numpy as np #for array manipulation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(xtr,ytr),(xt,yt) = mnist.load_data()
#xtrain ytrain xtest ytest

ytre = to_categorical(ytr)
yte = to_categorical(yt)
#to convert labels into arrays of 0s and 1s where 1 corresponds
#to the correct guess

xtrre = np.reshape(xtr,(60000,784)) # no of images,no of pixels
xtre = np.reshape(xt,(10000,784)) # same, no of imgs and 28 * 28 pixels

#Normalization:
xmean = np.mean(xtrre)
xstd = np.std(xtrre)

epsilon = 1e-10
# added epsilon to counteract an error that may occur sometimes
'''did not calculate differently for test sets and training 
set as that might've created an unfair bias'''
xtrn = (xtrre - xmean)/(xstd +  epsilon)
xtn = (xtre - xmean)/(xstd + epsilon)

model = Sequential([Dense(128,activation='relu',input_shape=(784,)),
                   Dense(128,activation = 'relu'),
                   Dense(10,activation = 'softmax')])
model.compile(optimizer = 'sgd',
              loss = 'categorical_crossentropy',
             metrics =['accuracy'])
model.summary()

model.fit(xtrn,ytre,epochs = 5)

loss,accuracy = model.evaluate(xtn,yte)

preds = model.predict(xtn)
print(preds.shape)

import matplotlib.pyplot as plt
plt.figure(figsize = (12,12))

start_index = 0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index + i])
    gt = yt[start_index + i]
    
    col = 'g'
    
    if pred != gt:
        col = 'r'
    plt.xlabel('i={} , pred = {} , gt = {}'.format(start_index+i,pred,gt),color = col)
    plt.imshow(xt[start_index+i],cmap = 'binary')
plt.show()  
