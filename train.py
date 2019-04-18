# -*- coding: utf-8 -*-
# @Time    : 2019/3/23 13:03
# @Author  : XuKun
# **************************
import numpy as np
import pandas as pd
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from sklearn.model_selection import train_test_split


times=1
epochs =50
batch_size = 8
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_trained_model.h5'


def robust_data(data,times=1,mean=0,deviation=1,max=255,min=0):
    li=[]
    for i in range(times):
        data1=data+np.random.normal(mean,deviation,data.shape)
        li.append(data1)
    data=np.vstack(li)
    data[data>max]=max
    data[data<min]=min
    return data

data=np.zeros(shape=(305,312,416,3),dtype=np.uint8)
for i in range(0,176):
    imgname='train_mini/'+str(i)+'.jpg'
    img=cv2.imread(imgname)
    img=np.array(img)
    data[i]=img
data=robust_data(data,times=times,mean=0,deviation=40)
data=np.array(data,dtype=np.uint8)
# try:
#     cv2.imshow('a',data[100])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imshow('a',data[405])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# except:
#     pass
labels=pd.read_csv('labels_ordered.csv')
labels=pd.get_dummies(labels,columns=['label'])
labels=labels.drop(['number'],axis=1)
labels=np.vstack([labels]*times)
print(labels.shape,data.shape)

x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size = 0.1, random_state = 0)

# create the model
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=x_train.shape[1:],
                 kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))
          )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same',
                 kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))
          )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same',
                 kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))
          ) # (None, 312, 416, 32)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same',
                 kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))
          )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same',
                 kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))
          )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same',
                 kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))
          )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(1024,kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
model.add(Activation('relu'))
model.add(Dense(1024,kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
model.add(Activation('relu'))
model.add(Dense(512,kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(8,kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
          bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1)))

print(model.summary())

# initiate RMSprop optimizer
opt = keras.optimizers.Adam()

# Let's train the model using RMSprop
model.compile(loss='mse',
              optimizer=opt,
              metrics=['mse'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# check whether model exists
if os.path.exists('saved_models'):
    print('loading weight')
    model.load_weights('saved_models/keras_trained_model.h5')
    print('loading over........')
else :
    pass

history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              verbose=True)
hist = pd.DataFrame(history.history)
hist.to_csv('history.csv',index=False)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
