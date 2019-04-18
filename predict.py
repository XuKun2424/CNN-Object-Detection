# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 22:30
# @Author  : XuKun
# **************************
import numpy as np
import pandas as pd
import cv2
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

shape=(312, 416, 3)

def decisioin_label(li=[]):
    li=list(li)
    label=''
    a=li.index(max(li))
    if a==0:
        label='up'
    elif a==1:
        label='left'
    elif a==2:
        label='down'
    elif a==3:
        label='right'
    return label
def toAbsolute(cx,cy,w,h,width,height):
    x1=(cx-w/2)*width
    x2=(cx+w/2)*width
    y1=(cy-h/2)*height
    y2=(cy+h/2)*height
    return int(x1),int(y1),int(x2),int(y2)

# create the model
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=shape,
                 kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1))
          )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # (None, 156, 208, 32)

model.add(Conv2D(128, (3, 3), padding='same',
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

model.add(Conv2D(256, (3, 3), padding='same',
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

# check whether model exists
if os.path.exists('saved_models'):
    print('loading weight')
    model.load_weights('saved_models/keras_trained_model.h5')
    print('loading over........')
else :
    pass

data=np.zeros(shape=(1,312,416,3),dtype=np.uint8)
# for i in range(0,305):
#     imgname='train_mini/'+str(i)+'.jpg'
#     img=cv2.imread(imgname)
#     img=np.array(img)
#     data[i]=img
# data=data/255
# labels=pd.read_csv('labels_ordered.csv')
# print(model.predict(data[:5]))
# print(labels[:5])
print("************作者：徐昆\n"
      "************领域：目标检测\n"
      "************使用方法：输入一个数字（代表照片的编号）后按下回车，程序会识别出目标的位置以及方向，输入小写字母s停止运行。\n"
      "************注意：exe程序必须和data文件夹，saved_models文件夹在同一级目录。")
while True:
    imagename=input('input the image number:')
    if imagename=='s':
        break
    try:
        path = 'train_mini/'+imagename+'.jpg'
        img = cv2.imread(path)
        data[0] = img
        # print(img.shape)
        # img = cv2.resize(img, dsize=(shape[1], shape[0]))
        # cv2.imshow('original', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    except:
        print("请输入正确的编号！")
        continue

    data=data/255
    p=model.predict(data)
    label=decisioin_label(p[0,4:8])
    p[p<0]=0
    x1,y1,x2,y2=toAbsolute(p[0,0],p[0,1],p[0,2],p[0,3],shape[1],shape[0])
    data=data*255
    data=np.array(data,dtype=np.uint8)
    cv2.rectangle(data[0], (x1, y1), (x2, y2), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(data[0], label, (x1, y1-5), font,0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('drawing', data[0])
    cv2.waitKey(4000)
    cv2.destroyAllWindows()
