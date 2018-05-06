# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 23:21:25 2018

@author: Karra's
"""

import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation,GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
from collections import Counter
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

img_rows, img_cols, img_channel = 224, 224, 3
num_classes=25

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))

for layer in base_model.layers:
    layer.trainable=False

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(num_classes, activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()


train_dir='D:/hackerearth/delivery/Train_new/'
generator = ImageDataGenerator(rescale=1. / 255)
val_dir='D:/hackerearth/delivery/Test_new/'
train_iterator = generator.flow_from_directory(train_dir,batch_size=32,target_size=(224,224))
validation_iterator = generator.flow_from_directory(val_dir,batch_size=32,target_size=(224,224))

log_dir = 'D:/hackerearth/delivery/tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb] 

training_steps = (len(train_iterator.filenames)//32)+1
val_steps =(len(validation_iterator.filenames)//32)+1

best_weights_filepath = 'D:/hackerearth/best_weights_vgg.hdf5'
earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
model.fit_generator(train_iterator,
                    steps_per_epoch = training_steps,
                    epochs = 10,
                    validation_data = validation_iterator,
                    validation_steps = val_steps,
                    workers=4,
                   callbacks=[earlyStopping, saveBestModel])

test_dir='D:/hackerearth/delivery/Test/'

test_generator=generator.flow_from_directory(test_dir,batch_size=1,target_size=(224,224))
test_steps = len(test_generator.filenames)
s=model.predict_generator(
    test_generator,steps=test_steps
    )
top_model_weights_path = 'bottleneck_fc_model.h5'
model.save_weights(top_model_weights_path)




