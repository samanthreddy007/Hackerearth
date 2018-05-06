
import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation,GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
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


num_classes=23

model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(256 ,256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()
train_dir='D:/hackerearth/myntra_final/Train_new/'
generator = ImageDataGenerator(rescale=1. / 255)
val_dir='D:/hackerearth/myntra_final/Test_new/'
train_iterator = generator.flow_from_directory(train_dir,batch_size=32,target_size=(256,256))
validation_iterator = generator.flow_from_directory(val_dir,batch_size=32,target_size=(256,256))

log_dir = 'D:/hackerearth/myntra_final/tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb] 
best_weights_filepath = 'D:/hackerearth/myntra_final/tf-log/best_weights.hdf5'
earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

training_steps = (len(train_iterator.filenames)//32)+1
val_steps =(len(validation_iterator.filenames)//32)+1
model.fit_generator(train_iterator,
                    steps_per_epoch = training_steps,
                    epochs = 10,
                    validation_data = validation_iterator,
                    validation_steps = val_steps,
                    callbacks=[earlyStopping, saveBestModel])

top_model_weights_path = 'D:/hackerearth/myntra_final/bottleneck_fc_model.h5'
model.save_weights(top_model_weights_path)

test_dir='D:/hackerearth/myntra_final/Test/'

test_generator=generator.flow_from_directory(test_dir,batch_size=1,shuffle=False,target_size=(256,256))
test_steps = len(test_generator.filenames)
test_files_names = test_generator.filenames
s=model.predict_generator(
    test_generator,steps=test_steps
    )
u=np.argmax(s,axis=1)
test_files_names = test_generator.filenames
import pandas as pd

predictions_df = pd.DataFrame(u, columns = ['Submission_category'])
predictions_df.insert(0, "name", test_files_names)

predictions_df['name'] = predictions_df['name'].map(lambda x: x.rstrip('.jpg'))
predictions_df['name'] = predictions_df['name'].map(lambda x: x.lstrip('Test_offline\\'))
predictions_df['name'] = pd.to_numeric(predictions_df['name'], errors = 'coerce')
predictions_df.sort_values('name', inplace = True)
Submission_online=pd.read_csv('D:/hackerearth/myntra_final/Submission_offline.csv')
Submission_online['name']=np.arange(15000)
Submission_online.drop(['Sub_category'],axis=1,inplace=True)
Submission_online=Submission_online.merge(predictions_df[['name','Submission_category']],how='left',on='name')
Submission_online.drop(['name'],axis=1,inplace=True)
Submission_online.rename(columns={'Submission_category':'Sub_category'},inplace=True)
#Submission_online['Sub_category']=predictions_df['invasive'] 
inv_map = {v: k for k, v in train_iterator.class_indices.items()}
print(inv_map)
classes = np.argmax(s, axis=1)

probes = np.max(s, axis=1)

unique_elements, counts_elements = np.unique(classes, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
Submission_online['Sub_category']=Submission_online['Sub_category'].map(inv_map)
Submission_online['Sub_category']=Submission_online['Sub_category'].fillna(Submission_online['Sub_category'].mode()[0])
Submission_online.to_csv('D:/hackerearth/myntra_final/predictions_df.csv', index = False)