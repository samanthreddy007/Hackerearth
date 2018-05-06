# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 20:25:54 2018

@author: Karra's
"""

import os
from IPython.display import clear_output
import pandas as pd  # For manipulating CSV files
import urllib.request  # For downloading files from the provided links
import time
from termcolor import colored
import requests
import glob
import requests                 # downloading images
import pandas as pd             # csv- / data-input
from scipy.misc import imread   # image-decoding -> numpy-array
import matplotlib.pyplot as plt # only for demo / plotting
import scipy.misc

test_dir='D:/hackerearth/myntra_final/Test'
train_dir='D:/hackerearth/myntra_final/Train'

traincsv = pd.read_csv('D:/hackerearth/myntra_final/myntra_train_dataset.csv')
testcsv = pd.read_csv('D:/hackerearth/myntra_final/myntra_test.csv')

missing_ids_train=[]
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
start = time.time()
for i in range(traincsv.shape[0]):
    link = traincsv.iloc[i]['Link_to_the_image']
    name = (traincsv.iloc[i]['Sub_category'])
    full_name = name+'_'+str(i)+'.jpg'
    img_name = full_name
    full_name = os.path.join(train_dir, img_name)
    
    full_name=full_name.replace('\\', '/')
    if not os.path.exists(full_name):
        try:
            clear_output(wait=True)
            r = requests.get(link, stream=True,verify=False)
            imgs=(imread(r.raw))
            #urllib.request.urlretrieve(link, full_name)
            print(colored(img_name+' downloaded', 'green'))
            scipy.misc.imsave(full_name, imgs)
        except:
            clear_output(wait=True)
            missing_ids_train.append(i)
            print(colored('Link Missing', color='red'))
    else:
        clear_output(wait=True)
        print(img_name,' has already been downloaded')
end = time.time()
print('Time taken: ', end-start)
import shutil
for filename in os.listdir(train_dir):
    filename.split('_')
    f=filename.split('_')[0]
    if not os.path.exists('D:/hackerearth/myntra_final/Train/'+f):
        os.mkdir('D:/hackerearth/myntra_final/Train/'+f)
    src=os.path.join('D:/hackerearth/myntra_final/Train/', filename) 
        
    dst = os.path.join('D:/hackerearth/myntra_final/Train/'+f+'/', filename) 
    shutil.move(src, dst)


""" Downloading the Testing data """
missing_ids_test=[]
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
start = time.time()
for i in range(testcsv.shape[0]):
    link = testcsv.iloc[i]['Link_to_the_image']
    name = str(i)+'.jpg'
    full_name = os.path.join(test_dir, name)
    full_name=full_name.replace('\\', '/')

    if not os.path.exists(full_name):
        try:
            clear_output(wait=True)
            r = requests.get(link, stream=True,verify=False)
            imgs=(imread(r.raw))
            #urllib.request.urlretrieve(link, full_name)
            print(colored(name+' downloaded', 'green'))
            scipy.misc.imsave(full_name, imgs)
            
        except:
            missing_ids_test.append(i)
            clear_output(wait=True)
            print(colored('Link Missing', color='red'))
    else:
        clear_output(wait=True)
        print(name, ' has already been downloaded')
end = time.time()
print('Time taken: ', end-start)