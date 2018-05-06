# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 22:24:35 2018

@author: Karra's
"""

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

test_dir='D:/hackerearth/myntra_final/Test_offline'


submission_offline = pd.read_csv('D:/hackerearth/myntra_final/Submission_offline.csv')


missing_ids_test=[]

start = time.time()
for i in range(submission_offline.shape[0]):
    link = submission_offline.iloc[i]['Link_to_the_image']
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