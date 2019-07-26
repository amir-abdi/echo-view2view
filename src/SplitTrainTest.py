import os
from glob import glob
import random
import numpy as np
import shutil
import pickle
train_ratio = 0.8

mode = 'load'


# dataset_path='/media/mohammadj/53f67744-2e22-44e6-99e3-fa5719d1486d1/Unify_Data/Public_Camus'
# dataset_path = '/mnt/rcl-DGX/mohammadj/MyData/ViewViewConvert/Data'
dataset_path='../Data/'

train_path = os.path.join(dataset_path, 'training')
test_path = os.path.join(dataset_path, 'testing')

if mode == 'save':
    list = glob(os.path.join(dataset_path, 'training', '*'))
    random.shuffle(list)
    num_data = len(list)
    train_number = np.int(train_ratio * num_data)
    train_patients = list[0:train_number]
    test_patients = list[train_number:]

    for patient in test_patients:
        shutil.move(patient, test_path)

    with open('train_list','wb') as fb:
        pickle.dump(train_patients,fb)


    with open('test_list','wb') as fb:
        pickle.dump(test_patients,fb)

if mode == 'load':

    with open('train_list', 'rb') as fp:
        train_patients = pickle.load(fp)
    with open('test_list', 'rb') as fp:
        test_patients = pickle.load(fp)

    for patient in test_patients:
        patient_ID = os.path.basename(patient)
        shutil.move(os.path.join(train_path,patient_ID), test_path)

