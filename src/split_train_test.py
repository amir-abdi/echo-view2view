import os
from glob import glob
import random
import numpy as np
import shutil
import pickle
from absl import flags
from absl import app
########################## THIS CODE EXTRACTS A PART OF DATA FROM CAMUS TRAINING DATASET TO BE USED AS TEST  ########################
########################## THIS IS NEEDED SINCE TEST DATA IN CAMUS DATASET DO NOT HAVE SEGMENTATION GROUND TRUTH ####################
flags.DEFINE_string('dataset_path', None, 'Path of the dataset.')
flags.DEFINE_string('mode', 'load', 'load: move test data using the test_list, save: create a new random train test list')
flags.mark_flag_as_required('dataset_path')
flags.mark_flag_as_required('mode')

FLAGS = flags.FLAGS
train_ratio = 0.8

def main(argv):
    dataset_path = FLAGS.dataset_path
    mode = FLAGS.mode

    train_path = os.path.join(dataset_path, 'training')
    test_path = os.path.join(dataset_path, 'testing_with_seg')
    if not os.path.exists(test_path):
        os.makedirs(test_path)

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
        # moves selected data from training to a test folder
        # needed since current test data in camus do not have segmentation ground truth

        # with open('train_list', 'rb') as fp:
        #     train_patients = pickle.load(fp)
        with open('test_list', 'rb') as fp:
            test_patients = pickle.load(fp)
        cnt = 0
        for patient in test_patients:
            patient_ID = os.path.basename(patient)
            patient_folder = os.path.join(train_path,patient_ID)
            if os.path.exists(patient_folder):
                shutil.move(patient_folder, test_path)
                cnt = cnt + 1
        print('Number of patients moved to the test folder: ', cnt)

if __name__ == '__main__':
    app.run(main)