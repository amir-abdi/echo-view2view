from glob import glob
import numpy as np
import os
import skimage.io as io
from PIL import Image
from prefetch_generator import background
from keras.preprocessing.image import ImageDataGenerator
import random
import cv2

from utils import get_LV_lenght
from utils import match_image_size

NUM_PREFETCH = 10
RANDOM_SEED = 7


class DataLoaderCamus:
    def __init__(self, dataset_path, input_name, target_name, img_res, target_rescale,
                 input_rescale, train_ratio, valid_ratio, labels, augment, equalize_lv_length):
        self.dataset_path = dataset_path
        self.img_res = tuple(img_res)
        self.target_rescale = target_rescale
        self.input_rescale = input_rescale
        self.input_name = input_name
        self.target_name = target_name
        self.augment = augment
        self.equalize_lv_length = equalize_lv_length

        patients = sorted(glob(os.path.join(self.dataset_path, 'training', '*')))
        random.Random(RANDOM_SEED).shuffle(patients)
        num = len(patients)
        num_train = int(num * train_ratio)
        valid_num = int(num_train * valid_ratio)

        self.valid_patients = patients[:valid_num]
        self.train_patients = patients[valid_num:num_train]
        self.test_patients = patients[num_train:]

        print('#train:', len(self.train_patients))
        print('#valid:', len(self.valid_patients))
        print('#test:', len(self.test_patients))
        print('Consistency check - First valid sample:', self.valid_patients[0])
        print('Consistency check - First test sample:', self.test_patients[0])

        all_labels = {0, 1, 2, 3}
        self.not_labels = all_labels - set(labels)

        data_gen_args = dict(rotation_range=augment['AUG_ROTATION_RANGE_DEGREES'],
                             width_shift_range=augment['AUG_WIDTH_SHIFT_RANGE_RATIO'],
                             height_shift_range=augment['AUG_HEIGHT_SHIFT_RANGE_RATIO'],
                             shear_range=augment['AUG_SHEAR_RANGE_ANGLE'],
                             zoom_range=augment['AUG_ZOOM_RANGE_RATIO'],
                             fill_mode='constant',
                             cval=0.,
                             data_format='channels_last')
        self.datagen = ImageDataGenerator(**data_gen_args)

    def read_mhd(self, img_path, is_gt):
        if not os.path.exists(img_path):
            return np.zeros(self.img_res + (1,))
        img = io.imread(img_path, plugin='simpleitk').squeeze()
        img = np.array(Image.fromarray(img).resize(self.img_res))
        img = np.expand_dims(img, axis=2)

        if is_gt:
            for not_l in self.not_labels:
                img[img == not_l] = 0
        return img

    def _get_paths(self, stage):
        if stage == 'train':
            return self.train_patients
        elif stage == 'valid':
            return self.valid_patients
        elif stage == 'test':
            return self.test_patients

    @background(max_prefetch=NUM_PREFETCH)
    def get_random_batch(self, batch_size=1, stage='train'):
        paths = self._get_paths(stage)

        num = len(paths)
        num_batches = num // batch_size

        for i in range(num_batches):
            batch_paths = np.random.choice(paths, size=batch_size)
            target_imgs, target_imgs_gt, input_imgs, input_imgs_gt = self._get_batch(batch_paths, stage)
            target_imgs = target_imgs * self.target_rescale
            input_imgs = input_imgs * self.input_rescale

            yield target_imgs, target_imgs_gt, input_imgs, input_imgs_gt

    def get_iterative_batch(self, batch_size=1, stage='test'):
        paths = self._get_paths(stage)

        num = len(paths)
        num_batches = num // batch_size

        start_idx = 0
        for i in range(num_batches):
            batch_paths = paths[start_idx:start_idx + batch_size]
            target_imgs, target_imgs_gt, input_imgs, input_imgs_gt, = self._get_batch(batch_paths, stage)
            target_imgs = target_imgs * self.target_rescale
            input_imgs = input_imgs * self.input_rescale
            start_idx += batch_size

            yield target_imgs, target_imgs_gt, input_imgs, input_imgs_gt

    def _get_batch(self, paths_batch, stage):
        target_imgs = []
        source_imgs = []
        target_imgs_gt = []
        source_gt_imgs = []
        for path in paths_batch:
            transform = self.datagen.get_random_transform(img_shape=self.img_res)
            head, patient_id = os.path.split(path)
            target_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.target_name))
            target_gt_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.target_name + '_gt'))
            source_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.input_name))
            source_gt_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.input_name + '_gt'))

            # get source
            source_img = self.read_mhd(source_path, '_gt' in self.input_name)
            source_gt_img = self.read_mhd(source_gt_path, 1)
            if stage == 'train':
                source_img = self.datagen.apply_transform(source_img, transform)
                source_gt_img = self.datagen.apply_transform(source_gt_img, transform)

            # get target
            target_img = self.read_mhd(target_path, '_gt' in self.target_name)
            target_gt_img = self.read_mhd(target_gt_path, 1)
            if self.augment['AUG_TARGET'] and stage == 'train':
                if not self.augment['AUG_SAME_FOR_BOTH']:
                    transform = self.datagen.get_random_transform(img_shape=self.img_res)
                target_img = self.datagen.apply_transform(target_img, transform)
                target_gt_img = self.datagen.apply_transform(target_gt_img, transform)

            # equalize LV height of source to target
            if self.equalize_lv_length:
                source_img, source_gt_img = self.equalize_lv(target_gt_img, source_img, source_gt_img)

            # add to list
            source_imgs.append(source_img)
            source_gt_imgs.append(source_gt_img)
            target_imgs.append(target_img)
            target_imgs_gt.append(target_gt_img)

        np.array(source_imgs)
        return np.array(target_imgs), np.array(target_imgs_gt), np.array(source_imgs), np.array(source_gt_imgs)

    def equalize_lv(self, target_gt_img, source_img, source_gt_img):
        def resize_img(img, ratio):
            img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
            img = match_image_size(img, self.img_res)
            img = np.expand_dims(img, -1)
            assert img.shape[0] == target_gt_img.shape[0] and img.shape[1] == target_gt_img.shape[1]
            return img

        # calculate ratio to resize
        source_lv_length, _, _ = get_LV_lenght(source_gt_img.squeeze(), True)
        target_lv_length, _, _ = get_LV_lenght(target_gt_img.squeeze(), True)
        ratio = target_lv_length / source_lv_length

        # resize source image and gt image
        source_img = resize_img(source_img, ratio)
        source_gt_img = resize_img(source_gt_img, ratio)

        return source_img, source_gt_img
