import scipy
from glob import glob
import numpy as np
import os
import skimage.io as io
from PIL import Image
from prefetch_generator import background

NUM_PREFETCH = 10


class DataLoaderCamus:
    def __init__(self, dataset_path, input_name, target_name, img_res, target_rescale, input_rescale, train_ratio,
                 labels):
        self.dataset_path = dataset_path
        self.img_res = tuple(img_res)
        self.target_rescale = target_rescale
        self.input_rescale = input_rescale
        self.input_name = input_name
        self.target_name = target_name

        patients = sorted(glob(os.path.join(self.dataset_path, 'training', '*')))
        num = len(patients)
        num_train = int(num * (1 - train_ratio))
        self.train_patients = patients[:num_train]
        self.valid_patients = patients[num_train:]
        self.test_patients = glob(os.path.join(self.dataset_path, 'testing', '*'))
        print('#train:', len(self.train_patients))
        print('#valid:', len(self.valid_patients))
        print('#test:', len(self.test_patients))

        all_labels = {0, 1, 2, 3}
        self.not_labels = all_labels - set(labels)

        # from keras.preprocessing.image import ImageDataGenerator
        # data_gen_args = dict(rotation_range=10,
        #                      shear_range=0.01,
        #                      zoom_range=[0.9, 1.1],
        #                      fill_mode='constant',
        #                      cval=0.,
        #                      data_format='channels_last')
        # self.image_gen = ImageDataGenerator(**data_gen_args, rescale=self.img_rescale)

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
            target_imgs, input_imgs = self._get_batch(batch_paths)
            target_imgs = target_imgs * self.target_rescale
            input_imgs = input_imgs * self.input_rescale

            yield target_imgs, input_imgs

    def get_iterative_batch(self, batch_size=1, stage='test'):
        paths = self._get_paths(stage)

        num = len(paths)
        num_batches = num // batch_size

        start_idx = 0
        for i in range(num_batches):
            batch_paths = paths[start_idx:start_idx + batch_size]
            target_imgs, input_imgs= self._get_batch(batch_paths)
            target_imgs = target_imgs * self.target_rescale
            input_imgs = input_imgs * self.input_rescale

            yield target_imgs, input_imgs

    def _get_batch(self, paths_batch):
        target_imgs = []
        input_imgs = []
        for path in paths_batch:
            head, patient_id = os.path.split(path)
            target_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.target_name))
            input_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.input_name))

            target_img = self.read_mhd(target_path, '_gt' in self.target_name)
            target_imgs.append(target_img)

            input_img = self.read_mhd(input_path, '_gt' in self.input_name)
            input_imgs.append(input_img)

        return np.array(target_imgs), np.array(input_imgs)

