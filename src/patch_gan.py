import datetime
import numpy as np
import os
import json

from keras.utils import multi_gpu_model
from keras.layers import Input
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras import backend as K
from keras.optimizers import tf

from models import Discriminator, Generator, loss_dice_coefficient_error
from utils import gen_fig, gen_fig_seg

RESULT_DIR = 'results'
VAL_DIR = 'val_images'
TEST_DIR = 'test_images'
MODELS_DIR = 'saved_models'


class PatchGAN:
    def __init__(self, data_loader, config, use_wandb):
        # Configure data loader
        self.config = config
        self.result_name = config['NAME']
        self.data_loader = data_loader
        self.use_wandb = use_wandb
        self.step = 0

        # Input shape
        self.channels = config['CHANNELS']
        self.img_rows = config['IMAGE_RES'][0]
        self.img_cols = config['IMAGE_RES'][1]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        assert self.img_rows == self.img_cols, 'The current code only works with same values for img_rows and img_cols'

        # scaling
        self.target_trans = config['TARGET_TRANS']
        self.input_trans = config['INPUT_TRANS']

        # Input images and their conditioning images
        self.conditional_d = config.get('CONDITIONAL_DISCRIMINATOR', False)
        input_target = Input(shape=self.img_shape)
        input_input = Input(shape=self.img_shape)

        if config['TYPE'] == 'Segmentation':
            self.gf = config['FIRST_LAYERS_FILTERS']
            self.skipconnections_generator = config['SKIP_CONNECTIONS_GENERATOR']
            self.output_activation = config['GEN_OUTPUT_ACT']
            self.decay_factor_G = config['LR_EXP_DECAY_FACTOR_G']
            self.optimizer_G = Adam(config['LEARNING_RATE_G'], config['ADAM_B1'])
            print('Building segmentation model')
            self.seg_model = Generator(self.img_shape, self.gf, self.channels, self.output_activation,
                                       self.skipconnections_generator).build()
            seg = self.seg_model(input_input)
            self.combined = Model(inputs=[input_input], outputs=[seg])
            num_gpu = len(K.tensorflow_backend._get_available_gpus())
            if num_gpu > 1:
                self.combined = multi_gpu_model(self.combined, gpus=num_gpu)

            # self.combined.compile(loss=['mse', 'mae'],
            #                       optimizer=self.optimizer_G,
            #                       loss_weights=[config['LOSS_WEIGHT_DISC'],
            #                                     config['LOSS_WEIGHT_GEN']])
            self.combined.compile(loss=loss_dice_coefficient_error,
                                  optimizer=self.optimizer_G)

        if config['TYPE'] in ['PatchGAN', 'PatchGAN_Constrained']:
            # Calculate output shape of D (PatchGAN)
            patch_size = config['PATCH_SIZE']
            patch_per_dim = int(self.img_rows / patch_size)
            self.num_patches = (patch_per_dim, patch_per_dim, 1)
            num_layers_D = int(np.log2(patch_size))

            # Number of filters in the first layer of G and D
            self.gf = config['FIRST_LAYERS_FILTERS']
            self.df = config['FIRST_LAYERS_FILTERS']
            self.skipconnections_generator = config['SKIP_CONNECTIONS_GENERATOR']
            self.output_activation = config['GEN_OUTPUT_ACT']
            self.decay_factor_G = config['LR_EXP_DECAY_FACTOR_G']
            self.decay_factor_D = config['LR_EXP_DECAY_FACTOR_D']
            self.optimizer_G = Adam(config['LEARNING_RATE_G'], config['ADAM_B1'])
            self.optimizer_D = Adam(config['LEARNING_RATE_D'], config['ADAM_B1'])

            # Build and compile the discriminator
            print('Building discriminator')
            self.discriminator = Discriminator(self.img_shape, self.df, num_layers_D,
                                               conditional=self.conditional_d).build()
            self.discriminator.compile(loss='mse', optimizer=self.optimizer_D, metrics=['accuracy'])

            # Build the generator
            print('Building generator')
            self.generator = Generator(self.img_shape, self.gf, self.channels, self.output_activation,
                                       self.skipconnections_generator).build()

            # Turn of discriminator training for the combined model (i.e. generator)
            fake_img = self.generator(input_input)
            self.discriminator.trainable = False
            if self.conditional_d:
                valid = self.discriminator([fake_img, input_target])
            else:
                valid = self.discriminator(fake_img)

            if config['TYPE'] == 'PatchGAN':
                if self.conditional_d:
                    self.combined = Model(inputs=[input_target, input_input], outputs=[valid, fake_img])
                else:
                    self.combined = Model(inputs=[input_input], outputs=[valid, fake_img])
                num_gpu = len(K.tensorflow_backend._get_available_gpus())
                print('num gpu: ', num_gpu)
                if num_gpu > 1:
                    self.combined = multi_gpu_model(self.combined, gpus=num_gpu)

                self.combined.compile(loss=['mse', 'mae'],
                                      optimizer=self.optimizer_G,
                                      loss_weights=[config['LOSS_WEIGHT_DISC'],
                                                    config['LOSS_WEIGHT_GEN']])
            if config['TYPE'] == 'PatchGAN_Constrained':
                seg_model_name = 'results/Basic_4CH_ED_4CH_ED_gt_01/saved_models/segmentation_model'
                with open(seg_model_name + '.json', 'r') as f:
                    self.seg_model = model_from_json(f.read())
                self.seg_model.load_weights(seg_model_name + '_weights.hdf5')
                self.seg_model.name = 'segmentation_model'

                valid_seg = self.seg_model(fake_img)

                # with tf.device('/cpu:0'):
                self.combined = Model(inputs=[input_input], outputs=[valid, fake_img, valid_seg])
                num_gpu = len(K.tensorflow_backend._get_available_gpus())
                print('num gpu: ', num_gpu)
                if num_gpu > 1:
                    self.combined = multi_gpu_model(self.combined, gpus=num_gpu)

                self.combined.compile(loss=['mse', 'mae', loss_dice_coefficient_error],
                                      optimizer=self.optimizer_G,
                                      loss_weights=[config['LOSS_WEIGHT_DISC'],
                                                    config['LOSS_WEIGHT_GEN'],
                                                    config['LOSS_WEIGHT_SEG']])

        # Training
        self.batch_size = config['BATCH_SIZE']
        self.max_iter = config['MAX_ITER']
        self.val_interval = config['VAL_INTERVAL']
        self.log_interval = config['LOG_INTERVAL']
        self.save_model_interval = config['SAVE_MODEL_INTERVAL']
        self.lr_G = config['LEARNING_RATE_G']
        self.lr_D = config['LEARNING_RATE_D']

    @staticmethod
    def exp_decay(global_iter, decay_factor, initial_lr):
        lrate = initial_lr * np.exp(-decay_factor * global_iter)
        return lrate

    def train(self):
        start_time = datetime.datetime.now()
        batch_size = self.batch_size
        max_iter = self.max_iter
        val_interval = self.val_interval
        log_interval = self.log_interval
        save_model_interval = self.save_model_interval

        # Adversarial loss ground truths
        if 'GAN' in self.config['TYPE']:
            valid = np.ones((batch_size,) + self.num_patches)
            fake = np.zeros((batch_size,) + self.num_patches)
            print('PatchGAN valid shape:', valid.shape)

        while self.step < max_iter:
            for targets, targets_gt, inputs in self.data_loader.get_random_batch(batch_size):

                # ----------- Train Segmentation Model -----------
                if self.config['TYPE'] == 'Segmentation':
                    g_loss = self.combined.train_on_batch([inputs], [targets_gt])
                    if self.step % log_interval == 0:
                        elapsed_time = datetime.datetime.now() - start_time
                        print('[iter %d/%d] [G loss: %f] time: %s'
                              % (self.step, max_iter, g_loss, elapsed_time))
                        K.set_value(self.optimizer_G.lr, self.exp_decay(self.step, self.decay_factor_G, self.lr_G))

                #  ---------- Train Discriminator -----------
                elif self.config['TYPE'] in ['PatchGAN', 'PatchGAN_Constrained']:
                    fake_imgs = self.generator.predict(inputs)

                    if self.conditional_d:
                        d_loss_real = self.discriminator.train_on_batch([targets, targets_gt], valid)
                        d_loss_fake = self.discriminator.train_on_batch([fake_imgs, targets_gt], fake)
                    else:
                        d_loss_real = self.discriminator.train_on_batch([targets], valid)
                        d_loss_fake = self.discriminator.train_on_batch([fake_imgs], fake)
                    d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
                    d_acc_real = d_loss_real[1] * 100
                    d_acc_fake = d_loss_fake[1] * 100

                    if self.conditional_d:
                        combined_inputs = [targets_gt, inputs]
                    else:
                        combined_inputs = [inputs]

                    if self.config['TYPE'] == 'PatchGAN':
                        combined_targets = [valid, targets]
                    else:
                        combined_targets = [valid, targets, targets_gt]

                    #  ---------- Train Generator -----------
                    g_loss = self.combined.train_on_batch(combined_inputs, combined_targets)

                    # Logging
                    if self.step % log_interval == 0:
                        elapsed_time = datetime.datetime.now() - start_time
                        print('[iter %d/%d] [D loss: %f, acc: r:%3d%% f:%3d%%] [G loss: %f] time: %s'
                              % (self.step, max_iter, d_loss, d_acc_real, d_acc_fake, g_loss[0], elapsed_time))

                        K.set_value(self.optimizer_G.lr, self.exp_decay(self.step, self.decay_factor_G, self.lr_G))
                        K.set_value(self.optimizer_D.lr, self.exp_decay(self.step, self.decay_factor_D, self.lr_D))

                        if self.use_wandb:
                            import wandb
                            wandb.log({'d_loss': d_loss, 'd_acc_real': d_acc_real, 'd_acc_fake': d_acc_fake,
                                       'g_loss': g_loss[0],
                                       'lr_G': K.eval(self.optimizer_G.lr),
                                       'lr_D': K.eval(self.optimizer_D.lr)},
                                      step=self.step)

                if self.step % val_interval == 0:
                    self.gen_valid_results(self.step)

                if self.step % save_model_interval == 0:
                    self.save_model()

                self.step += 1

    def gen_valid_results(self, step_num, prefix=''):
        path = '%s/%s/%s' % (RESULT_DIR, self.result_name, VAL_DIR)
        if not os.path.exists(path):
            os.makedirs(path)

        targets, targets_gt, inputs = next(self.data_loader.get_random_batch(batch_size=3, stage='valid'))
        if self.config['TYPE'] == 'Segmentation':
            seg_pred = self.seg_model.predict(inputs)
            fig = gen_fig(inputs / self.input_trans,
                          seg_pred / self.target_trans,
                          targets_gt / self.target_trans)

        if self.config['TYPE'] in ['PatchGAN', 'PatchGAN_Constrained']:
            fake_imgs = self.generator.predict(inputs)
            fig = gen_fig(inputs / self.input_trans,
                          fake_imgs / self.target_trans,
                          targets / self.target_trans)

        fig.savefig('%s/%s/%s/%s_%d.png' % (RESULT_DIR, self.result_name, VAL_DIR, prefix, step_num))

        if self.use_wandb:
            import wandb
            wandb.log({'val_image': fig}, step=self.step)

    def load_model(self, root_model_path):
        self.generator.load_weights(os.path.join(root_model_path, 'generator_weights.hdf5'))
        self.discriminator.load_weights(os.path.join(root_model_path, 'discriminator_weights.hdf5'))

        generator_json = json.load(open(os.path.join(root_model_path, 'generator.json')))
        discriminator_json = json.load(open(os.path.join(root_model_path, 'discriminator.json')))
        self.step = generator_json['iter']
        assert self.step == discriminator_json['iter']

        print('Weights loaded: {} @{}'.format(root_model_path, self.step))

    def save_model(self):
        model_dir = '%s/%s/%s' % (RESULT_DIR, self.result_name, MODELS_DIR)
        print(model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # os.makedirs(model_dir, exist_ok=True)

        def save(model, model_name):
            model_json_path = '%s/%s.json' % (model_dir, model_name)
            weights_path = '%s/%s_weights.hdf5' % (model_dir, model_name)
            options = {'file_arch': model_json_path,
                       'file_weight': weights_path}
            json_string = model.to_json()
            json_obj = json.loads(json_string)
            json_obj['iter'] = self.step
            open(options['file_arch'], 'w').write(json.dumps(json_obj, indent=4))
            model.save_weights(options['file_weight'])

        if self.config['TYPE'] == 'Segmentation':
            save(self.seg_model, 'segmentation_model')
        if self.config['TYPE'] in ['PatchGAN', 'PatchGAN_Constrained']:
            save(self.generator, 'generator')
            save(self.discriminator, 'discriminator')
        print('Model saved in {}'.format(model_dir))

    @staticmethod
    def rotate_degree(mask):
        from scipy import ndimage
        import math
        [COG_x, COG_y] = ndimage.measurements.center_of_mass(
            mask)  # returns tuple: x,y => first vertical second horizontal
        horizontal_sum = np.sum(mask > 0, axis=1)
        top_x = np.min(np.where(horizontal_sum > 0))
        top_y = np.median(np.where(mask[top_x, :] != 0))

        degree = np.arctan((COG_y - top_y) / (COG_x - top_x))
        return -math.degrees(degree)

    @staticmethod
    def fill_and_get_LCC(seg):  # binary fill and get largest connected component
        from scipy.ndimage.morphology import binary_fill_holes
        from skimage.measure import label
        seg[seg >= 0.5] = 1
        seg[seg < 0.5] = 0
        seg = binary_fill_holes(seg)
        seg = np.uint8(seg)
        if np.sum(seg) > 0:
            labels = label(seg)
            largestCC = labels == np.argmax(np.bincount(labels.flat, weights=seg.flat))
            return largestCC
        return -1

    def get_LV_lenght(self, mask):
        from scipy.ndimage import rotate
        mask = mask.astype('uint8')
        deg = 0
        if self.config['ROTATION_FOR_APICAL_MATCH'] == True:
            deg = self.rotate_degree(mask)
            mask = rotate(mask, deg)
        horizontal_sum = np.sum(mask > 0, axis=1)
        L_lenght = np.max(np.where(horizontal_sum > 0)) - np.min(np.where(horizontal_sum > 0))
        return mask, deg, L_lenght

    def match_apical(self, input, target_real,
                     target_fake):  # match ap2 and ap4 based on LV length and calculate the difference in area
        import cv2
        from scipy.ndimage import rotate
        _, _, L_i = self.get_LV_lenght(input)
        target_real, deg_tr, L_tr = self.get_LV_lenght(target_real)
        target_fake, deg_tf, L_tf = self.get_LV_lenght(target_fake)
        ratio_tr = L_i / L_tr
        ratio_tf = L_i / L_tf
        target_real = cv2.resize(target_real, (0, 0), fx=ratio_tr, fy=ratio_tr)
        target_fake = cv2.resize(target_fake, (0, 0), fx=ratio_tf, fy=ratio_tf)

        target_real = rotate(target_real, -deg_tr)
        target_fake = rotate(target_fake, -deg_tf)

        return target_real, target_fake

    def test(self, seg_load_addr):
        from xlwt import Workbook
        wb = Workbook()
        sheet1 = wb.add_sheet('Area')
        # seg_model = load_model(seg_load_addr + '/AP{}_Seg.h5'.format(self.config['TARGET_NAME'][0]))
        json_file = open(os.path.join(seg_load_addr, 'segmentation_model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        seg_model = model_from_json(loaded_model_json)
        seg_model.load_weights(os.path.join(seg_load_addr, 'segmentation_model_weights.hdf5'))

        image_dir = '%s/%s/%s' % (RESULT_DIR, self.result_name, TEST_DIR)
        print(image_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        # os.makedirs(image_dir, exist_ok=True)
        sheet1.write(0, 0, 'counter')
        sheet1.write(0, 1, 'real_area')
        sheet1.write(0, 2, 'fake_area')
        # sheet1.write(0, 3, 'difference')

        cnt = 1
        for batch_i, (targets, targets_seg_gt, inputs, inputs_seg_gt) in enumerate(
                self.data_loader.get_iterative_batch(2, stage='test')):
            # generate fake target image
            fake_imgs = self.generator.predict(inputs)

            # estimate the segmentation mask for real (target) and fake
            fake_segs = seg_model.predict(fake_imgs)
            target_segs = seg_model.predict(targets)

            for i in range(0, fake_segs.shape[0]):
                mask_fake = self.fill_and_get_LCC(fake_segs[i, :, :, 0])
                mask_target = self.fill_and_get_LCC(target_segs[i, :, :, 0])
                if len(mask_fake) == 1 or len(mask_target) == 1:
                    print('segmentation model error')
                    sheet1.write(cnt, 0, cnt)
                    sheet1.write(cnt, 1, '-')
                    sheet1.write(cnt, 2, '-')
                    sheet1.write(cnt, 3, '-')
                    cnt = cnt + 1
                    continue
                fake_segs[i, :, :, 0] = mask_fake
                target_segs[i, :, :, 0] = mask_target
                mask_real, mask_fake = self.match_apical(inputs_seg_gt[i, :, :, 0], target_segs[i, :, :, 0],
                                                         fake_segs[i, :, :, 0])
                sheet1.write(cnt, 0, cnt)
                sheet1.write(cnt, 1, np.sum(mask_real).astype('float64'))
                sheet1.write(cnt, 2, np.sum(mask_fake).astype('float64'))
                sheet1.write(cnt, 3, np.sum(targets_seg_gt).astype('float64'))

                # sheet1.write(cnt, 3, abs(np.sum(mask_real) - np.sum(mask_fake)).astype('float64'))
                cnt = cnt + 1
                print('test#: %d' % cnt)

            fig = gen_fig_seg(inputs / self.input_trans,
                              fake_imgs / self.target_trans,
                              targets / self.target_trans,
                              fake_segs / self.target_trans,
                              target_segs / self.target_trans)

            fig.savefig('%s/%d.png' % (image_dir, batch_i))

        wb.save('%s/Areas.xls' % (image_dir))
