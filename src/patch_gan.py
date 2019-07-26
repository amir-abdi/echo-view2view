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
from utils import gen_fig

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
        # input_target = Input(shape=self.img_shape)
        input_layer = Input(shape=self.img_shape)

        if config['TYPE'] == 'Segmentation':
            self.gf = config['FIRST_LAYERS_FILTERS']
            self.skipconnections_generator = config['SKIP_CONNECTIONS_GENERATOR']
            self.output_activation = config['GEN_OUTPUT_ACT']
            self.decay_factor_G = config['LR_EXP_DECAY_FACTOR_G']
            self.optimizer_G = Adam(config['LEARNING_RATE_G'], config['ADAM_B1'])
            print('Building segmentation model')
            self.seg_model = Generator(self.img_shape, self.gf, self.channels, self.output_activation, self.skipconnections_generator).build()
            seg = self.seg_model(input_layer)
            self.combined = Model(inputs=[input_layer], outputs=[seg])
            num_gpu = len(K.tensorflow_backend._get_available_gpus())
            if num_gpu > 1:
                self.combined = multi_gpu_model(self.combined, gpus=num_gpu)

            # self.combined.compile(loss=['mse', 'mae'],
            #                       optimizer=self.optimizer_G,
            #                       loss_weights=[config['LOSS_WEIGHT_DISC'],
            #                                     config['LOSS_WEIGHT_GEN']])
            self.combined.compile(loss=loss_dice_coefficient_error,
                                   optimizer=self.optimizer_G)

        if config['TYPE'] in ['PatchGAN','PatchGAN_Constrained']:
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
            self.discriminator = Discriminator(self.img_shape, self.df, num_layers_D).build()
            self.discriminator.compile(loss='mse', optimizer=self.optimizer_D, metrics=['accuracy'])

            # Build the generator
            print('Building generator')
            self.generator = Generator(self.img_shape, self.gf, self.channels, self.output_activation,
                                       self.skipconnections_generator).build()


            # Turn of discriminator training for the combined model (i.e. generator)
            fake_img = self.generator(input_layer)
            self.discriminator.trainable = False
            valid = self.discriminator(fake_img)

            if config['TYPE'] == 'PatchGAN':
                # with tf.device('/cpu:0'):
                self.combined = Model(inputs=[input_layer], outputs=[valid, fake_img])
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
                self.combined = Model(inputs=[input_layer], outputs=[valid, fake_img, valid_seg])
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
        if self.config['TYPE'] in ['PatchGAN', 'PatchGAN_Constrained']:
            valid = np.ones((batch_size,) + self.num_patches)
            fake = np.zeros((batch_size,) + self.num_patches)
            print(valid.shape)

        while self.step < max_iter:
            for targets, targets_gt, inputs in self.data_loader.get_random_batch(batch_size):
                #  ---------- Train Discriminator -----------


                # ----------- Train Generator -----------
                if self.config['TYPE'] == 'Segmentation':
                    g_loss = self.combined.train_on_batch([inputs], [targets_gt])
                    if self.step % log_interval == 0:
                        elapsed_time = datetime.datetime.now() - start_time
                        print('[iter %d/%d] [G loss: %f] time: %s'
                              % (self.step, max_iter, g_loss, elapsed_time))
                        K.set_value(self.optimizer_G.lr, self.exp_decay(self.step, self.decay_factor_G, self.lr_G))

                elif self.config['TYPE'] in ['PatchGAN','PatchGAN_Constrained']:
                    fake_imgs = self.generator.predict(inputs)
                    d_loss_real = self.discriminator.train_on_batch([targets], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_imgs], fake)
                    d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
                    d_acc_real = d_loss_real[1] * 100
                    d_acc_fake = d_loss_fake[1] * 100
                    if self.config['TYPE'] == 'PatchGAN':
                        g_loss = self.combined.train_on_batch([inputs], [valid, targets])

                    elif self.config['TYPE'] == 'PatchGAN_Constrained':
                        g_loss = self.combined.train_on_batch([inputs], [valid, targets, targets_gt])

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

        # os.makedirs('%s/%s/%s' % (RESULT_DIR, self.result_name, VAL_DIR), exist_ok=True)

        targets, targets_gt, inputs = next(self.data_loader.get_random_batch(batch_size=3, stage='valid'))
        if self.config['TYPE'] == 'Segmentation':
            seg_pred = self.seg_model.predict(inputs)
            fig = gen_fig(inputs / self.input_trans,
                          seg_pred / self.target_trans,
                          targets_gt / self.target_trans)

        if self.config['TYPE'] in ['PatchGAN','PatchGAN_Constrained']:
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

    def test(self):
        image_dir = '%s/%s/%s' % (RESULT_DIR, self.result_name, TEST_DIR)
        print(image_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        # os.makedirs(image_dir, exist_ok=True)

        for batch_i, (targets, inputs) in enumerate(self.data_loader.get_iterative_batch(3, stage='test')):
            fake_imgs = self.generator.predict(inputs)
            fig = gen_fig(inputs / self.input_trans,
                          fake_imgs / self.target_trans,
                          targets / self.target_trans)
            fig.savefig('%s/%d.png' % (image_dir, batch_i))
