# 1) Removed skip connections of the generator
# 2) Added segmentation model

from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, UpSampling2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Model


class Generator:
    def __init__(self, img_shape, filters, channels, output_activation, skipconnections_generator):
        self.img_shape = img_shape
        self.filters = filters
        self.channels = channels
        self.output_activation = output_activation
        self.skipconnections_generator = skipconnections_generator

    def build(self):
        def conv2d(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size,
                       strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = Conv2DTranspose(filters, kernel_size=f_size, strides=(2, 2),
                                padding='same', activation='linear')(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1,
                       padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            if self.skipconnections_generator:
                u = Concatenate()([u, skip_input])


            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling: 7 x stride of 2 --> x1/128 downsampling
        d1 = conv2d(d0, self.filters, bn=False)
        d2 = conv2d(d1, self.filters * 2)
        d3 = conv2d(d2, self.filters * 4)
        d4 = conv2d(d3, self.filters * 8)
        d5 = conv2d(d4, self.filters * 8)
        d6 = conv2d(d5, self.filters * 8)
        d7 = conv2d(d6, self.filters * 8)

        # Upsampling: 6 x stride of 2 --> x64 upsampling
        u1 = deconv2d(d7, d6, self.filters * 8)
        u2 = deconv2d(u1, d5, self.filters * 8)
        u3 = deconv2d(u2, d4, self.filters * 8)
        u4 = deconv2d(u3, d3, self.filters * 4)
        u5 = deconv2d(u4, d2, self.filters * 2)
        u6 = deconv2d(u5, d1, self.filters)
        u7 = Conv2DTranspose(self.channels, kernel_size=4, strides=(2, 2),
                             padding='same', activation='linear')(u6)

        # added conv layers after the deconvs to avoid the pixelated outputs
        output_img = Conv2D(self.channels, kernel_size=4,
                            strides=1, padding='same',
                            activation=self.output_activation)(u7)

        return Model(d0, output_img)


class Discriminator:
    def __init__(self, img_shape, filters, num_layers):
        self.img_shape = img_shape
        self.filters = filters
        self.num_layers = num_layers

    def build(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        input_targets = Input(shape=self.img_shape)
        input_inputs = Input(shape=self.img_shape)
        combined_imgs = Concatenate(axis=-1)([input_targets, input_inputs])

        # 4 d_layers with stride of 2 --> output is 1/16 in each dimension
        d = d_layer(combined_imgs, self.filters, bn=False)

        for i in range(self.num_layers - 1):
            d = d_layer(d, self.filters * (2 ** (i + 1)))

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d)

        return Model([input_targets, input_inputs], validity)



class Segmentation_model:
    def __init__(self, img_shape, filters):
        self.img_shape = img_shape
        self.gf = filters

    def build(self):
        def conv2d(layer_input, filters, f_size=3, bn=True, dropout_rate=0.0):

            d = Conv2D(filters, kernel_size=f_size, padding='same', kernel_initializer='he_normal')(layer_input)
            d = Activation('relu')(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)

            if dropout_rate:
                d = Dropout(dropout_rate)(d)

            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, bn=True, dropout_rate=0.0):

            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=2, padding='same', kernel_initializer='he_normal')(u)
            u = Concatenate()([skip_input, u])
            u = Conv2D(filters, kernel_size=f_size, padding='same', kernel_initializer='he_normal')(u)
            if bn:
                u = BatchNormalization(momentum=0.8)(u)

            u = Activation('relu')(u)

            if dropout_rate:
                u = Dropout(dropout_rate)(u)

            # print(u.shape)
            return u

            # Image input

        # d0 = Input((self.img_rows, self.img_cols, self.img_temp))

        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d1 = conv2d(d1, self.gf, bn=False)
        d1_p = MaxPooling2D(pool_size=(2, 2))(d1)
        d2 = conv2d(d1_p, self.gf * 2)
        d2 = conv2d(d2, self.gf * 2)
        d2_p = MaxPooling2D(pool_size=(2, 2))(d2)
        d3 = conv2d(d2_p, self.gf * 4)
        d3 = conv2d(d3, self.gf * 4)
        d3_p = MaxPooling2D(pool_size=(2, 2))(d3)
        d4 = conv2d(d3_p, self.gf * 8)
        d4 = conv2d(d4, self.gf * 8)
        d4_p = MaxPooling2D(pool_size=(2, 2))(d4)
        d5 = conv2d(d4_p, self.gf * 16)
        d5 = conv2d(d5, self.gf * 16)

        # Upsampling
        u0 = deconv2d(d5, d4, self.gf * 8)
        u0 = conv2d(u0, self.gf * 8)
        u1 = deconv2d(u0, d3, self.gf * 4)
        u1 = conv2d(u1, self.gf * 4)
        u2 = deconv2d(u1, d2, self.gf * 2)
        u2 = conv2d(u2, self.gf * 2)

        u3 = deconv2d(u2, d1, self.gf)
        u3 = conv2d(u3, self.gf)
        output_img = Conv2D(1, 1)(u3)
        output_img = Activation('sigmoid')(output_img)

        return Model(d0, output_img)

import keras.backend as K
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor))


def loss_dice_coefficient_error(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)