import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from keras.src.legacy.saving.legacy_h5_format import HDF5_OBJECT_HEADER_LIMIT

from tensorflow.keras import layers
from tensorflow.keras import activations
from lib.utils import ModelInspector

# ═════════════════════════════════════════════════════════════════════════════
# 0. config
# ═════════════════════════════════════════════════════════════════════════════

# @dataclasses
# class Level:
#     name: str
#     kernel_size: tuple
#     strides: int
#     type: str

# mode, kernel_size, exp_size, output_channels, SE, HS, strides
CONFIG = [
    ("l", 3, 16, 16, False, False, 1),
    ("l", 3, 64, 24, False, False, 2),
    ("l", 3, 72, 24, False, False, 1),
    ("l", 5, 72, 40, True, False, 2),
    ("l", 5, 120, 40, True, False, 1),
    ("l", 5, 120, 40, True, False, 1),
    ("l", 3, 240, 80, False, True, 2),
    ("l", 3, 200, 80, False, True, 1),
    ("l", 3, 184, 80, False, True, 1),
    ("l", 3, 184, 80, False, True, 1),
    ("l", 3, 480, 112, True, True, 1),
    ("l", 3, 672, 112, True, True, 1),
    ("l", 5, 672, 160, True, True, 2),
    ("l", 5, 960, 160, True, True, 1),
    ("l", 5, 960, 160, True, True, 1),


    ("s", 3, 16, 16, False, True, 2),
    ("s", 3, 72, 24, False, True, 2),
    ("s", 3, 88, 24, False, True, 1),
    ("s", 5, 96, 40, False, True, 2),
    ("s", 5, 240, 40, False, True, 1),
    ("s", 5, 240, 40, False, True, 1),
    ("s", 5, 120, 48, False, True, 1),
    ("s", 5, 144, 48, False, True, 1),
    ("s", 5, 288, 96, False, True, 2),
    ("s", 5, 576, 96, False, True, 1),
    ("s", 5, 576, 96, False, True, 1),
]

# ═════════════════════════════════════════════════════════════════════════════
# 1. BASE BLOCKS
# ═════════════════════════════════════════════════════════════════════════════
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(layers.Layer):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        reduce_channel = _make_divisible(channel // reduction, 8)
        self.fc = tf.keras.Sequential([
            layers.Dense(units=reduce_channel, use_bias=True),
            layers.ReLU(),
            layers.Dense(units=channel, use_bias=True),
            layers.Activation('hard_sigmoid')
        ])

    def call(self, inputs):
        y = self.avg_pool(inputs)
        y = self.fc(y)
        y = tf.reshape(y, (-1, 1, 1, tf.shape(y)[-1]))
        return inputs * y

class InvertedResidual(layers.Layer):
    def __init__(self, input_channels, kernel_size, exp_size, output_channels, use_se, use_hs, strides):
        super(InvertedResidual, self).__init__()

        assert strides in [1, 2]

        self.use_res_connect = strides == 1 and input_channels == output_channels

        self.activation = layers.Activation(hard_swish) if use_hs else layers.ReLU()
        self.se = SELayer(exp_size) if use_se else layers.Identity()

        self.conv_pw_expand = None
        if exp_size != input_channels:
            self.conv_pw_expand = tf.keras.Sequential([
                layers.Conv2D(exp_size, 1, 1, padding="same", use_bias=False),
                layers.BatchNormalization(),
                self.activation
            ])

        self.conv_dw = tf.keras.Sequential([
            layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False),
            layers.BatchNormalization(),
            self.activation
        ])

        self.conv_pw_project = tf.keras.Sequential([
            layers.Conv2D(output_channels, 1, 1, padding="same", use_bias=False),
            layers.BatchNormalization(),
        ])

    def call(self, x, training=False):
        skip = x

        if self.conv_pw_expand is not None:
            x = self.conv_pw_expand(x, training=training)
        y = self.conv_dw(x, training=training)
        y = self.se(y, training=training)
        y = self.conv_pw_project(y, training=training)

        return skip + y if self.use_res_connect else y


class Conv3x3Bn(layers.Layer):
    def __init__(self, out, strides, use_norm, use_hs=True):
        super(Conv3x3Bn, self).__init__()
        layers_list = []

        layers_list.append(layers.Conv2D(out, 3, strides, "same",use_bias=False))
        if use_norm:
            layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation(hard_swish) if use_hs else layers.ReLU())
        self.conv = tf.keras.Sequential(layers_list)

    def call(self, inputs, training=False):
        return self.conv(inputs, training=training)


class Conv1x1Bn(layers.Layer):
    def __init__(self, out, strides, use_norm, use_hs=False):
        super(Conv1x1Bn, self).__init__()
        layers_list = []

        layers_list.append(layers.Conv2D(out, 1, strides, "same", use_bias=False),)
        if use_norm:
            layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation(hard_swish) if use_hs else layers.ReLU())
        self.conv = tf.keras.Sequential(layers_list)

    def call(self, inputs, training=False):
        return self.conv(inputs, training=training)



class MobileNetV3(layers.Layer):
    def __init__(self, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        assert mode in ["large", "small"]

        self.conv_in = Conv3x3Bn(16, 2, use_norm=True, use_hs=True)

        if mode == "large":
            conf = [c for c in CONFIG if 'l' in c]
            last_conv_channels = 960
            final_expand_channels = 1280
        else:
            conf = [c for c in CONFIG if 's' in c]
            last_conv_channels = 576
            final_expand_channels =  1024

        bneck_layers = []
        input_channels = _make_divisible(16 * width_mult, 8)
        for _, kernel_size, exp_size, output_channels, use_se, use_hs, strides in conf:
            print(_, kernel_size, exp_size, output_channels, use_se, use_hs, strides)
            output_channels = _make_divisible(output_channels * width_mult, 8)
            exp_size = _make_divisible(exp_size * width_mult, 8)
            bneck_layers.append(InvertedResidual(input_channels, kernel_size, exp_size, output_channels, use_se, use_hs, strides))
            input_channels = output_channels

        self.bneck_block = tf.keras.Sequential(bneck_layers)

        last_conv_channels = _make_divisible(last_conv_channels * width_mult, 8)
        final_expand_channels = _make_divisible(final_expand_channels * width_mult, 8)
        self.head = tf.keras.Sequential([
            Conv1x1Bn(last_conv_channels, 1, use_norm=True, use_hs=True),
            layers.GlobalAveragePooling2D(),
            # classification
            # layers.Reshape((1, 1, last_conv_channels)),
            # Conv1x1Bn(final_expand_channels, 1, use_norm=False, use_hs=True),
            # layers.Conv2D(num_classes, 1, strides=1, padding="same", use_bias=False),
            # layers.Flatten(),
            layers.Dense(final_expand_channels),
            layers.Activation(hard_swish),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation=None)
        ])

    def call(self, inputs, training=False):
        y = self.conv_in(inputs, training=training)
        y = self.bneck_block(y, training=training)
        y = self.head(y, training = training)
        return y



def hard_swish(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0


if __name__ == "__main__":
    print("--- Check ---")
    model = MobileNetV3(mode="large", num_classes=1000)

    ModelInspector.trace(model, input_shape=(224, 224, 3))





