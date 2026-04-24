import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from keras.src.legacy.saving.legacy_h5_format import HDF5_OBJECT_HEADER_LIMIT

from tensorflow.keras import layers
from tensorflow.keras import activations

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

class Identity(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs


class SELayer(layers.Layer):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.Sequential([
            layers.Dense(units=channel, use_bias=True),
            layers.ReLU(),
            layers.Dense(units=channel, use_bias=True),
            layers.Activation('hard_sigmoid')
        ])

    def call(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)

        return x * y

class InvertedResidual(layers.layer):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()

        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = tf.keras.Sequential([
                layers.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                layers.BatchNormalization(hidden_dim),
                activations.hard_silu() if use_hs else activations.RelU(inplace=True),
                SELayer(hidden_dim) if use_se else Identity(),
                layers.Conv2D(hidden_dim, oup, 1, 1, 0, bias=False),
                layers.BatchNormalization(oup),
            ])

        else:
            self.conv = tf.keras.Sequential([
                layers.Conv2D(inp, hidden_dim, 1, 1, 0, bias=False),
                layers.BatchNormalization(hidden_dim),
                activations.hard_silu() if use_hs else layers.ReLU(inplace=True),
                layers.Conv2D(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                layers.BatchNormalization(hidden_dim),
                SELayer(hidden_dim) if use_se else Identity(),
                activations.hard_silu() if use_hs else layers.ReLU(inplace=True),
                layers.Conv2D(hidden_dim, oup, 1, 1, 0, bias=False),
                layers.BatchNormalization(oup),
            ])

    def call(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            self.conv(x)


class Conv3x3Bn(layers.Layer):
    def __init__(self, out, s):
        super(Conv3x3Bn, self).__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv2D(out, 3, s, 0,use_bias=False),
            layers.BatchNormalization(out),
            activations.hard_silu()
        ])

    def call(self, x):
        return self.conv(x)

class Conv1x1Bn(layers.Layer):
    def __init__(self, out, s, use_hs=Falsed):
        super(Conv1x1Bn, self).__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv2D(out, 1, s, 0, use_bias=False),
            layers.BatchNormalization(out),
            activations.hard_silu() if use_hs else layers.ReLU(inplace=True),
        ])

    def call(self, x):
        return self.conv(x)



class MobileNetV3(layers.Layer):
    def __init(self, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        assert mode in ["large", "small"]

        self.conv = Conv3x3Bn(16, 2)
        self.bneck1 = InvertedResidual(0, 0, 16, 3, 1, False, False)
        self.bneck2 = InvertedResidual(0, 0, 24, 3, 2, False, False)
        self.bneck3 = InvertedResidual(0, 0, 24, 3, 1, False, False)
        self.bneck4 = InvertedResidual(0, 0, 40, 5, 2, True, False)
        self.bneck5 = InvertedResidual(0, 0, 40, 5, 1, True, False)
        self.bneck6 = InvertedResidual(0, 0, 40, 5, 1, True, False)
        self.bneck7 = InvertedResidual(0, 0, 80, 3, 2, False, True)
        self.bneck8 = InvertedResidual(0, 0, 80, 3, 1, False, True)
        self.bneck9 = InvertedResidual(0, 0, 80, 3, 1, False, True)
        self.bneck10 = InvertedResidual(0, 0, 80, 3, 1, False, True)
        self.bneck11 = InvertedResidual(0, 0, 112, 3, 1, True, True)
        self.bneck12 = InvertedResidual(0, 0, 112, 3, 1, True, True)
        self.bneck13 = InvertedResidual(0, 0, 160, 5, 2, True, True)
        self.bneck14 = InvertedResidual(0, 0, 160, 5, 1, True, True)
        self.bneck15 = InvertedResidual(0, 0, 160, 5, 1, True, True)
        self.conv2 = Conv1x1Bn(1280, 1, True)
        self.conv3 = Conv1x1Bn(0, 1, False)

    def call(self, x):
        y = self.conv(x)



