import tensorflow as tf
import pandas as pd
import cv2

from tensorflow.keras import layers
from lib.utils import ModelInspector

# ═════════════════════════════════════════════════════════════════════════════
# 0. config
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# 1. BASE BLOCKS
# ═════════════════════════════════════════════════════════════════════════════

class HeatmapHead(layers.Layer):
    def __init__(self, input_shape, num_kp: int, num_bins: int =64):
        super(HeatmapHead).__init__()
        self.num_kp = num_kp
        self.num_bins = num_bins

        self.bins = [
            tf.linspace(-1.0, 1.0, num_bins),
            tf.linspace(-1.0, 1.0, num_bins),
            tf.linspace(0.0, 1.1, num_bins),
        ]

        self.kp_proj = tf.keras.Sequential([
            layers.Dense(units=num_kp * input_shape[-1], use_bias=True),
            layers.Reshape((num_kp, input_shape[-1])),
        ])

        self.axis_heatmaps = [
                tf.keras.Sequential([
                tf.keras.layers.Dense(self.num_bins),
            ])
        ]

    def call(self, features):
        kp_feats = self.kp_proj(features)

        coords = [
            tf.reduce_sum(tf.nn.softmax(axis(kp_feats), axis=-1) * bins, axis=-1)
            for axis, bins in zip(self.axis_heatmaps, self.bins)
        ]

        keypoints = tf.stack(coords, axis=-1)
        return keypoints


# ═════════════════════════════════════════════════════════════════════════════
# 5. DATASET
# ═════════════════════════════════════════════════════════════════════════════



if __name__ == "__main__":
    print("--- Check ---")
    model = HeatmapHead(input_shape=(1, 96), num_kp=96, num_bins=64)

    ModelInspector.trace(model, input_shape=(1, 96))
