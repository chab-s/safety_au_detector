import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import os

from tensorflow.keras import layers
from lib.utils import ModelInspector

# ═════════════════════════════════════════════════════════════════════════════
# 0. config
# ═════════════════════════════════════════════════════════════════════════════

# min, max, num_bins
CONFIG_BINS = [
    (-1.0, 1.0, 64),
    (-1.0, 1.0, 64),
    (0.0, 1.0, 64),
]

CONFIG_HEAD = [
    (128, 68, 64)
]

# ═════════════════════════════════════════════════════════════════════════════
# 1. BASE BLOCKS
# ═════════════════════════════════════════════════════════════════════════════

class HeatmapHead(tf.keras.Model):
    def __init__(self, input_shape, num_lm: int, num_bins: int =64):
        super().__init__()
        self.num_lm = num_lm
        self.num_bins = num_bins

        self.bins = [
            tf.linspace(-1.0, 1.0, num_bins),
            tf.linspace(-1.0, 1.0, num_bins),
            tf.linspace(0.0, 1.1, num_bins),
        ]

        self.lm_proj = tf.keras.Sequential([
            layers.Dense(units=num_lm * input_shape[-1], use_bias=True),
            layers.Reshape((num_lm, input_shape[-1])),
        ])

        self.axis_heatmaps = [
                tf.keras.Sequential([
                tf.keras.layers.Dense(self.num_bins),
            ]) for _ in range(3)
        ]

    def call(self, features):
        lm_feats = self.lm_proj(features)

        raw_logits = [axis(lm_feats) for axis in self.axis_heatmaps]
        heatmaps = tf.stack(raw_logits, axis=2)

        coords = [
            tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * bins, axis=-1)
            for logits, bins in zip(raw_logits, self.bins)
        ]

        landmarks = tf.stack(coords, axis=-1)
        return landmarks, heatmaps


# ═════════════════════════════════════════════════════════════════════════════
# 4. LOSS
# ═════════════════════════════════════════════════════════════════════════════

def _decode_prediction(x_bins_base, y_bins_base, center, scale):
    x_bins = x_bins_base[None, None, :] * scale[:, None, 0:1] / 2 + center[:, None, 0:1]
    y_bins = y_bins_base[None, None, :] * scale[:, None, 1:2] / 2 + center[:, None, 1:2]

    return x_bins, y_bins

def decode_xy(x_hms, y_hms, z_hms, x_bins, y_bins, z_bins):
    x = tf.reduce_sum(x_hms * tf.expand_dims(x_bins, axis=1), axis=-1)
    y = tf.reduce_sum(y_hms * tf.expand_dims(y_bins, axis=1), axis=-1)
    z = tf.reduce_sum(z_hms * tf.expand_dims(z_bins, axis=1), axis=-1)

    return tf.stack([x, y, z], axis=-1)

def generate_heatmaps(landmarks_gt, bbox_cs, sigmas, areas, x_bins_base, y_bins_base, z_bins_base):
    center = bbox_cs[:, :2]
    scale = bbox_cs[:, 2:]

    x_gt = landmarks_gt[..., 0:1]
    y_gt = landmarks_gt[..., 1:2]
    z_gt = landmarks_gt[..., 2:3]
    z_gt = tf.clip_by_value(z_gt, 0.0, 1.1)

    x_bins, y_bins = _decode_prediction(x_bins_base, y_bins_base, center, scale)
    dist_x = tf.abs(x_gt - x_bins)
    dist_y = tf.abs(y_gt - y_bins)
    dist_z = tf.abs(z_gt - z_bins_base)

    areas = tf.sqrt(tf.maximum(areas, 1.0))
    sigmas = tf.maximum(sigmas, 1e-3)

    dist_x = dist_x / areas[..., None] / sigmas[..., None]
    dist_y = dist_y / areas[..., None] / sigmas[..., None]
    dist_z = dist_z / areas[..., None] / sigmas[..., None]

    hm_x = tf.exp(-dist_x / 2.0) / sigmas[..., None]
    hm_y = tf.exp(-dist_y / 2.0) / sigmas[..., None]
    hm_z = tf.exp(-dist_z / 2.0) / sigmas[..., None]

    return hm_x, hm_y, hm_z

def compute_hm_loss(target_hm, pred_hm, num_bins: int = 64):
    pred_hm = tf.nn.softmax(pred_hm, axis=-1)
    target_hm = target_hm / (tf.reduce_sum(target_hm, axis=-1, keepdims=True) + 1e-8)
    kl_loss = tf.keras.losses.KLDivergence(reduction='sum_over_batch_size', name='kl_divergence')
    # loss_hm = kl_loss(
    #     tf.reshape(target_hm, (-1, num_bins)),
    #     tf.reshape(pred_hm, (-1, num_bins))
    # )
    loss_hm = kl_loss(target_hm, pred_hm)
    return loss_hm

# Wing loss
def compute_lm_loss(pred_lm, target_lm, w=10.0, epsilon=2.0):
    # pred, target : (B, N_landmarks, 3) pour x, y, z
    diff = tf.abs(pred_lm - target_lm)
    C = w - w * tf.math.log(1.0 + w / epsilon)
    loss = tf.where(
        diff < w,
        w * tf.math.log(1.0 + diff / epsilon),
        diff - C
    )
    return tf.reduce_mean(loss)

def compute_nme(pred_landmarks, gt_landmarks, bbox_size):
    # pred/gt : (batch, num_lm, 2) — seulement X et Y
    diff = tf.norm(pred_landmarks[..., :2] - gt_landmarks[..., :2], axis=-1)
    nme = tf.reduce_mean(diff) / bbox_size
    return nme


def compute_physical_loss():
    pass

# ═════════════════════════════════════════════════════════════════════════════
# 5. DATASET
# ═════════════════════════════════════════════════════════════════════════════


import yaml
from types import SimpleNamespace

def load_config(file_path: str) -> SimpleNamespace:
    with open(file_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    def wrap(d):
        if not isinstance(d, dict):
            return d
        return SimpleNamespace(**{k: wrap(v) for k, v in d.items()})

    return wrap(config_dict)

def get_aflw2k3d_dataset(cfg):
    ds = tfds.load('aflw2k3d', split='train')

    def process_data(x):
        # 1. Image : Resize et Normalisation
        img = x['image']
        img = tf.image.resize(img, (cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]))
        img = tf.cast(img, tf.float32) / 255.0

        # 2. Landmarks : Fusionner XY et Z
        xy = x['landmarks_68_3d_xy_normalized']  # (68, 2) - Valeurs entre 0 et 1
        z = x['landmarks_68_3d_z']  # (68, 1)

        # Remettre XY à l'échelle des pixels (ex: 256)
        x_coords = xy[:, 0:1] * cfg.MODEL.IMAGE_SIZE[1]
        y_coords = xy[:, 1:2] * cfg.MODEL.IMAGE_SIZE[0]

        # Concaténer en (68, 3) -> [x_px, y_px, z]
        lms_3d = tf.concat([x_coords, y_coords, z], axis=-1)

        return img, lms_3d

    return ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)


# ═════════════════════════════════════════════════════════════════════════════
# 6. DÉTECTEUR COMPLET
# ═════════════════════════════════════════════════════════════════════════════




if __name__ == "__main__":
    pass