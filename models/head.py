import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import numpy as np
from pathlib import Path
from time import time

from models.mobileNetV3 import MobileNetV3
from lib.training import (LandmarkTrainingHistory, save_checkpoint, load_latest_checkpoint,
                           print_step, print_epoch, plot_history)
from lib.vision import visualize_side_by_side

# ═════════════════════════════════════════════════════════════════════════════
# 0. config
# ═════════════════════════════════════════════════════════════════════════════

# min, max, num_bins
CONFIG_BINS = [
    (-1.0, 1.0, 64),  # x
    (-1.0, 1.0, 64),  # y
    (0.0, 1.1, 64),   # z
]

CONFIG_HEAD = [
    (128, 68, 64)
]

# ═════════════════════════════════════════════════════════════════════════════
# 1. BASE BLOCKS
# ═════════════════════════════════════════════════════════════════════════════

class HeatmapHead(tf.keras.Model):
    def __init__(self, input_shape, num_lm: int):
        super().__init__()
        self.bins = []
        self.axis_heatmaps = []

        self.lm_proj = tf.keras.Sequential([
            layers.Dense(units=num_lm * input_shape[-1], use_bias=True),
            layers.Reshape((num_lm, input_shape[-1])),
        ])

        for b_min, b_max, num_bins in CONFIG_BINS:
            self.bins.append(tf.linspace(b_min, b_max, num_bins))
            self.axis_heatmaps.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(num_bins)
                    ]
                )
            )

    def call(self, features):
        lm_feats = self.lm_proj(features)
        raw_logits = [axis(lm_feats) for axis in self.axis_heatmaps]
        coords = [
            tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * bins, axis=-1)
            for logits, bins in zip(raw_logits, self.bins)
        ]

        heatmaps = tf.stack(raw_logits, axis=2)
        landmarks = tf.stack(coords, axis=-1)
        return heatmaps, landmarks

# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL
# ═════════════════════════════════════════════════════════════════════════════

class LandmarksHead(tf.keras.Model):
    def __init__(self, cfg, chekpoint_path: str = "./checkpoints"):
        super(LandmarksHead, self).__init__()
        self.cfg = cfg
        self.checkpoint_path = chekpoint_path

        self.vis_dir = Path(cfg.TRAIN.VIS_DIR)
        self.vis_dir.mkdir(exist_ok=True)

        self.body = MobileNetV3(
            mode=cfg.MODEL.MODE,
            num_classes=cfg.MODEL.NUM_LANDMARKS
        )
        self.head = HeatmapHead(
            input_shape=cfg.MODEL.IMAGE_SIZE,
            num_lm=cfg.MODEL.NUM_LANDMARKS,
        )
        dummy = tf.ones((1, 128, 128, 3))
        features = self.body(dummy)
        self.head(features)

        self.log_sigma = tf.Variable(
            tf.zeros([68]),
            trainable=True,
            name="log_sigma",
        )
        self.areas = tf.fill([cfg.TRAIN.BATCH_SIZE, cfg.MODEL.NUM_LANDMARKS], 1e-6)
        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.TRAIN.LEARNING_RATE)

    @property
    def sigmas(self):
        return tf.exp(self.log_sigma)  # avoid negatives values

    @tf.function
    def _train_step(self, images, gt_landmarks):  # seems pas ok.
        x_bins_norm = tf.linspace(-1., 1., self.cfg.MODEL.NUM_BINS)
        y_bins_norm = tf.linspace(-1., 1., self.cfg.MODEL.NUM_BINS)
        z_bins_range = tf.linspace(0., 1.1, self.cfg.MODEL.NUM_BINS)
        gt_lm_norm, bbox_cs, bbox_size, sigmas = self._preprocess_data(
            images, gt_landmarks)

        with tf.GradientTape() as tape:
            sigmas = self.sigmas
            gt_hm_x, gt_hm_y, gt_hm_z = generate_heatmaps(
                gt_lm_norm,
                sigmas,
                x_bins_norm, y_bins_norm, z_bins_range
            )
            gt_hm = tf.stack([gt_hm_x, gt_hm_y, gt_hm_z], axis=2)

            features = self.body(images, training=True)
            pred_heatmaps, pred_landmarks = self.head(features, training=True)

            loss_hm = compute_hm_loss(gt_hm, pred_heatmaps)
            loss_lm = compute_lm_loss(pred_landmarks, gt_lm_norm, sigmas)
            variance = tf.square(sigmas)
            loss_lm_reg = (loss_lm / (variance + 1e-6)) + (2.0 * self.log_sigma)
            loss_lm = tf.reduce_mean(loss_lm_reg)

            # Équilibrage dynamique
            lambda_lm = tf.stop_gradient(loss_hm / (loss_lm + 1e-8))
            loss = loss_hm + lambda_lm * loss_lm

        all_vars = self.trainable_variables
        gradients = tape.gradient(loss, all_vars)
        self.adam_optimizer.apply_gradients(zip(gradients, all_vars))

        return loss, loss_hm, loss_lm, pred_landmarks, pred_heatmaps, bbox_cs, bbox_size, gt_hm


    def _preprocess_data(self, images, landmarks):  # seems ok too.
        batch_size = tf.shape(images)[0]
        num_landmarks = tf.shape(landmarks)[1]
        tf.debugging.assert_equal(
            batch_size,
            tf.constant(self.cfg.TRAIN.BATCH_SIZE, dtype=tf.int32),
            message="Batch size mismatch"
        )
        tf.debugging.assert_equal(
            num_landmarks,
            tf.constant(self.cfg.MODEL.NUM_LANDMARKS, dtype=tf.int32),
            message="Num landmarks mismatch"
        )

        bbox_cs = build_bbox_cs(landmarks, scale=1.2)   # (center_x, center_y, width, height)
        bbox_size = bbox_cs[:, 2:]

        sigmas = tf.tile(self.sigmas[None], [batch_size, 1])  # build a sigma for each batch

        center = bbox_cs[:, :2]
        gt_xy_norm = (landmarks[..., :2] - center[:, None, :]) / bbox_size[:, None, :] # center & normalize between -1 & 1
        gt_z = tf.clip_by_value(landmarks[..., 2:3], 0.0, 1.1)

        gt_lm_norm = tf.concat([gt_xy_norm, gt_z], axis=-1)

        return gt_lm_norm, bbox_cs, bbox_size, sigmas

    def predict(self, images):
        pass

    def fit(self, train_dataset, epochs, resume=True, viz_training=False):  # seems ok.
        history = LandmarkTrainingHistory()
        start_epoch = load_latest_checkpoint(self.head, self.checkpoint_path) if resume else 0
        total_steps = len(train_dataset)

        for epoch in range(start_epoch, epochs):
            epoch_start = time()
            epoch_losses = []
            epoch_nmes = []
            for step, (images, landmarks) in enumerate(train_dataset):
                images_input = tf.image.resize(images, self.cfg.MODEL.IMAGE_SIZE)
                loss, loss_hm, loss_lm, pred_landmarks, pred_heatmaps, bbox_cs, bbox_size, gt_lm_norm\
                    = self._train_step(images_input, landmarks)
                epoch_losses.append(loss.numpy())

                nme = compute_nme(pred_landmarks, gt_lm_norm, bbox_cs=bbox_cs)
                epoch_nmes.append(float(tf.reduce_mean(nme).numpy()))

                print_step(epoch, epochs, step + 1, total_steps,
                           np.mean(epoch_losses), 'nme', np.mean(epoch_nmes),
                           time() - epoch_start)

                if viz_training & ((epoch * step + step) % self.cfg.TRAIN.VIS_EVERY_N_STEPS) == 0:
                    save_path = ( self.vis_dir / f"epoch_{epoch:03d}_step_{step:05d}.png")
                    visualize_side_by_side(
                        images_input[0].numpy(),
                        gt_lm_norm[0].numpy(),
                        pred_landmarks[0].numpy(),
                        pred_heatmaps[0, :, 0, :].numpy(),
                        pred_heatmaps[0, :, 1, :].numpy(),
                        bbox_cs[0].numpy(),
                        save_path=save_path
                    )

            epoch_loss = np.mean(epoch_losses)
            epoch_nme = np.mean(epoch_nmes)
            history.update(epoch_loss, epoch_nme)

            save_checkpoint(self.head, epoch, self.checkpoint_path)
            print_epoch(epoch, epochs, 'nme', *history.last(), time() - epoch_start)

        return history  # seems good

# ═════════════════════════════════════════════════════════════════════════════
# 3. BOX & HEATMAP
# ═════════════════════════════════════════════════════════════════════════════

def build_bbox_cs(landmarks_gt, scale=1.0):  # (center_x, center_y, width, height)
    landmarks_2d = landmarks_gt[:, :, :2]
    max_xy = tf.reduce_max(landmarks_2d, axis=1)
    min_xy = tf.reduce_min(landmarks_2d, axis=1)

    center = (max_xy + min_xy) / 2.0

    size = (max_xy - min_xy) * scale

    return tf.concat([center, size], axis=-1)

def _decode_prediction(x_bins_norm, y_bins_norm, center, scale):
    x_bins = x_bins_norm[None, None, :] * scale[:, None, 0:1] / 2 + center[:, None, 0:1]
    y_bins = y_bins_norm[None, None, :] * scale[:, None, 1:2] / 2 + center[:, None, 1:2]

    return x_bins, y_bins

def decode_xy(x_hms, y_hms, z_hms, x_bins, y_bins, z_bins):
    x = tf.reduce_sum(x_hms * tf.expand_dims(x_bins, axis=1), axis=-1)
    y = tf.reduce_sum(y_hms * tf.expand_dims(y_bins, axis=1), axis=-1)
    z = tf.reduce_sum(z_hms * tf.expand_dims(z_bins, axis=1), axis=-1)

    return tf.stack([x, y, z], axis=-1)

# normalized version
def generate_heatmaps(landmarks_gt, sigmas, x_bins_norm, y_bins_norm, z_bins_range):
    x_gt = landmarks_gt[..., 0:1]
    y_gt = landmarks_gt[..., 1:2]
    z_gt = landmarks_gt[..., 2:3]

    dist_x = tf.abs(x_gt - x_bins_norm)
    dist_y = tf.abs(y_gt - y_bins_norm)
    dist_z = tf.abs(z_gt - z_bins_range)

    sigmas = tf.clip_by_value(sigmas, 0.01, 0.5)

    dist_x = dist_x / sigmas[..., None]
    dist_y = dist_y / sigmas[..., None]
    dist_z = dist_z / sigmas[..., None]

    hm_x = tf.exp(-dist_x / 2.0) / sigmas[..., None]
    hm_y = tf.exp(-dist_y / 2.0) / sigmas[..., None]
    hm_z = tf.exp(-dist_z / 2.0) / sigmas[..., None]

    return hm_x, hm_y, hm_z

# Unnormalized version
# def generate_heatmaps(landmarks_gt, bbox_cs, sigmas, x_bins_norm, y_bins_norm, z_bins_range):
#     center = bbox_cs[:, :2]
#     scale = bbox_cs[:, 2:]
#
#     x_gt = landmarks_gt[..., 0:1]
#     y_gt = landmarks_gt[..., 1:2]
#     z_gt = landmarks_gt[..., 2:3]
#     z_gt = tf.clip_by_value(z_gt, 0.0, 1.1)  # pas sur
#
#     # x_bins, y_bins = _decode_prediction(x_bins_norm, y_bins_norm, center, scale)
#     dist_x = tf.abs(x_gt - x_bins)
#     dist_y = tf.abs(y_gt - y_bins)
#     dist_z = tf.abs(z_gt - z_bins_range)
#
#     areas = bbox_cs[:, 2] * bbox_cs[:, 3]
#     areas = tf.tile(areas[:, None], [1, tf.shape(landmarks_gt)[1]])
#     sigmas = tf.clip_by_value(sigmas, 0.01, 0.5)  # clip values between minimum and maximum avoiding extremums
#
#     areas = tf.sqrt(tf.maximum(areas, 1e-6))
#
#     dist_x = dist_x / areas[..., None] / sigmas[..., None]
#     dist_y = dist_y / areas[..., None] / sigmas[..., None]
#     dist_z = dist_z / areas[..., None] / sigmas[..., None]
#
#     hm_x = tf.exp(-dist_x / 2.0) / sigmas[..., None]
#     hm_y = tf.exp(-dist_y / 2.0) / sigmas[..., None]
#     hm_z = tf.exp(-dist_z / 2.0) / sigmas[..., None]
#
#     return hm_x, hm_y, hm_z

# ═════════════════════════════════════════════════════════════════════════════
# 4. LOSS
# ═════════════════════════════════════════════════════════════════════════════

def compute_hm_loss(target_hm, pred_hm):
    pred_hm = tf.nn.softmax(pred_hm, axis=-1)
    target_hm = target_hm / (tf.reduce_sum(target_hm, axis=-1, keepdims=True) + 1e-8)
    kl_loss = tf.keras.losses.KLDivergence(reduction='sum_over_batch_size', name='kl_divergence')
    loss_hm = kl_loss(target_hm, pred_hm)
    return loss_hm

# Wing loss
def compute_lm_loss(pred_lm, target_lm, sigmas):
    # pred, target : (B, N_landmarks, 3) pour x, y, z
    dist = tf.reduce_sum(pred_lm - target_lm)
    sigma_sq = tf.square(sigmas)[None]
    loss = dist / (2.0 * sigma_sq) + tf.math.log(sigmas)[None]
    return tf.reduce_mean(loss)

def compute_nme(pred_landmarks, gt_landmarks, bbox_cs):
    # pred/gt : (batch, num_lm, 2) — seulement X et Y
    bbox_cs = tf.cast(bbox_cs, tf.float32)

    cx = bbox_cs[:, 0]
    cy = bbox_cs[:, 1]
    w = bbox_cs[:, 2]
    h = bbox_cs[:, 3]

    gt_x = gt_landmarks[..., 0] * w[:, None] + cx[:, None]
    gt_y = gt_landmarks[..., 1] * h[:, None] + cy[:, None]
    pred_x = pred_landmarks[..., 0] * w[:, None] + cx[:, None]
    pred_y = pred_landmarks[..., 1] * h[:, None] + cy[:, None]

    diff = tf.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
    bbox_size = tf.sqrt(w * h)

    return tf.reduce_mean(diff / (bbox_size[:, None] + 1e-8))

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


if __name__ == '__main__':
    conf = load_config("config/config.yaml")
    afl_dataset = get_aflw2k3d_dataset(conf)
    train_dataset = afl_dataset.shuffle(1000).batch(conf.TRAIN.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    model = LandmarksHead(conf)
    model.fit(train_dataset, epochs=conf.TRAIN.EPOCHS, resume=True, viz_training=False)