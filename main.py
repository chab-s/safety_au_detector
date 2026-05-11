# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
from lib.training import (LandmarkTrainingHistory, save_checkpoint, load_latest_checkpoint,
                           print_step, print_epoch, plot_history)
from lib.vision import visualize_side_by_side
import time
from pathlib import Path

from config.config import load_config
from mobileNetV3.mobileNetV3 import MobileNetV3
from HeatmapHead.head import HeatmapHead, get_aflw2k3d_dataset, generate_heatmaps, compute_hm_loss, compute_lm_loss, compute_nme


def build_bbox_cs(landmarks_gt, scale=1.0):
    landmarks_2d = landmarks_gt[:, :, :2]
    max_xy = tf.reduce_max(landmarks_2d, axis=1)
    min_xy = tf.reduce_min(landmarks_2d, axis=1)

    center = (max_xy + min_xy) / 2.0

    half_size = ((max_xy - min_xy) / 2.0) * scale

    return tf.concat([center, half_size], axis=-1)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class LandmarksHead(tf.keras.Model):

    INPUT_SIZE = 128
    SCORE_THRESHOLD = 0.75
    NMS_THRESHOLD = 0.3
    LR = 1e-4
    CHECKPOINT_DIR = "checkpoints"

    def __init__(self, cfg, chekpoint_path: str = "./checkpoints"):
        super(LandmarksHead, self).__init__()
        self.cfg = cfg
        self.vis_dir = Path(cfg.TRAIN.VIS_DIR)
        self.vis_dir.mkdir(exist_ok=True)
        self.body = MobileNetV3(
            mode=cfg.MODEL.MODE,
            num_classes=cfg.MODEL.NUM_LANDMARKS
        )
        self.head = HeatmapHead(
            input_shape=cfg.MODEL.IMAGE_SIZE,
            num_lm=cfg.MODEL.NUM_LANDMARKS,
            num_bins=cfg.MODEL.NUM_BINS
        )
        dummy = tf.ones((1, 128, 128, 3))
        features = self.body(dummy)
        self.head(features)
        self.sigmas = tf.fill([cfg.TRAIN.BATCH_SIZE, cfg.MODEL.NUM_LANDMARKS], 1.0)
        self.areas = tf.fill([cfg.TRAIN.BATCH_SIZE, cfg.MODEL.NUM_LANDMARKS], 1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.checkpoint_path = chekpoint_path

    def predict(self, images):
        pass

    def fit(self, train_dataset, epochs, resume=True):
        history = LandmarkTrainingHistory()
        start_epoch = load_latest_checkpoint(self.head, self.checkpoint_path) if resume else 0
        total_steps = len(train_dataset)

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            epoch_losses = []
            epoch_nmes = []
            for step, (images, landmarks) in enumerate(train_dataset):
                loss, nme, pred_landmarks, pred_heatmaps, bbox_cs = self._train_step(images, landmarks)
                epoch_losses.append(loss.numpy())
                epoch_nmes.append(float(tf.reduce_mean(nme).numpy()))

                print_step(epoch, epochs, step + 1, total_steps,
                           np.mean(epoch_losses), 'nme', np.mean(epoch_nmes),
                           time.time() - epoch_start)
                if (epoch * step) % self.cfg.TRAIN.VIS_EVERY_N_STEPS == 0:
                    save_path = ( self.vis_dir / f"epoch_{epoch:03d}_step_{step:05d}.png")
                    visualize_side_by_side(
                        images[0].numpy(),
                        pred_landmarks[0].numpy(),
                        pred_heatmaps[0, :, 0, :].numpy(),
                        pred_heatmaps[0, :, 1, :].numpy(),
                        bbox_cs[0].numpy(),save_path=save_path
                    )

            epoch_loss = np.mean(epoch_losses)
            epoch_nme = np.mean(epoch_nmes)
            history.update(epoch_loss, epoch_nme)

            save_checkpoint(self.head, epoch, self.checkpoint_path)
            print_epoch(epoch, epochs, 'nme', *history.last(), time.time() - epoch_start)

        return history

    @tf.function
    def _train_step(self, images, gt_landmarks):
        images = tf.image.resize(images, [128, 128])
        with tf.GradientTape() as tape:
            features = self.body(images, training=True)
            pred_landmarks, pred_heatmaps = self.head(features, training=True)
            bbox_cs, bbox_size, sigmas, areas, x_bins_base, y_bins_base, z_bins_base = self.preprocess_data(images, gt_landmarks)
            gt_hm_x, gt_hm_y, gt_hm_z = generate_heatmaps(
                                    gt_landmarks, bbox_cs, sigmas, areas,
                                    x_bins_base, y_bins_base, z_bins_base
                                )
            gt_hm = tf.stack([gt_hm_x, gt_hm_y, gt_hm_z], axis=2)
            loss_hm = compute_hm_loss(gt_hm, pred_heatmaps, images)

        gradients = tape.gradient(loss_hm, self.head.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.head.trainable_variables))

        nme = compute_nme(pred_landmarks, gt_landmarks, bbox_size=bbox_size)

        return loss_hm, nme, pred_landmarks, pred_heatmaps, bbox_cs

    def preprocess_data(self, images, landmarks):
        batch_size = tf.shape(images)[0]
        num_landmarks = tf.shape(landmarks)[1]
        bbox_cs = build_bbox_cs(landmarks, scale=1.2)
        bbox_size = tf.reduce_mean(bbox_cs[:, 3:], axis=-1)
        sigmas = tf.fill([batch_size, num_landmarks], 1.0)
        areas = tf.fill([batch_size, num_landmarks], 1.0)
        # areas = tf.tile(tf.expand_dims(bbox_size, 1), [1, num_landmarks])
        num_bins = 64
        x_bins_base = tf.linspace(-1., 1., num_bins)
        y_bins_base = tf.linspace(-1., 1., num_bins)
        z_bins_base = tf.linspace(0., 1.1, num_bins)

        return bbox_cs, bbox_size, sigmas, areas, x_bins_base, y_bins_base, z_bins_base


if __name__ == '__main__':
    conf = load_config("config/config.yaml")
    afl_dataset = get_aflw2k3d_dataset(conf)
    train_dataset = afl_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    model = LandmarksHead(conf)
    model.fit(train_dataset, epochs=conf.TRAIN.EPOCHS, resume=True)