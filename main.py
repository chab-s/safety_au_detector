# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import cv2
from lib.vision import visualize_side_by_side
from lib.utils import ModelInspector

from config.config import load_config
from blazeface.blazeface import BlazeModel, compute_bbox_loss, build_anchors, _apply_nms, _decode_boxes
from mobileNetV3.mobileNetV3 import MobileNetV3
from HeatmapHead.head import HeatmapHead, get_aflw2k3d_dataset, generate_heatmaps, compute_hm_loss, compute_lm_loss


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

    def __init__(self, cfg, chekpoint_path: str = "./checkpoints"):
        super(LandmarksHead, self).__init__()
        self.cfg = cfg
        self.backbone = BlazeModel(backbone_mode=False)
        self.anchors = build_anchors()
        self.body = MobileNetV3(
            mode=cfg.MODEL.MODE,
            num_classes=cfg.MODEL.NUM_LANDMARKS
        )
        self.head = HeatmapHead(
            input_shape=cfg.MODEL.IMAGE_SIZE,
            num_lm=cfg.MODEL.NUM_LANDMARKS,
            num_bins=cfg.MODEL.NUM_BINS
        )
        self.sigmas = tf.fill([cfg.TRAIN.BATCH_SIZE, cfg.MODEL.NUM_LANDMARKS], 1.0)
        self.areas = tf.fill([cfg.TRAIN.BATCH_SIZE, cfg.MODEL.NUM_LANDMARKS], 1.0)
        self.optimizer = None
        self.checkpoint_path = chekpoint_path

    def predict(self, images):
        pass

    def _train_step(self, images, grid_small=None, grid_large=None, landmarks=None):
        images = tf.image.resize(images, [128, 128])
        with tf.GradientTape() as tape:
            features, scores, pred_offsets = self.backbone(images, training=False)
            features = self.body(features, training=True)
            pred_landmarks = self.head(features, training=True)

            x_bins_base = tf.linspace(-1., 1., self.cfg.MODEL.NUM_BINS)
            y_bins_base = tf.linspace(-1., 1., self.cfg.MODEL.NUM_BINS)
            z_bins_base = tf.linspace(-1., 1., self.cfg.MODEL.NUM_BINS)
            # pred_bbox_cs = _decode_boxes(pred_offsets, self.anchors, self.cfg.MODEL.IMAGE_SIZE)
            pred_bbox_cs = _apply_nms(
                scores, pred_offsets, self.anchors, self.cfg.MODEL.IMAGE_SIZE, self.SCORE_THRESHOLD, self.NMS_THRESHOLD
            )
            print(np.array(pred_bbox_cs).shape, pred_landmarks.shape)
            hm_x, hm_y, hm_z = generate_heatmaps(
                pred_landmarks, pred_bbox_cs, self.sigmas, self.areas,
                x_bins_base, y_bins_base, z_bins_base
            )
            print(hm_x, hm_y, hm_z)
            exit()
            bbox_loss = compute_bbox_loss(scores, bbox_cs, grid_small=grid_small, grid_large=grid_large, anchors=self.anchors, image_size=self.cfg.MODEL.IMAGE_SIZE)
            heatmap_loss = compute_hm_loss()
            landamrks_loss = compute_lm_loss()

            all_vars = self.backbone.trainable_variables + \
                       self.body.trainable_variables + \
                       self.head.trainable_variables

            total_loss = None
            gradient = tape.gradient(total_loss, features)
            self.optimizer.apply_gradients(zip(gradient, all_vars))
            exit()


if __name__ == '__main__':
    conf = load_config("config/config.yaml")
    afl_dataset = get_aflw2k3d_dataset(conf)
    dataset = afl_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    model = LandmarksHead(conf)
    for images, landmarks in dataset.take(1):
        model._train_step(images)

    # afl_dataset = get_aflw2k3d_dataset(conf)
    # dataset = afl_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    # blaze_backbone = BlazeModel(backbone_mode=True)
    # mobileNet_body = MobileNetV3(mode='small', num_classes=68)
    # heatmap_head = HeatmapHead(input_shape=(576,), num_lm=68, num_bins=64)
    #
    # for images, landmarks in dataset.take(1):
    #     print("landmarks min/max:", landmarks.numpy().min(), landmarks.numpy().max())
    #     print(landmarks.shape)
    #     batch_size = tf.shape(images)[0]
    #     num_landmarks = tf.shape(landmarks)[1]
    #     bbox_cs = build_bbox_cs(landmarks, scale=1.2)
    #     bbox_size = tf.reduce_mean(bbox_cs[:, 3:], axis=-1)
    #     sigmas = tf.fill([batch_size, num_landmarks], 1.0)
    #     areas = tf.fill([batch_size, num_landmarks], 1.0)
    #     # areas = tf.tile(tf.expand_dims(bbox_size, 1), [1, num_landmarks])
    #     num_bins = 64
    #     x_bins_base = tf.linspace(-1., 1., num_bins)
    #     y_bins_base = tf.linspace(-1., 1., num_bins)
    #     z_bins_base = tf.linspace(-1., 1., num_bins)
    #
    #     hm_x, hm_y, hm_z = generate_heatmaps(
    #         landmarks, bbox_cs, sigmas, areas,
    #         x_bins_base, y_bins_base, z_bins_base
    #     )
    #
    #     visualize_side_by_side(
    #         images[0].numpy(), landmarks[0].numpy(), hm_x[0].numpy(), hm_y[0].numpy(), bbox_cs[0].numpy()
    #     )