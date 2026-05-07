# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import cv2
from config.config import load_config
from blazeface.blazeface import BlazeModel
from mobileNetV3.mobileNetV3 import MobileNetV3
from HeatmapHead.head import HeatmapHead, get_aflw2k3d_dataset, generate_heatmaps


def build_bbox_cs(landmarks_gt, scale=1.0):
    """
    landmarks_gt: (B, K, 3)
    returns: (B, 6) → [cx, cy, cz, w, h, d]
    """
    max_xyz = tf.reduce_max(landmarks_gt, axis=1)
    min_xyz = tf.reduce_min(landmarks_gt, axis=1)

    center = (max_xyz + min_xyz) / 2.0

    half_size = ((max_xyz - min_xyz) / 2.0) * scale

    return tf.concat([center, half_size], axis=-1)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_all_heatmaps(image, lms, hms_x, hms_y, single_bbox, alpha=0.5):
    # 1. Extraction des coordonnées (format centre, demi-taille)
    cx, cy = single_bbox[0], single_bbox[1]
    sx, sy = single_bbox[3], single_bbox[4]

    x0, x1 = cx - sx, cx + sx
    y0, y1 = cy - sy, cy + sy
    width, height = 2 * sx, 2 * sy

    H, W = 64, 64  # Taille de tes heatmaps
    combined_hm = np.zeros((H, W))

    for i in range(hms_x.shape[0]):
        single_lm_hm = np.outer(hms_y[i], hms_x[i])

        # On fusionne avec la heatmap globale (soit par somme, soit par max)
        combined_hm = np.maximum(combined_hm, single_lm_hm)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.imshow(combined_hm, cmap='jet', alpha=alpha,
              extent=[x0, x1, y1, y0],  # [gauche, droite, bas, haut]
              aspect='auto')
    rect = patches.Rectangle((x0, y0), width, height,
                             linewidth=2, edgecolor='r', facecolor='none', label='BBox')
    ax.add_patch(rect)
    ax.scatter(lms[:, 0], lms[:, 1], c='lime', s=10, label='Landmarks')

    plt.legend()
    plt.show()


def visualize_side_by_side(image, lms, hms_x, hms_y, single_bbox, alpha=0.5):
    # 1. Calcul des coordonnées de la BBox
    cx, cy = single_bbox[0], single_bbox[1]
    sx, sy = single_bbox[3], single_bbox[4]

    x0, x1 = cx - sx, cx + sx
    y0, y1 = cy - sy, cy + sy
    width, height = 2 * sx, 2 * sy

    # 2. Construction de la heatmap combinée (Correction de l'effet de grille)
    H, W = 64, 64
    combined_hm = np.zeros((H, W))
    for i in range(hms_x.shape[0]):
        single_lm_hm = np.outer(hms_y[i], hms_x[i])
        combined_hm = np.maximum(combined_hm, single_lm_hm)

    # 3. Création de la figure avec 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Plot 1 : Image brute avec Landmarks et BBox ---
    ax1.imshow(image)
    ax1.scatter(lms[:, 0], lms[:, 1], c='lime', s=10, label='Landmarks')
    rect1 = patches.Rectangle((x0, y0), width, height, linewidth=2,
                              edgecolor='r', facecolor='none', label='BBox')
    ax1.add_patch(rect1)
    ax1.set_title("Image & Ground Truth")
    ax1.legend()

    # --- Plot 2 : Heatmap superposée ---
    ax2.imshow(image)
    im2 = ax2.imshow(combined_hm, cmap='jet', alpha=alpha,
                     extent=[x0, x1, y1, y0],
                     aspect='auto')

    rect2 = patches.Rectangle((x0, y0), width, height, linewidth=2,
                              edgecolor='r', facecolor='none')
    ax2.add_patch(rect2)
    ax2.set_title("Heatmap Overlay")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    conf = load_config("config/config.yaml")

    afl_dataset = get_aflw2k3d_dataset(conf)
    dataset = afl_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    blaze_backbone = BlazeModel(backbone_mode=True)
    mobileNet_body = MobileNetV3(mode='large', num_classes=68)
    heatmap_head = HeatmapHead(input_shape=(960,), num_kp=68, num_bins=64)

    for images, landmarks in dataset.take(1):
        print("landmarks min/max:", landmarks.numpy().min(), landmarks.numpy().max())
        print(landmarks.shape)
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
        z_bins_base = tf.linspace(-1., 1., num_bins)

        hm_x, hm_y, hm_z = generate_heatmaps(
            landmarks, bbox_cs, sigmas, areas,
            x_bins_base, y_bins_base, z_bins_base
        )

        visualize_side_by_side(
            images[0].numpy(), landmarks[0].numpy(), hm_x[0].numpy(), hm_y[0].numpy(), bbox_cs[0].numpy()
        )