import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tensorflow.keras import layers

from lib.vision   import compute_iou, mean_iou, nms, crop_and_resize_with_boxes
from lib.metrics  import evaluate_detections, DetectionMetrics
from lib.training import (TrainingHistory, save_checkpoint, load_latest_checkpoint,
                           print_step, print_epoch, plot_history)

# ═════════════════════════════════════════════════════════════════════════════
# 1. BASE BLOCKS
# ═════════════════════════════════════════════════════════════════════════════

class BlazeBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, use_pool=True, **kwargs):
        super(BlazeBlock, self).__init__(**kwargs)
        self.strides = strides
        self.filters = filters
        self.use_pool = use_pool

        if use_pool:
            self.skip = layers.MaxPool2D(pool_size=2, strides=strides, padding='same')
            self.channel_pad = layers.Lambda(lambda x: self._pad_channels(x, filters))

        self.dw_conv = layers.DepthwiseConv2D((5, 5), strides=strides, padding='same')
        self.conv = layers.Conv2D(filters, (1, 1), strides=(1, 1))

        self.norm_1 = layers.BatchNormalization()
        self.norm_2 = layers.BatchNormalization()

        self.activation = layers.ReLU()

    def _pad_channels(self, x, target_channels):
        current_channels = x.shape[-1]
        if current_channels is None:
            return x

        channels_to_add = target_channels - current_channels
        if channels_to_add <= 0:
            return x

        paddings = [[0, 0], [0, 0], [0, 0], [0, channels_to_add]]
        return tf.pad(x, paddings, mode='CONSTANT', constant_values=0)

    def call(self, inputs, training=False):
        x = self.dw_conv(inputs)
        x = self.norm_1(x)
        x = self.conv(x)
        x = self.norm_2(x)

        if self.use_pool:
            skip = self.skip(inputs)
            skip = self.channel_pad(skip)
            x = x + skip

        x = self.activation(x)
        return x


class DoubleBlazeBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides, use_pool=True, **kwargs):
        super(DoubleBlazeBlock, self).__init__(**kwargs)

        self.filters = filters
        self.strides = strides
        self.use_pool = use_pool

        if use_pool:
            self.skip = layers.MaxPool2D(pool_size=2, strides=strides, padding='same')
            self.channel_pad = layers.Lambda(lambda x: self._pad_channels(x, filters))

        self.dw_conv_1 = layers.DepthwiseConv2D((5, 5), strides=strides, padding='same')
        self.dw_conv_2 = layers.DepthwiseConv2D((5, 5), strides=(1, 1), padding='same')

        self.conv_project = layers.Conv2D(self.filters, (1, 1), strides=(1, 1))
        self.conv_expand = layers.Conv2D(self.filters, (1, 1), strides=(1, 1))

        self.norm_1 = layers.BatchNormalization()
        self.norm_2 = layers.BatchNormalization()
        self.norm_3 = layers.BatchNormalization()
        self.norm_4 = layers.BatchNormalization()

        self.activation_1 = layers.ReLU()
        self.activation_2 = layers.ReLU()

    def _pad_channels(self, x, target_channels):
        current_channels = x.shape[-1]
        if current_channels is None:
            return x

        channels_to_add = target_channels - current_channels
        if channels_to_add <= 0:
            return x

        paddings = [[0, 0], [0, 0], [0, 0], [0, channels_to_add]]
        return tf.pad(x, paddings, mode='CONSTANT', constant_values=0)

    def call(self, inputs, training=False):
        x = self.dw_conv_1(inputs)
        x = self.norm_1(x)
        x = self.conv_project(x)
        x = self.norm_2(x)

        x = self.activation_1(x)

        x = self.dw_conv_2(x)
        x = self.norm_3(x)
        x = self.conv_expand(x)
        x = self.norm_4(x)

        if self.use_pool:
            skip = self.skip(inputs)
            skip = self.channel_pad(skip)
            x = x + skip

        x = self.activation_2(x)

        return x


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL
#
# Input : image 128x128x3
# Output : scores [B, 896, 1] + boxes [B, 896, 4]
#          896 = 512 anchors small (16x16x2) + 384 anchors big (8x8x6)
# ═════════════════════════════════════════════════════════════════════════════

# %% md
## Model
# %%
# class BlazeModel(tf.keras.Model):
#     def __init__(self, **kwargs):
#         super(BlazeModel, self).__init__(**kwargs)
#
#         self.stem = layers.Conv2D(24, (5, 5), strides=2, padding='same')
#         self.stem_relu = layers.ReLU()
#
#         self.encoder_small = tf.keras.Sequential([
#             BlazeBlock(filters=24, strides=1),
#             BlazeBlock(filters=24, strides=1),
#             BlazeBlock(filters=48, strides=2),
#             BlazeBlock(filters=48, strides=1),
#             BlazeBlock(filters=48, strides=1)
#         ], name="encoder_small")
#
#         self.encoder_large = tf.keras.Sequential([
#             DoubleBlazeBlock(filters=96, strides=2),
#             DoubleBlazeBlock(filters=96, strides=1),
#             DoubleBlazeBlock(filters=96, strides=1),
#             DoubleBlazeBlock(filters=96, strides=2),
#             DoubleBlazeBlock(filters=96, strides=1),
#             DoubleBlazeBlock(filters=96, strides=1)
#         ], name="encoder_large")
#
#         self.scores_small = layers.Conv2D(2, (1, 1), strides=(1, 1), activation='sigmoid')
#         self.scores_large = layers.Conv2D(6, (1, 1), strides=(1, 1), activation='sigmoid')
#
#         self.boxes_small = layers.Conv2D(8, (1, 1), strides=(1, 1))
#         self.boxes_large = layers.Conv2D(24, (1, 1), strides=(1, 1))
#
#     def call(self, inputs, training=False):
#         x = self.stem_relu(self.stem(inputs))
#
#         features_16x16 = self.encoder_small(x, training=training)
#         features_8x8 = self.encoder_large(features_16x16, training=training)
#
#         scores = self._merge_heads(
#             self.scores_small(features_16x16),
#             self.scores_large(features_8x8),
#             last_dim=1
#         )
#         boxes = self._merge_heads(
#             self.boxes_small(features_16x16),
#             self.boxes_large(features_8x8),
#             last_dim=4
#         )
#
#         return scores, boxes
#
#     def _merge_heads(self, small, large, last_dim):
#         return layers.concatenate([
#             layers.Reshape((-1, last_dim))(small),
#             layers.Reshape((-1, last_dim))(large),
#         ], axis=1)

class BlazeModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(BlazeModel, self).__init__(**kwargs)

        self.conv = layers.Conv2D(24, (5, 5), strides=2, padding='same')
        self.activation = layers.ReLU()

        self.block1 = BlazeBlock(filters=24, strides=1)
        self.block2 = BlazeBlock(filters=24, strides=1)
        self.block3 = BlazeBlock(filters=48, strides=2)
        self.block4 = BlazeBlock(filters=48, strides=1)
        self.block5 = BlazeBlock(filters=48, strides=1)

        self.block6 = DoubleBlazeBlock(filters=96, strides=2)
        self.block7 = DoubleBlazeBlock(filters=96, strides=1)
        self.block8 = DoubleBlazeBlock(filters=96, strides=1)
        self.block9 = DoubleBlazeBlock(filters=96, strides=2)
        self.block10 = DoubleBlazeBlock(filters=96, strides=1)
        self.block11 = DoubleBlazeBlock(filters=96, strides=1)

        self.classifier_8 = layers.Conv2D(2, (1, 1), strides=(1, 1), activation='sigmoid')
        self.classifier_16 = layers.Conv2D(6, (1, 1), strides=(1, 1), activation='sigmoid')

        self.regressor_8 = layers.Conv2D(8, (1, 1), strides=(1, 1))
        self.regressor_16 = layers.Conv2D(24, (1, 1), strides=(1, 1))

    def call(self, inputs):
        # tf.print("Input size conv:", tf.shape(inputs))
        x = self.conv(inputs)
        # tf.print("Input size ReLu:", tf.shape(x))
        x = self.activation(x)

        # tf.print("Input size Single BlazeBlock_1:", tf.shape(x))
        x = self.block1(x)
        # tf.print("Input size Single BlazeBlock_2:", tf.shape(x))
        x = self.block2(x)
        # tf.print("Input size Single BlazeBlock_3:", tf.shape(x))
        x = self.block3(x)
        # tf.print("Input size Single BlazeBlock_4:", tf.shape(x))
        x = self.block4(x)
        # tf.print("Input size Single BlazeBlock_5:", tf.shape(x))
        x = self.block5(x)

        # tf.print("Input size Double BlazeBlock_1:", tf.shape(x))
        x = self.block6(x)
        # tf.print("Input size Double BlazeBlock_2:", tf.shape(x))
        x = self.block7(x)
        # tf.print("Input size Double BlazeBlock_3:", tf.shape(x))
        x = self.block8(x)
        # tf.print("Input size Double BlazeBlock_4:", tf.shape(x))
        h = self.block9(x)
        # tf.print("Input size Double BlazeBlock_5:", tf.shape(h))
        h = self.block10(h)
        # tf.print("Input size Double BlazeBlock_6:", tf.shape(h))
        h = self.block11(h)

        # tf.print("Input size classifier_8:", tf.shape(x))
        c1 = self.classifier_8(x)
        # tf.print("Input size reshape:", tf.shape(c1))
        c1 = layers.Reshape((-1, 1))(c1)

        # tf.print("Input size classifier_16:", tf.shape(h))
        c2 = self.classifier_16(h)
        # tf.print("Input size reshape:", tf.shape(c2))
        c2 = layers.Reshape((-1, 1))(c2)

        c = layers.concatenate([c1, c2], axis=1)
        # tf.print("Output size classifier_concat:", tf.shape(c))

        # tf.print("Input size regressor_8:", tf.shape(x))
        r1 = self.regressor_8(x)
        # tf.print("Input size reshape:", tf.shape(r1))
        r1 = layers.Reshape((-1, 4))(r1)

        # tf.print("Input size regressor_16:", tf.shape(h))
        r2 = self.regressor_16(h)
        # tf.print("Input size reshape:", tf.shape(r2))
        r2 = layers.Reshape((-1, 4))(r2)

        r = layers.concatenate([r1, r2], axis=1)
        # tf.print("Output size regressor_concat:", tf.shape(r))

        return c, r


# ═════════════════════════════════════════════════════════════════════════════
# 3. ANCHORS
#
# BlazeFace utilise des anchors fixes : une grille 16x16 (small) et 8x8 (big).
# Chaque cellule couvre une région de l'image — les anchors sont les centres
# de ces régions. On en génère 896 au total.
# ═════════════════════════════════════════════════════════════════════════════

def build_anchors():
    small_coords = np.linspace(0.03125, 0.96875, 16, endpoint=True, dtype=np.float32)  # 16x16
    large_coords = np.linspace(0.0625, .9375, 8, endpoint=True, dtype=np.float32)  # 8x8

    small_anchors = _make_anchor_grid(small_coords, anchors_per_cell=2)
    large_anchors = _make_anchor_grid(large_coords, anchors_per_cell=6)

    return tf.concat([small_anchors, large_anchors], axis=0)

def _make_anchor_grid(coords, anchors_per_cell):
    x = tf.tile(tf.repeat(coords, repeats=anchors_per_cell), [len(coords)])
    y = tf.repeat(coords, repeats=len(coords)*anchors_per_cell)
    return tf.stack([x, y], axis=1)

def assign_anchors_to_boxes(boxes, anchors, image_size: int = 128):
    n_small = 16 * 16 * 2
    small_anchors = anchors[:n_small]
    large_anchors = anchors[n_small:]

    grid_small = np.zeros((16, 16, 5), dtype=np.float32)
    grid_large = np.zeros((8, 8, 5), dtype=np.float32)

    for box in boxes:
        _fill_anchor_grid(box, small_anchors, grid_small, grid_size=16, anchors_per_cell=2, image_size=image_size)
        _fill_anchor_grid(box, large_anchors, grid_large, grid_size=8, anchors_per_cell=6, image_size=image_size)

    return grid_small, grid_large

def _fill_anchor_grid(box, anchors, grid, grid_size, anchors_per_cell, image_size):
    step = 1.0 / grid_size
    best_iou = -1

    for i, anchor_center in enumerate(anchors.numpy()):
        cx, cy = anchor_center
        anchor_box = np.array([cx - step, cy - step, cx + step, cy + step])
        iou = compute_iou(box * image_size, anchor_box * image_size)

        if iou >= best_iou:
            best_iou = iou
            row = int(i // (grid_size * anchors_per_cell))
            col = int(i % (grid_size * anchors_per_cell) // anchors_per_cell)

    grid[row, col] = [1, *box]

def shuffle_anchors(grid_small, grid_large):
    return (
        _shuffle_grid(grid_small, slots=2, slot_size=5),
        _shuffle_grid(grid_large, slots=6, slot_size=5),
    )

def _shuffle_grid(grids, slots, slot_size):
    result = []
    total = slots * slot_size

    for grid in grids:
        slot = tf.random.uniform([], 0, slots, dtype=tf.int32)
        offset = slot * slot_size
        h, w, _ = grid.shape

        left = tf.zeros((h, w, offset))
        right = tf.zeros((h, w, total - offset - slot_size))
        result.append(tf.concat([left, grid, right], axis=-1))

    return tf.stack(result, axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# 4. LOSS
#
# Deux composantes :
#   - Classification : binary cross-entropy avec hard negative mining
#     (on garde 3x plus de négatifs que de positifs, les plus difficiles)
#   - Régression     : Huber loss sur les coordonnées des boîtes positives
#
# Poids issus du papier BlazeFace (section 3.2).
# ═════════════════════════════════════════════════════════════════════════════

REGRESSION_WEIGHT = 150
CLASSIFICATION_WEIGHT = 35
HARD_BACKGROUND_RATIO = 3

huber_loss = tf.keras.losses.Huber()

def compute_loss(predicted_scores, predicted_boxes, grid_small, grid_large, anchors, image_size: int = 128):
    true_boxes, true_classes = _extract_ground_truth(grid_small, grid_large)
    face_mask = tf.dtypes.cast(true_classes, tf.bool)
    cls_loss = _classification_loss(predicted_scores, true_classes, face_mask)
    reg_loss, pred_coords, true_coords = _regression_loss(
        predicted_boxes, true_boxes, face_mask, anchors, image_size
    )

    total_loss = reg_loss * REGRESSION_WEIGHT + cls_loss * CLASSIFICATION_WEIGHT
    return total_loss, pred_coords, true_coords

def _extract_ground_truth(grid_small, grid_large):
    B = grid_small.shape[0]
    flat_small = tf.reshape(grid_small, (B, -1, 5))
    flat_large = tf.reshape(grid_large, (B, -1, 5))
    all_anchors = tf.concat([flat_small, flat_large], axis=1)

    true_coords = all_anchors[:, :, 1:]
    true_classes = all_anchors[:, :, 0]

    return true_coords, true_classes

def _classification_loss(predicted_scores, true_classes, face_mask):
    scores = tf.squeeze(predicted_scores, axis=-1)
    n_positives = int(tf.reduce_sum(true_classes))
    n_background = n_positives * HARD_BACKGROUND_RATIO

    background_scores = tf.where(face_mask, -99.0, scores)
    hard_background = tf.sort(background_scores, axis=-1, direction='DESCENDING')[:, :n_background
    ]
    positives_scores = tf.boolean_mask(scores, face_mask)

    background_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        tf.zeros_like(hard_background), hard_background
    ))
    positive_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
        tf.ones_like(positives_scores), positives_scores
    ))

    return background_loss + positive_loss

def _regression_loss(predicted_boxes, true_boxes, face_mask, anchors, image_size):
    decoded_boxes = _decode_boxes(predicted_boxes, anchors, image_size)

    true_coords = tf.boolean_mask(true_boxes, face_mask)
    pred_coords = tf.boolean_mask(decoded_boxes, face_mask)

    loss = huber_loss(true_coords, pred_coords)
    return loss, pred_coords, true_coords

def _decode_boxes(predicted_offsets, anchors, image_size):
    cx = anchors[:, 0:1] + predicted_offsets[..., 0:1] / image_size
    cy = anchors[:, 1:2] + predicted_offsets[..., 1:2] / image_size
    w = predicted_offsets[..., 2:3] / image_size
    h = predicted_offsets[..., 3:4] / image_size

    x_min = cx - w / 2.
    y_min = cy - h / 2.
    x_max = cx + w / 2.
    y_max = cy + h / 2.

    return tf.concat([x_min, y_min, x_max, y_max], axis=-1)


# ═════════════════════════════════════════════════════════════════════════════
# 5. DATASET
#
# Pipeline :
#   CSV avec chemins + boîtes → recadrage aléatoire carré → resize 128x128
#   → augmentations (flip, saturation, luminosité) → assignation aux anchors
# ═════════════════════════════════════════════════════════════════════════════

def set_dataset(csv_path, output_dir, anchors):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0)[['group', 'image_path', 'x1', 'y1', 'w', 'h']]

    image_paths = []
    boxes_dict = []

    for img_path, indices in df.groupby("image_path").groups.items():
        group = df.loc[indices].values[0][0]
        boxes = df.loc[indices].values[:, 2:]
        image = cv2.imread(f"face_dataset/{img_path}")

        for variant in ["set-1", "set-2"]:
            clean_path = img_path.replace("/", "-") if group == "None" else img_path.split("/")[-1]
            output_path = f"{output_dir}/{variant}-{clean_path}"

            resized, new_boxes = crop_and_resize_with_boxes(image, boxes)
            cv2.imwrite(output_path, resized)

            image_paths.append(output_path)
            boxes_dict[output_path] = [np.array(b, dtype=np.float32) for b in new_boxes]

        if len(image_paths) % 500 == 0:
            print(f"  {len(image_paths)} images traitées...")

        return np.array(image_paths), boxes_dict

def build_dataset(image_paths, boxes_dict, anchors, augment=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    if augment:
        dataset = dataset.shuffle(len(image_paths))
        dataset = dataset.map(
            lambda path: _load_and_augment(path, boxes_dict, anchors),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda path: _load_image(path, boxes_dict, anchors),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

def _load_and_augment(image_path, boxes_dict, anchors):
    image = _read_image(image_path)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, 0.2)

    flip = tf.random.uniform([]) > 0.5
    if flip:
        image = tf.image.flip_left_right(image)

    grid_small, grid_large = tf.py_function(
        lambda p, f: _build_anchor_grids(p, f, boxes_dict, anchors),
        [image_path, flip],
        [tf.float32, tf.float32]
    )

    return image, grid_small, grid_large, image_path, flip

def _load_image(image_path, boxes_dict, anchors):
    image = _read_image(image_path)
    grid_small, grid_large = tf.py_function(
        lambda p: _build_anchor_grids(p, False, boxes_dict, anchors),
        [image_path],
        [tf.float32, tf.float32]
    )
    return image_path, image, grid_small, grid_large


def _read_image(image_path):
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [128, 128])


def _build_anchor_grids(image_path, flip, boxes_dict, anchors):
    """Construit les grilles d'anchors pour une image (appelé via py_function)."""
    key = image_path.numpy().decode("utf-8")
    boxes = boxes_dict[key]

    if bool(flip):
        boxes = [_flip_box_horizontal(b) for b in boxes]

    return assign_anchors_to_boxes(boxes, anchors)


def _flip_box_horizontal(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    return np.array([1 - x1 - w, y1, 1 - x1, y2], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 6. DÉTECTEUR COMPLET
#
# BlazeFaceDetector regroupe modèle + anchors + pipeline en une interface simple.
# Usage :
#   detector = BlazeFaceDetector()
#   history  = detector.fit(train_dataset, epochs=100)
#   metrics  = detector.evaluate(test_dataset)
#   boxes    = detector.predict(images)
# ═════════════════════════════════════════════════════════════════════════════

class BlazeFaceDetector:

    INPUT_SIZE = 128
    SCORE_THRESHOLD = 0.75
    NMS_THRESHOLD = 0.3

    def __init__(self, lr=1e-4, checkpoint_path="./checkpoints"):
        self.anchors = build_anchors()
        self.model = BlazeModel()
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.checkpoint_path = checkpoint_path

        dummy = tf.ones((1, self.INPUT_SIZE, self.INPUT_SIZE, 3))
        self.model(dummy)

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, images):
        scores, boxes = self.model.predict(images)
        return self._apply_nms(scores, boxes)

    def _apply_nms(self, scores, boxes):
        decoded = _decode_boxes(boxes, self.anchors, self.INPUT_SIZE)
        scores_flat = tf.squeeze(scores, axis=-1)
        detections = []

        for i in range(scores_flat.shape[0]):
            mask = tf.where(scores_flat[i] >= self.SCORE_THRESHOLD)[:, 0]
            candidates = [
                tf.concat([scores_flat[i, j:j + 1], decoded[i, j]], axis=0)
                for j in mask
            ]
            detections.append(nms(candidates, self.NMS_THRESHOLD))

        return detections

    # ── Training ──────────────────────────────────────────────────────

    def fit(self, train_dataset, epochs, resume=True):
        history = TrainingHistory()
        start_epoch = load_latest_checkpoint(self.model, self.checkpoint_path) if resume else 0
        total_steps = len([train_dataset])

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            for step, (images, grid_small, grid_large, _, _) in enumerate(train_dataset):
                grid_small, grid_large = shuffle_anchors(grid_small, grid_large)
                loss, pred_coords, true_coords = self._train_step(images, grid_small, grid_large)

                iou = mean_iou(true_coords * self.INPUT_SIZE, pred_coords * self.INPUT_SIZE)
                history.update(loss, iou)

                print_step(epoch, epochs, step + 1, total_steps, *history.last(), time.time() - epoch_start)

            save_checkpoint(self.model, epoch, self.checkpoint_path)
            print_epoch(epoch, epochs, *history.last(), time.time() - epoch_start)

        return history

    @tf.function
    def _train_step(self, images, grid_small, grid_large):
        with tf.GradientTape() as tape:
            scores, boxes = self.model(images, training=True)
            loss, pred_coords, true_coords = compute_loss(
                scores, boxes, grid_small, grid_large, self.anchors, self.INPUT_SIZE
            )
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, pred_coords, true_coords

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, test_dataset):
        all_pred, all_true = [], []

        for _, images, grid_small, grid_large in test_dataset:
            detections = self.predict(images)
            true_boxes, _ = _extract_ground_truth(grid_small, grid_large)
            face_mask = tf.reduce_sum(true_boxes, axis=-1) > 0
            for i, detection in enumerate(detections):
                pred_boxes = tf.convert_to_tensor([d[1:] for d in detection], dtype=tf.float32)
                true = tf.boolean_mask(true_boxes[i], face_mask[i])
                all_pred.append(pred_boxes)
                all_true.append(true)
        metrics = evaluate_detections(all_pred, all_true)
        return metrics

    # ── Visualisation ─────────────────────────────────────────────────────

    def visualize(self, images, n=8):
        detections = self.predict(images[:n])
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for ax, image, image_detections in zip(axes.flatten(), images, detections):
            ax.imshow(image.numpy())
            for det in image_detections:
                x1, y1, x2, y2 = det[1:].numpy() * self.INPUT_SIZE
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                ))
            ax.axis('off')

        plt.suptitle("Prediction BlazeFace", fontsize=13)
        plt.tight_layout()
        plt.show()

        # ── Sauvegarde ────────────────────────────────────────────────────────

    def save(self, path):
        self.model.save_weights(path)
        print(f"Model saved : {path}")

    def load(self, path):
        self.model.load_weights(path)
        print(f"Model load : {path}")