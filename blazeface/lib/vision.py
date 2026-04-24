import numpy as np
import tensorflow as tf
import cv2
from dataclasses import dataclass

# ─────────────────────────────────────────────
# Intersection over Union (IOU)
# ─────────────────────────────────────────────

def compute_iou(box, other_box):
    x1 = np.maximum(box[0], other_box[0])
    y1 = np.maximum(box[1], other_box[1])
    x2 = np.minimum(box[2], other_box[2])
    y2 = np.minimum(box[3], other_box[3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_other_box = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
    union = area_box + area_other_box - intersection

    return intersection / (union + 1e-6)

def batch_iou(true_boxes, pred_boxes):
    x1 = tf.math.maximum(true_boxes[..., 0], pred_boxes[..., 0])
    y1 = tf.math.maximum(true_boxes[..., 1], pred_boxes[..., 1])
    x2 = tf.math.minimum(true_boxes[..., 2], pred_boxes[..., 2])
    y2 = tf.math.minimum(true_boxes[..., 3], pred_boxes[..., 3])

    intersection = tf.math.maximum(0.0, x2 - x1) * tf.math.maximum(0.0, y2 - y1)
    area_true = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])
    area_pred = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    union = area_true + area_pred - intersection

    return intersection / (union + 1e-6)


def mean_iou(true_boxes, pred_boxes):
    return tf.math.reduce_mean(batch_iou(true_boxes, pred_boxes))


# ─────────────────────────────────────────────
# Non-Maximum Suppression (NMS)
# ─────────────────────────────────────────────

def nms(detections, iou_threshold=0.3):
    if len(detections) == 0:
        return []

    detections = tf.stack(detections)
    remaining = tf.argsort(detections[:, 0], direction='DESCENDING')
    kept = []

    while remaining.shape[0] > 0:
        best = detections[remaining[0]]
        best_box = best[1:]
        all_boxes = tf.gather(detections, remaining)[:, 1:]

        ious = batch_iou(best_box, all_boxes)
        overlapping = tf.boolean_mask(remaining, ious > iou_threshold)
        remaining = tf.boolean_mask(remaining, ious <= iou_threshold)

        kept.append(_weighted_merge(detections, overlapping))

    return kept


def _weighted_merge(detections, indices):
    if indices.shape[0] == 1:
        return detections[indices[0]]

    group = tf.gather(detections, indices)
    scores = group[:, 0:1]
    boxes = group[:, 1:]
    total_score = tf.reduce_sum(scores)

    merged_box = tf.reduce_sum(boxes * scores, axis=0) / total_score
    merged_score = total_score / tf.cast(indices.shape[0], tf.float32)

    return tf.concat([tf.reshape(merged_score, (1,)), merged_box], axis=0)


# ─────────────────────────────────────────────
# Preprocessing image
# ─────────────────────────────────────────────

@dataclass
class CropParams:
    """Décrit l'espace disponible autour des boîtes dans une image."""
    boxes_width:  int
    boxes_height: int
    image_height: int
    image_width:  int
    space_up:     int
    space_down:   int
    space_left:   int
    space_right:  int

def crop_and_resize_with_boxes(image, boxes, target_size=128):
    params = _compute_crop_params(image, boxes)
    crop = _random_crop(image, params, target_size)
    new_boxes = _remap_boxes(boxes, crop, image.shape)

    resized = cv2.resize(crop["image"], (target_size, target_size), interpolate=cv2.INTER_AREA)
    return resized, new_boxes


def _compute_crop_params(image, boxes):
    H, W, _ = image.shape
    x1_min = min(b[0] for b in boxes)
    y1_min = min(b[1] for b in boxes)
    x2_max = max(b[0] + b[2] for b in boxes)
    y2_max = max(b[1] + b[3] for b in boxes)

    return CropParams(
        boxes_width=x2_max - x1_min,
        boxes_height=y2_max - y1_min,
        image_height=H,
        image_width=W,
        space_up=y1_min,
        space_down=H - y2_max,
        space_left=x1_min,
        space_right=W - x2_max,
    )


def _random_crop(image, params: CropParams, target_size):
    p = params

    pad_h = max(0, target_size - p.boxes_height)
    pad_w = max(0, target_size - p.boxes_width)

    top = np.random.randint(0, min(p.space_up, pad_h // 2) + 1)
    bottom = np.random.randint(0, min(p.space_down, pad_h - top) + 1)
    left = np.random.randint(0, min(p.space_left, pad_w // 2) + 1)
    right = np.random.randint(0, min(p.space_right, pad_w - left) + 1)

    y1 = p.space_up - top
    y2 = p.image_height - p.space_down + bottom
    x1 = p.space_left - left
    x2 = p.image_width - p.space_right + right

    h, w = y2 - y1, x2 - x1
    size = max(h, w, target_size)

    y2 = min(y1 + size, p.image_height)
    x2 = min(x1 + size, p.image_width)

    return {"image": image[y1:y2, x1:x2], "origin": (x1, y1), "size": (x2 - x1, y2 - y1)}


def _remap_boxes(boxes, crop, original_shape):
    ox, oy = crop["origin"]
    cw, ch = crop["size"]
    new_boxes = []

    for box in boxes:
        x1 = (box[0] - ox) / cw
        y1 = (box[1] - oy) / ch
        x2 = (box[0] + box[2] - ox) / cw
        y2 = (box[1] + box[3] - oy) / ch
        new_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    return new_boxes
