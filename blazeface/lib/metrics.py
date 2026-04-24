import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from lib.vision import batch_iou


@dataclass
class DetectionMetrics:
    mean_iou: float
    precision: float
    recall: float
    f1: float
    all_ious: list = field(default_factory=list)

    def __str__(self):
        return (
            f"\n{'=' * 40}\n"
            f"  Mean IoU  : {self.mean_iou:.4f}\n"
            f"  Precision : {self.precision:.4f}\n"
            f"  Recall    : {self.recall:.4f}\n"
            f"  F1        : {self.f1:.4f}\n"
            f"{'=' * 40}"
        )

def evaluate_detections(pred_boxes_batch, true_boxes_batch, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    all_ious = []

    for pred_boxes, true_boxes in zip(pred_boxes_batch, true_boxes_batch):
        tp, fp, fn, ious = _match_boxes(pred_boxes, true_boxes, iou_threshold)
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        all_ious.extend(ious)

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = float(true_positives) / (float(true_positives) + float(false_negatives) + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return DetectionMetrics(
        mean_iou=float(np.mean(all_ious)) if all_ious else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        all_ious=all_ious,
    )

def _match_boxes(pred_boxes, true_boxes, iou_threshold):
    if tf.shape(pred_boxes)[0] == 0 and tf.shape(true_boxes)[0] == 0:
        return 0, 0, 0, []
    if tf.shape(pred_boxes)[0] == 0:
        return 0, 0, tf.shape(true_boxes)[0], []
    if tf.shape(true_boxes)[0] == 0:
        return 0, tf.shape(pred_boxes)[0], 0, []

    pred_boxes = tf.cast(pred_boxes, tf.float32)
    true_boxes = tf.cast(true_boxes, tf.float32)

    n = min(len(pred_boxes), len(true_boxes))
    ious = batch_iou(true_boxes[:n], pred_boxes[:n])
    matched = ious >= iou_threshold

    tp = int(tf.reduce_sum(tf.cast(matched, tf.int32)))
    fp = int(max(0, len(pred_boxes) - len(true_boxes)))
    fn = int(tf.reduce_sum(tf.cast(~matched, tf.int32)))

    return tp, fp, fn, ious.numpy().tolist()