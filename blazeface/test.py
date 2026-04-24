import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from blazeface import BlazeFaceDetector, build_dataset, build_anchors
from lib.training import plot_history

# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True, help="Chemin vers les poids (.h5 ou .keras)")
parser.add_argument("--samples", type=int, default=8, help="Nombre d'images à visualiser")
parser.add_argument("--threshold", type=float, default=0.75, help="Seuil de score pour la détection")
args = parser.parse_args()

# ─────────────────────────────────────────────
# Load model + data
# ─────────────────────────────────────────────

anchors = build_anchors()
image_paths = np.load("data_files/images_paths.npy")
boxes_dict = np.load("data_files/boxes_dict.npy", allow_pickle=True).item()

# On prend uniquement le test set (20% de la fin)
split = int(len(image_paths) * 0.8)
test_paths = image_paths[split:]

test_dataset = build_dataset(test_paths, boxes_dict, anchors, augment=False, batch_size=32)

detector = BlazeFaceDetector()
detector.SCORE_THRESHOLD = args.threshold
detector.load(args.weights)

print(f"\n{len(test_paths)} images de test | seuil : {args.threshold}\n")

# ─────────────────────────────────────────────
# metrics on test set
# ─────────────────────────────────────────────

metrics = detector.evaluate(test_dataset)
print(metrics)


# ─────────────────────────────────────────────
# Visualisation on a sample
# ─────────────────────────────────────────────

def visualize_sample(detector, test_dataset, n_samples, boxes_dict):

    collected_images = []
    collected_truths = []
    collected_paths = []

    for paths, images, grid_small, grid_large in test_dataset:
        for i in range(len(images)):
            if len(collected_images) >= n_samples:
                break
            collected_images.append(images[i])
            collected_paths.append(paths[i].numpy().decode())

            true_boxes = boxes_dict.get(collected_paths[-1], [])
            collected_truths.append(true_boxes)

        if len(collected_images) >= n_samples:
            break

    images_tensor = tf.stack(collected_images)
    detections = detector.predict(images_tensor)

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i, (ax, image, detection, true_boxes) in enumerate(
            zip(axes, collected_images, detections, collected_truths)
    ):
        ax.imshow(image.numpy())

        for box in true_boxes:
            x1, y1, x2, y2 = box * detector.INPUT_SIZE
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='lime', facecolor='none', label='GT'
            ))

        for det in detection:
            score = float(det[0])
            x1, y1, x2, y2 = det[1:].numpy() * detector.INPUT_SIZE
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            ))
            ax.text(x1, y1 - 4, f"{score:.2f}", color='red', fontsize=7)

        ax.axis('off')

    for ax in axes[n_samples:]:
        ax.axis('off')

    plt.suptitle("Ground truth    Prediction", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("results/test_sample.png", bbox_inches='tight', dpi=120)
    plt.show()
    print("Vis saved : results/test_sample.png")


import os

os.makedirs("results", exist_ok=True)

visualize_sample(detector, test_dataset, n_samples=args.samples, boxes_dict=boxes_dict)