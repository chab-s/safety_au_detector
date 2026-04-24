import numpy as np
from blazeface import BlazeFaceDetector, set_dataset, build_dataset, build_anchors
from lib.training import plot_history

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
IMAGE_SIZE = 128

CSV_PATH = "data_files/fixed_images.csv"
OUTPUT_DIR = "created_dataset"
CHECKPOINT_DIR = "checkpoints"
WEIGHTS_PATH = "weights/blazeface.keras"

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

anchors = build_anchors()

# Prepare images (cropping + resizing) — to be done once
# image_paths, boxes_dict = set_dataset(CSV_PATH, OUTPUT_DIR, anchors)
# np.save("data_files/image_paths.npy", image_paths)
# np.save("data_files/boxes_dict.npy",  boxes_dict)

image_paths = np.load("data_files/images_paths.npy")
boxes_dict = np.load("data_files/boxes_dict.npy", allow_pickle=True).item()

# Split train / test 80-20
split = int(len(image_paths) * 0.8)
train_paths = image_paths[:split]
test_paths = image_paths[split:]

train_dataset = build_dataset(train_paths, boxes_dict, anchors, augment=True, batch_size=BATCH_SIZE)
test_dataset = build_dataset(test_paths, boxes_dict, anchors, augment=False, batch_size=BATCH_SIZE)

print(f"Train : {len(train_paths)} images | Test : {len(test_paths)} images")

detector = BlazeFaceDetector(lr=LR, checkpoint_path=CHECKPOINT_DIR)

history = detector.fit(train_dataset, epochs=EPOCHS, resume=True)
plot_history(history, save_path="results/training_curves.png")

metrics = detector.evaluate(test_dataset)

detector.save(WEIGHTS_PATH)