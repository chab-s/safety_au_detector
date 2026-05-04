import tensorflow
import tensorflow as tf
import os
import csv
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from tqdm import tqdm

NUM_LM = 98

_contour = list(range(32, -1, -1))
_ebrow_l = list(range(33, 42))
_ebrow_r = list(range(42, 51))
_nose = [51, 52, 53, 54, 59, 58, 57, 56, 55]
_eye_l = list(range(60, 68))
_eye_r = list(range(68, 76))
_mouth_out = [82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83]
_mouth_in = [92, 91, 90, 89, 88, 95, 94, 93]
_pupils = [97, 96]

FLIP_MAP = (
        _contour
        + _ebrow_r + _ebrow_l
        + _nose
        + _eye_r + _eye_l
        + _mouth_out
        + _mouth_in
        + _pupils
)


class WFLW():
    def __init__(self, cfg, training: bool = True, transform = None):
        self.cfg = cfg
        self.is_train = training
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.augmentation = cfg.TRAIN.AUGMENTATION

        self.out_csv = None
        if cfg.DATASET.SAVE:
            self.out_csv = cfg.DATASET.ROOT.CSV

        csv_exists = any(f.endswith(".csv") for f in os.listdir(cfg.DATASET.ROOT.TRAIN))
        if not csv_exists:
            self.data = self.preprocess(self.out_csv, add_attribute_subsets=False, normalize=True)
        else:
            self.data = pd.read_csv(cfg.DATASET.ROOT.CSV)

        if self.is_train:
            self.data = self.data[self.data['split'] == 'train'].reset_index(drop=True)
        else:
            self.data = self.data[self.data['split'] == 'test'].reset_index(drop=True)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if self.augmentation:
            self.augm = self.augmentation_pipeline(self.input_size)

    def __len__(self):
        return self.data.shape

    def __getitem__(self, i: int = None, image_name: str = None):
        if image_name is not None:
            i = self.data[self.data["image_path"] == image_name].index[0]
        image_name = self.data["image_path"][i]
        landmarks = self.data.iloc[i, :196].values
        split = self.data["split"][i]

        image_path = os.path.join(cfg.DATASET.ROOT.IMAGES, image_name)
        image = np.array(cv2.imread(image_path, cv2.IMREAD_COLOR_RGB), dtype=np.float32)
        image = cv2.resize(image, self.input_size)
        image = (image/255.0 - self.mean) / self.std
        image = image.transpose([2, 0, 1])
        landmarks = tf.convert_to_tensor(landmarks, dtype=tf.float32)
        landmarks = tf.reshape(landmarks, (-1, 2))

        # scale *= 1,25

        meta = {'index': i, 'image_name': image_name, 'split': split}

        return tf.convert_to_tensor(image), tf.convert_to_tensor(landmarks), meta

    def process(self, image_name: str, augmentation: bool = False):
        image, landmarks, meta = self.__getitem__(image_name=image_name)
        landmarks *= self.input_size
        if meta["split"] == "train":
            self.images_train.append(image)
            self.landmarks_train.append(landmarks)
            if augmentation:
                image = image.numpy()
                if image.shape[0] == 3 or image.shape[0] == 1:
                    image = image.transpose(1, 2, 0)
                augm = self.augmentation_pipeline(cfg.MODEL.IMAGE_SIZE)
                apply_aug = augm(image=image, landmarks=landmarks)
                aug_image = apply_aug.get('image')
                aug_landmarks = list(apply_aug["landmarks"])
                # if dataset_wflw.flip_was_applied(apply_aug.get("replay")):
                #     aug_landmarks = dataset_wflw.apply_flip_map(aug_landmarks)
                aug_image = aug_image.transpose([2, 0, 1])
                aug_image = tf.convert_to_tensor(aug_image, dtype=tf.float32)
                self.images_train.append(aug_image)
                self.landmarks_train.append(aug_landmarks)

                # def draw(ax, img, kps, title):
                #     ax.imshow(img)
                #     ax.scatter([p[0] for p in kps], [p[1] for p in kps], s=8, c="lime", linewidths=0.5)
                #     ax.set_title(title)
                #     ax.axis("off")
                #
                # import matplotlib.pyplot as plt
                # _, axes = plt.subplots(1, 2, figsize=(10, 5))
                # draw(axes[0], image, landmarks.numpy(), "Original")
                # draw(axes[1], aug_image, aug_landmarks, "Augmenté")
                # plt.tight_layout()
                # plt.savefig("augmentation_sample.png", dpi=150)
                # plt.show()
                # print("→ augmentation_sample.png")

        elif meta["split"] == "test":
            self.images_test.append(image)
            self.landmarks_test.append(landmarks)

        yield self.images_train, self.landmarks_train, self.images_test, self.landmarks_test

    def _iter_rows(self, add_attribute_subsets: bool = False, normalize: bool = True):
        list_dir = (
            os.listdir(self.cfg.DATASET.ROOT.TRAIN) + os.listdir(self.cfg.DATASET.ROOT.ATTRIBUTE_SUBSET)
            if add_attribute_subsets
            else os.listdir(self.cfg.DATASET.ROOT.TRAIN)
        )

        for file in tqdm(list_dir, desc="Preprocessing Dataset"):
            if not file.endswith('.txt'):
                continue

            split = "train" if "train.txt" in file.split("_") else "test"
            file_path = os.path.join(self.cfg.DATASET.ROOT.TRAIN, file)

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = line.split()
                    image_path = data[-1]
                    if not data:
                        continue

                    landmarks = np.array(data[:196], dtype=float)

                    if normalize:
                        image = cv2.imread(os.path.join(self.cfg.DATASET.ROOT.IMAGES, image_path))
                        h, w = image.shape[:2]
                        landmarks[0::2] /= w
                        landmarks[1::2] /= h

                    row = landmarks.tolist() + data[196:] + [split]
                    yield self._row_to_dict(row)

    def _row_to_dict(self, row: list) -> dict:
        keys = (
                [coord for i in range(98) for coord in (f'x{i}', f'y{i}')]
                + ['x_upper_left_corner', 'y_upper_left_corner',
                   'x_lower_right_corner', 'y_lower_right_corner',
                   'pose', 'expression', 'illumination', 'make-up',
                   'occlusion', 'blur', 'image_path', 'split']
        )
        return dict(zip(keys, row))


    def preprocess(self, out_csv: str = None, add_attribute_subsets: bool = False, normalize: bool = True):
        header = list(self._row_to_dict([None] * 208).keys())
        rows = self._iter_rows(add_attribute_subsets=add_attribute_subsets, normalize=normalize)
        self.data = pd.DataFrame(rows, columns=header)
        if out_csv is not None:
            self.data.to_csv(out_csv, index=False, encoding='utf-8')
            print(f"CSV sauvegardé : {out_csv}")

        return self.data


    def augmentation_pipeline(self, size: int) -> A.ReplayCompose:
        return A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(
                    limit=(-20, 20),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.7
                ),
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    scale=(0.85, 1.15),
                    rotate=(-20, 20),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5
                ),
                A.RandomResizedCrop(
                    size=size,
                    scale=(0.85, 1.0),
                    p=0.6
                ),
                A.Resize(
                    height=size[0], width=size[1],
                    interpolation=cv2.INTER_LINEAR,
                    p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.7, 1.3),
                    contrast_limit=(0.7, 1.3),
                    p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=(-10, 10),
                    sat_shift_limit=(-30, 30),
                    val_shift_limit=(-20, 20),
                    p=0.4
                ),
                A.GaussNoise(std_range=(0.02, 0.12), p=0.3),
                A.MotionBlur(blur_limit=5, p=0.2),
                A.CLAHE(clip_limit=2.0, p=0.2),
                A.ImageCompression(quality_range=(70, 100), p=0.2),

                # ── Occlusions / CoarseDropout ────────────────────────────
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(10, 40),
                    hole_width_range=(10, 40),
                    fill=0,
                    p=0.3
                )
            ],
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,
            )
        )

    def apply_flip_map(self, kps: list) -> list:
        return [kps[FLIP_MAP[i]] for i in range(NUM_LM)]

    def flip_was_applied(self, replay: dict) -> bool:
        transforms = replay.get("transforms", [])

        for t in transforms:
            if "HorizontalFlip" in t.get("__class_fullname__", ""):
                if t.get("applied", False):
                    return True
            if self.flip_was_applied(t):
                return True
        return False

    # def reconstruct_keypoints(self, aug_kps: list, orig_kps: list, size: int) -> list:
    #     result = []
    #     for pt in aug_kps:
    #         result.append((
    #             max(0.0, min(float(pt[0]), size)),
    #             max(0.0, min(float(pt[1]), size)),
    #         ))
    #     for i in range(len(aug_kps), NUM_LM):
    #         result.append((
    #             max(0.0, min(float(orig_kps[i][0]), size)),
    #             max(0.0, min(float(orig_kps[i][1]), size)),
    #         ))
    #     return result



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

if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    dataset_wflw = WFLW(cfg)
    print(dataset_wflw.__len__())
    train_images, train_landmarks, test_images, test_landmarks = dataset_wflw.process(augmentation = cfg.TRAIN.AUGMENTATION)
    print(tf.shape(dataset_wflw.images_train), tf.shape(dataset_wflw.landmarks_train))
    print(tf.shape(dataset_wflw.images_test), tf.shape(dataset_wflw.landmarks_test))





