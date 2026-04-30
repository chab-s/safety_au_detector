import tensorflow as tf
import os
import csv
import numpy as np
import pandas as pd
import cv2
import albumentations as A

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
    def __init__(self, cfg, training: bool = True, transform = None, augmentation: bool = False):
        self.cfg = cfg
        csv_exists = any(f.endswith(".csv") for f in os.listdir(cfg.DATASET.ROOT.TRAIN))
        if not csv_exists:
            self.preprocess(cfg.DATASET.ROOT.CSV, special_data=True, augmentation=augmentation, normalize=True)
        self.data = pd.read_csv(cfg.DATASET.ROOT.CSV)
        self.is_train = training
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.landmarks = None

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return self.data.shape

    def __getitem__(self, i: int, image_name: str = None):
        if image_name is None:
            image_name = self.data["image_path"][i]
        image_path = os.path.join(cfg.DATASET.ROOT.IMAGES, image_name)
        image = np.array(cv2.imread(image_path, cv2.IMREAD_COLOR_RGB), dtype=np.float32)
        h, w = image.shape[:2]
        image = (image/255.0 - self.mean) / self.std
        image = image.transpose([2, 0, 1])
        image = tf.convert_to_tensor(np.expand_dims(image, 0))
        landmarks = self.data.iloc[i, :196].values
        landmarks = tf.convert_to_tensor(landmarks, dtype=tf.float32)
        landmarks = tf.reshape(landmarks, (-1, 2))

        # scale *= 1,25

        meta = {'index': i, 'image_name': image_name, 'pts': tf.convert_to_tensor(landmarks)}

        return image, meta

    def preprocess(self, out_csv: str, special_data: bool = False, augmentation: bool = False, normalize: bool = True):
        header = []

        for i in range(98):
            header.extend([f'x{i}', f'y{i}'])
        header.extend(['x_upper_left_corner', 'y_upper_left_corner'])
        header.extend(['x_lower_right_corner', 'y_lower_right_corner'])
        header.append('pose')
        header.append('expression')
        header.append('illumination')
        header.append('make-up')
        header.append('occlusion')
        header.append('blur')
        header.append('image_path')
        header.append("split")


        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            list_dir = os.listdir(self.cfg.DATASET.ROOT.TRAIN) + os.listdir(self.cfg.DATASET.ROOT.SPECIAL) if special_data else os.listdir(self.cfg.DATASET.ROOT.TRAIN)
            for file in list_dir:
                if "train" in file.split("_"):
                    split = "train"
                else:
                    split = "test"
                if file.endswith('.txt'):
                    file_path = os.path.join(self.cfg.DATASET.ROOT.TRAIN, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = line.split()
                            if normalize:
                                image = cv2.imread(os.path.join(self.cfg.DATASET.ROOT.IMAGES, data[-1]))

                                h, w = image.shape[:2]

                                landmarks = np.array(data[:196], dtype=float)

                                landmarks[0::2] /= w
                                landmarks[1::2] /= h

                                data = landmarks.tolist() + data[196:]

                        data.append(split)
                        if data:
                            writer.writerow(data)

        print(f"cvs file saved to {out_csv}")


    def augmentation_pipeline(self, size: int) -> A.ReplayCompose:
        return A.ReplayCompose(
            [
                A.HorizontalFlip(p=1.0),
            #     A.Rotate(
            #         limit=20,
            #         border_mode=cv2.BORDER_CONSTANT,
            #         p=0.7
            #     ),
            #     A.Affine(
            #         translate_percent=0.06,
            #         scale=0.15,
            #         rotate=20,
            #         border_mode=cv2.BORDER_CONSTANT,
            #         p=0.5
            #     ),
            #     A.RandomResizedCrop(
            #         size=(size, size),
            #         scale=(0.85, 1.0),
            #         p=0.4
            #     ),
            #     A.Resize(size, size, p=1.0),
            #     A.RandomBrightnessContrast(
            #         brightness_limit=0.3,
            #         contrast_limit=0.3,
            #         p=0.6
            #     ),
            #     A.HueSaturationValue(
            #         hue_shift_limit=10,
            #         sat_shift_limit=30,
            #         val_shift_limit=20,
            #         p=0.4
            #     ),
            #     A.GaussNoise(std_range=(0.02, 0.12), p=0.3),
            #     A.MotionBlur(blur_limit=5, p=0.2),
            #     A.CLAHE(clip_limit=2.0, p=0.2),
            #     A.ImageCompression(quality_range=(70, 100), p=0.2),
            #
            #     # ── Occlusions / CoarseDropout ────────────────────────────
            #     A.CoarseDropout(
            #         num_holes_range=(1, 4),
            #         hole_height_range=(10, 40),
            #         hole_width_range=(10, 40),
            #         fill=0,
            #         p=0.3
            #     )
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
    out_csv = os.path.join(cfg.DATASET.ROOT.TRAIN, "wflw.csv")
    dataset_wflw = WFLW(cfg)
    print(dataset_wflw.__len__())
    print(dataset_wflw[0])
    # dataset_wflw.preprocess(out_csv, special_data=False)$
    exit()
    data = pd.read_csv(out_csv)
    # df2 = pd.read_csv(out_csv_full)
    # print(data.info(), df2.info())

    augm = dataset_wflw.augmentation_pipeline(256)
    im_path = os.path.join("datasets/WFLW_images",  data['image_path'][0])
    img = cv2.imread(im_path, cv2.IMREAD_COLOR_RGB)
    row = data.iloc[0]
    coords = row[:196].values.astype('float32')
    keypoints = [(float(p[0]), float(p[1])) for p in coords.reshape(-1, 2)]
    applied_augm = augm(image=img, keypoints=keypoints)
    aug_img = applied_augm.get('image')
    aug_keypoints = list(applied_augm["keypoints"])
    print(dataset_wflw.flip_was_applied(applied_augm.get("replay")))
    if dataset_wflw.flip_was_applied(applied_augm.get("replay")):
        aug_keypoints = dataset_wflw.apply_flip_map(aug_keypoints)
    print(len(aug_keypoints))
    if len(aug_keypoints) < 98:
        aug_keypoints = dataset_wflw.reconstruct_keypoints(aug_keypoints, keypoints, 256)

    new_name = f"test.jpg"
    cv2.imwrite(
        new_name,
        cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
    )


    def draw(ax, img, kps, title):
        ax.imshow(img)
        ax.scatter([p[0] for p in kps], [p[1] for p in kps], s=8, c="lime", linewidths=0.5)
        ax.set_title(title)
        ax.axis("off")

    import matplotlib.pyplot as plt
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    draw(axes[0], img, keypoints, "Original")
    draw(axes[1], aug_img, aug_keypoints, "Augmenté")
    plt.tight_layout()
    plt.savefig("augmentation_sample.png", dpi=150)
    plt.show()
    print("→ augmentation_sample.png")




