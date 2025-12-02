#!/usr/bin/env python3
"""
preprocessing.py

Creates a preprocessing pipeline artifact (SavedModel) and uses it
to preprocess train/validation/test splits.

Reads configuration from config.yaml.

This script is designed for integration into an MLOps pipeline.
"""

import os
import yaml
import logging
from pathlib import Path
import pandas as pd
import tqdm

import tensorflow as tf
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Logger Configuration
# ---------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "preprocessing.log", mode="a")
    ]
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Global Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------
def load_config(config_path: Path) -> dict:
    """Load configuration parameters from config.yaml."""
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Missing configuration file: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration successfully loaded.")
    return config


# ---------------------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------------------
def get_train_val_df(raw_dir: Path):
    """
    Build a DataFrame of (filepath, label) for images in raw_dir/Training,
    then stratify-split into train/validation.
    """
    train_path = PROJECT_ROOT / raw_dir / "Training"
    all_paths = []

    for class_name in os.listdir(train_path):
        class_folder = train_path / class_name
        for img_name in os.listdir(class_folder):
            all_paths.append((str(class_folder / img_name), class_name))

    df_train = pd.DataFrame(all_paths, columns=["filepath", "label"])

    train_df, val_df = train_test_split(
        df_train,
        test_size=0.2,
        stratify=df_train["label"],
        random_state=42
    )

    return train_df, val_df


def get_test_df(raw_dir: Path):
    """
    Build a DataFrame of (filepath, label) for images in raw_dir/Testing.
    """
    test_path = PROJECT_ROOT / raw_dir / "Testing"
    all_paths = []

    for class_name in os.listdir(test_path):
        class_folder = test_path / class_name
        for img_name in os.listdir(class_folder):
            all_paths.append((str(class_folder / img_name), class_name))

    return pd.DataFrame(all_paths, columns=["filepath", "label"])


# ---------------------------------------------------------------------
# Preprocessing Pipeline (Artifact)
# ---------------------------------------------------------------------
class PreprocessingLayer(tf.Module):
    """
    Standalone preprocessing module saved as a TF SavedModel.
    It accepts (path, label) → returns (preprocessed_image, label_index).
    """

    def __init__(self, low_clip, high_clip, classes, target_size=(260, 260)):
        super().__init__()
        self.low_clip = low_clip
        self.high_clip = high_clip
        self.target_size = target_size
        self.classes = tf.constant(classes)

    def crop_black_background(self, img, thresh=10):
        """Remove black borders around the MRI image."""
        if img.shape[-1] == 3:
            img_gray = tf.image.rgb_to_grayscale(img)
        else:
            img_gray = img

        mask = img_gray > thresh
        coords = tf.where(mask[:, :, 0])

        def crop():
            y0 = tf.reduce_min(coords[:, 0])
            x0 = tf.reduce_min(coords[:, 1])
            y1 = tf.reduce_max(coords[:, 0])
            x1 = tf.reduce_max(coords[:, 1])
            return img[y0:y1 + 1, x0:x1 + 1, :]

        def no_crop():
            return img

        return tf.cond(tf.shape(coords)[0] > 0, crop, no_crop)

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.string),
        tf.TensorSpec([], tf.string)
    ])
    def __call__(self, path, label):
        """Main preprocessing function executed inside the SavedModel."""
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        img = self.crop_black_background(img)
        img = tf.image.resize(img, self.target_size, method='area')

        mean, var = tf.nn.moments(img, axes=[0, 1, 2])
        std = tf.maximum(tf.sqrt(var), 1e-6)
        img = (img - mean) / std

        img = tf.clip_by_value(img, self.low_clip, self.high_clip)

        label_idx = tf.where(self.classes == label)[0][0]
        return img, label_idx


# ---------------------------------------------------------------------
# TFRecord Writing
# ---------------------------------------------------------------------
def save_preprocessed_dataset(
    df: pd.DataFrame,
    artefact_dir: Path,
    save_dir: Path,
    filename_prefix="data",
    batch_size=1
):
    """
    Apply the saved preprocessing pipeline to a DataFrame and store
    the processed samples as TFRecord files.
    """
    logger.info(f"Saving preprocessed TFRecords to: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    preproc_layer = tf.saved_model.load(str(artefact_dir))

    ds = tf.data.Dataset.from_tensor_slices(
        (df["filepath"].values, df["label"].values)
    )
    ds = ds.map(
        lambda p, l: preproc_layer(p, l),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # TFRecord serializer
    def serialize_example(image, label):
        image_bytes = tf.io.serialize_tensor(image).numpy()
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()]))
        }))
        return example.SerializeToString()

    writer = None
    file_idx = 0
    count = 0

    for image, label in tqdm.tqdm(ds):
        if count % batch_size == 0:
            if writer:
                writer.close()
            record_path = save_dir / f"{filename_prefix}_{file_idx:03d}.tfrecord"
            writer = tf.io.TFRecordWriter(str(record_path))
            file_idx += 1

        writer.write(serialize_example(image, label))
        count += 1

    if writer:
        writer.close()

    logger.info(f"Saved {count} samples into {file_idx} TFRecord files.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    logger.info("=== Preprocessing Script Started ===")

    config = load_config(CONFIG_PATH)

    # Build and save the TF preprocessing layer
    preproc_layer = PreprocessingLayer(
        config["data_preprocessing"]["low_clip"],
        config["data_preprocessing"]["high_clip"],
        config["general"]["classes"]
    )

    artefact_dir = PROJECT_ROOT / config["path"]["models_dir"] / "preproc_pipeline"
    tf.saved_model.save(preproc_layer, str(artefact_dir))
    logger.info(f"Saved preprocessing artifact at: {artefact_dir}")

    # Load & preprocess datasets
    raw_dir = PROJECT_ROOT / config["path"]["raw_dir"]

    train_df, val_df = get_train_val_df(raw_dir)
    test_df = get_test_df(raw_dir)

    save_preprocessed_dataset(
        train_df,
        artefact_dir,
        PROJECT_ROOT / config["path"]["processed_dir"] / "Training",
        filename_prefix="data",
        batch_size=1
    )

    save_preprocessed_dataset(
        val_df,
        artefact_dir,
        PROJECT_ROOT / config["path"]["processed_dir"] / "Validation",
        filename_prefix="data",
        batch_size=1
    )

    save_preprocessed_dataset(
        test_df,
        artefact_dir,
        PROJECT_ROOT / config["path"]["processed_dir"] / "Testing",
        filename_prefix="data",
        batch_size=1
    )

    logger.info("=== Preprocessing Script Completed Successfully ===")


if __name__ == "__main__":
    main()
