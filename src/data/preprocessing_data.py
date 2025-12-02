#!/usr/bin/env python3
"""
prepare_processed_data.py

Loads raw images, splits Training into train/validation, applies
preprocessing using the SavedModel pipeline, and saves all data as TFRecords.

Optimized for large datasets and MLOps pipelines.
"""

import logging
from pathlib import Path
import yaml
import pandas as pd
import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "prepare_processed_data.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ---------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------
def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Missing configuration file: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration successfully loaded.")
    return config

# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------
def build_df_from_folder(folder: Path):
    all_paths = []
    for class_name in os.listdir(folder):
        class_folder = folder / class_name
        if not class_folder.is_dir():
            continue
        for img_file in os.listdir(class_folder):
            all_paths.append((str(class_folder / img_file), class_name))
    return pd.DataFrame(all_paths, columns=["filepath", "label"])

def get_train_val_df(raw_dir: Path, val_ratio: float=0.2, seed: int=42):
    df = build_df_from_folder(raw_dir / "Training")
    train_df, val_df = train_test_split(df, test_size=val_ratio, stratify=df["label"], random_state=seed)
    return train_df, val_df

def get_test_df(raw_dir: Path):
    return build_df_from_folder(raw_dir / "Testing")

# ---------------------------------------------------------------------
# TFRecord writer
# ---------------------------------------------------------------------
def save_preprocessed_dataset(df, artefact_dir: Path, save_dir: Path, filename_prefix="data", batch_size=32):
    logger.info(f"Saving {len(df)} samples to {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessing pipeline
    preproc_layer = tf.saved_model.load(str(artefact_dir))

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label"].values))
    ds = ds.map(lambda p,l: preproc_layer(p,l), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Helper to serialize one example
    def serialize_example(img, label):
        img_bytes = tf.io.serialize_tensor(img).numpy()
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()]))
        }))
        return example.SerializeToString()

    writer = None
    count = 0
    file_idx = 0

    for img, lbl in tqdm.tqdm(ds, total=len(df)):
        if count % batch_size == 0:
            if writer:
                writer.close()
            record_path = save_dir / f"{filename_prefix}_{file_idx:03d}.tfrecord"
            writer = tf.io.TFRecordWriter(str(record_path))
            file_idx += 1
        writer.write(serialize_example(img, lbl))
        count += 1

    if writer:
        writer.close()

    logger.info(f"Saved {count} samples into {file_idx} TFRecord files in {save_dir}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    logger.info("=== Prepare Processed Data Script Started ===")
    config = load_config(CONFIG_PATH)

    raw_dir = PROJECT_ROOT / config["path"]["raw_dir"]
    processed_dir = PROJECT_ROOT / config["path"]["processed_dir"]
    artefact_dir = PROJECT_ROOT / config["path"]["models_dir"] / "preproc_pipeline"
    seed = config["general"].get("seed", 42)
    val_ratio = config["data_preprocessing"].get("split",[0.7,0.3])[1]

    train_df, val_df = get_train_val_df(raw_dir, val_ratio=val_ratio, seed=seed)
    test_df = get_test_df(raw_dir)

    save_preprocessed_dataset(train_df, artefact_dir, processed_dir / "Training", filename_prefix="data", batch_size=32)
    save_preprocessed_dataset(val_df, artefact_dir, processed_dir / "Validation", filename_prefix="data", batch_size=32)
    save_preprocessed_dataset(test_df, artefact_dir, processed_dir / "Testing", filename_prefix="data", batch_size=32)

    logger.info("=== Prepare Processed Data Script Completed ===")

if __name__ == "__main__":
    main()
