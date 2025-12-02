#!/usr/bin/env python3
"""
preprocessing.py

Create and save a preprocessing pipeline (SavedModel) and apply it to produce
TFRecord datasets from raw image file lists.

This script is designed to be used in an MLOps pipeline.
Configuration comes from config.yaml (project root).
"""

from pathlib import Path
import logging
import sys
from typing import List, Tuple

import yaml
import tensorflow as tf
import pandas as pd
import tqdm

# --- Logging setup ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "preprocessing.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


# --- Helpers ---
def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuration loaded")
    return cfg


def df_from_raw_dir(raw_dir: Path, subfolder: str = "Training") -> pd.DataFrame:
    """
    Build a DataFrame with columns ['filepath', 'label'] by walking
    raw_dir / subfolder and enumerating images per class folder.
    """
    root = Path(raw_dir) / subfolder
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    records = []
    for class_name in sorted([p.name for p in root.iterdir() if p.is_dir()]):
        folder = root / class_name
        for img_path in sorted(folder.iterdir()):
            if img_path.is_file():
                records.append((str(img_path), class_name))
    df = pd.DataFrame(records, columns=["filepath", "label"])
    return df


# --- Preprocessing Module (serializable) ---
class PreprocessingModule(tf.Module):
    """
    A serializable preprocessing module. It exposes a signature "serving_default"
    that accepts (path: tf.string, label: tf.string) and returns a dict:
      { "image": tf.float32 [H,W,3], "label": tf.int64 }
    """

    def __init__(self, low_clip: float, high_clip: float, classes: List[str], target_size: Tuple[int, int] = (260, 260)):
        super().__init__()
        # constants
        self.low_clip = tf.constant(float(low_clip), dtype=tf.float32)
        self.high_clip = tf.constant(float(high_clip), dtype=tf.float32)
        self.target_size = target_size

        # lookup table (serializable)
        keys = tf.constant(classes, dtype=tf.string)
        values = tf.cast(tf.range(len(classes)), tf.int64)
        self.lookup = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=tf.constant(-1, dtype=tf.int64)
        )

    def _crop_black_background(self, img: tf.Tensor, thresh: int = 10) -> tf.Tensor:
        """Crop the black borders of an image (works with float images [0,1])."""
        # Convert to grayscale
        img_gray = tf.image.rgb_to_grayscale(img) if img.shape[-1] == 3 else img
        mask = img_gray > tf.cast(thresh / 255.0, img_gray.dtype) if img.dtype.is_floating else img_gray > thresh
        coords = tf.where(mask[:, :, 0])
        def do_crop():
            y0 = tf.reduce_min(coords[:, 0])
            x0 = tf.reduce_min(coords[:, 1])
            y1 = tf.reduce_max(coords[:, 0])
            x1 = tf.reduce_max(coords[:, 1])
            # clip bounds to image shape
            y0 = tf.maximum(y0, 0)
            x0 = tf.maximum(x0, 0)
            y1 = tf.minimum(y1, tf.shape(img)[0] - 1)
            x1 = tf.minimum(x1, tf.shape(img)[1] - 1)
            return img[y0:y1 + 1, x0:x1 + 1, :]
        return tf.cond(tf.shape(coords)[0] > 0, do_crop, lambda: img)

    @tf.function(input_signature=[tf.TensorSpec([], tf.string), tf.TensorSpec([], tf.string)])
    def preprocess(self, path: tf.Tensor, label: tf.Tensor):
        """
        Read file, crop, resize, z-score normalize (image-wise), clip and encode label.

        Inputs:
          - path: scalar tf.string path to a PNG/JPG image
          - label: scalar tf.string class name (must be in classes list)
        Outputs:
          - dict with keys 'image' (tf.float32 HxWx3) and 'label' (tf.int64)
        """
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)  # supports png/jpg
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1] float32

        # crop + resize
        img = self._crop_black_background(img, thresh=10)
        img = tf.image.resize(img, self.target_size, method="area")

        # per-image z-score
        mean, var = tf.nn.moments(img, axes=[0, 1, 2])
        std = tf.maximum(tf.sqrt(var), tf.constant(1e-6, dtype=var.dtype))
        img = (img - mean) / std

        # clip
        img = tf.clip_by_value(img, self.low_clip, self.high_clip)

        # label lookup
        label_idx = self.lookup.lookup(label)

        return {"image": img, "label": label_idx}


# --- TFRecord utilities ---
def _serialize_example(image: tf.Tensor, label: tf.Tensor) -> bytes:
    """
    Serialize image tensor and label into a TFRecord Example (bytes).
    image: tf.Tensor float32 (H,W,3)
    label: tf.Tensor int64 scalar
    """
    # serialize image as raw bytes with tf.io.serialize_tensor (keeps dtype & shape)
    image_ser = tf.io.serialize_tensor(image)  # scalar string tensor
    # Build Example
    feature = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_ser.numpy()])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label.numpy())]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_tfrecords_from_dataset(dataset: tf.data.Dataset, out_dir: Path, prefix: str = "data", examples_per_file: int = 1024):
    """
    Write dataset of (image,label) tensors into TFRecord files.
    dataset must yield tuples (image_tensor, label_tensor).
    This function writes examples_per_file per TFRecord file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    file_idx = 0
    written = 0

    for i, (img, lbl) in enumerate(tqdm.tqdm(dataset)):
        if i % examples_per_file == 0:
            if writer:
                writer.close()
            out_path = out_dir / f"{prefix}_{file_idx:03d}.tfrecord"
            writer = tf.io.TFRecordWriter(str(out_path))
            file_idx += 1

        # serialize and write (we call .numpy() here because we are in eager mode)
        ser = _serialize_example(img, lbl)
        writer.write(ser)
        written += 1

    if writer:
        writer.close()

    logger.info(f"Wrote {written} examples into {file_idx} TFRecord files at {out_dir}")


# --- main processing functions ---
def build_and_save_preproc(cfg: dict):
    """Build the PreprocessingModule and save it as a SavedModel artifact."""
    models_dir = Path(cfg["path"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir = models_dir / "preproc_pipeline"
    classes = cfg["general"]["classes"]
    low_clip = cfg["data_preprocessing"]["low_clip"]
    high_clip = cfg["data_preprocessing"]["high_clip"]
    target_size = tuple(cfg["data_preprocessing"].get("target_size", (260, 260)))

    preproc = PreprocessingModule(low_clip, high_clip, classes, target_size=target_size)

    # Save with signature
    tf.saved_model.save(preproc, str(pipeline_dir), signatures={"serving_default": preproc.preprocess})
    logger.info(f"Saved preprocessing pipeline to {pipeline_dir}")
    return pipeline_dir


def apply_preproc_and_write(df: pd.DataFrame, artifact_dir: Path, out_dir: Path, prefix: str = "data", examples_per_file: int = 1024):
    """
    Apply the saved preprocessing pipeline to a pandas DataFrame (filepath,label)
    and write resulting (image,label) to TFRecords in out_dir.
    """
    # load saved model and get signature
    model = tf.saved_model.load(str(artifact_dir))
    preprocess_fn = model.signatures["serving_default"]

    # create tf dataset from dataframe
    paths = df["filepath"].astype(str).values
    labels = df["label"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # map using the signature (it returns a dict)
    def _map_fn(p, l):
        out = preprocess_fn(p, l)
        # out["image"] is a tensor, out["label"] is scalar tensor
        return out["image"], out["label"]
    ds = ds.map(lambda p, l: _map_fn(p, l), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # write TFRecords
    write_tfrecords_from_dataset(ds, out_dir, prefix, examples_per_file)


def main():
    # load config
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust depth: repo_root/scripts -> parents[1]
    cfg_path = PROJECT_ROOT / "config.yaml"
    cfg = load_config(cfg_path)

    # build/save preproc artifact
    pipeline_dir = build_and_save_preproc(cfg)

    # get dataframes
    raw_dir = Path(cfg["path"]["raw_dir"])
    train_df = df_from_raw_dir(raw_dir, subfolder="Training")
    val_df = df_from_raw_dir(raw_dir, subfolder="Training")  # if you split separately, replace by split
    # If you want a stratified split, perform it here (I keep it simple)
    # For reproducible splitting, use sklearn.model_selection.train_test_split with stratify

    # Write preprocessed TFRecords
    processed_dir = Path(cfg["path"]["processed_dir"])
    apply_preproc_and_write(train_df, pipeline_dir, processed_dir / "Training", prefix="train", examples_per_file=cfg["data_preprocessing"].get("examples_per_file", 1024))
    apply_preproc_and_write(val_df, pipeline_dir, processed_dir / "Validation", prefix="val", examples_per_file=cfg["data_preprocessing"].get("examples_per_file", 1024))

    logger.info("Preprocessing completed")


if __name__ == "__main__":
    main()
