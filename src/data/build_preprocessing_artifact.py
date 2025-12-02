#!/usr/bin/env python3
"""
build_preprocessing_artifact.py

Builds and saves the preprocessing pipeline as a TensorFlow SavedModel.

Reads configuration from config.yaml.

Designed for MLOps integration.
"""

import logging
from pathlib import Path
import yaml
import tensorflow as tf

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
        logging.FileHandler(LOG_DIR / "build_preproc_artifact.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ---------------------------------------------------------------------
# Load config
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
# Preprocessing pipeline module
# ---------------------------------------------------------------------
class PreprocessingLayer(tf.Module):
    """
    Standalone preprocessing module saved as a TF SavedModel.
    Accepts (path, label) → returns (preprocessed_image, label_index).
    """
    def __init__(self, low_clip, high_clip, classes, target_size=(260,260)):
        super().__init__()
        self.low_clip = low_clip
        self.high_clip = high_clip
        self.target_size = target_size
        self.classes = tf.constant(classes)

    def crop_black_background(self, img, thresh=10):
        if img.shape[-1] == 3:
            img_gray = tf.image.rgb_to_grayscale(img)
        else:
            img_gray = img
        mask = img_gray > thresh
        coords = tf.where(mask[:, :, 0])
        def crop():
            y0 = tf.reduce_min(coords[:,0])
            x0 = tf.reduce_min(coords[:,1])
            y1 = tf.reduce_max(coords[:,0])
            x1 = tf.reduce_max(coords[:,1])
            return img[y0:y1+1, x0:x1+1, :]
        def no_crop():
            return img
        return tf.cond(tf.shape(coords)[0] > 0, crop, no_crop)

    @tf.function(input_signature=[
        tf.TensorSpec([], tf.string),
        tf.TensorSpec([], tf.string)
    ])
    def __call__(self, path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = self.crop_black_background(img)
        img = tf.image.resize(img, self.target_size, method='area')
        mean, var = tf.nn.moments(img, axes=[0,1,2])
        std = tf.maximum(tf.sqrt(var), 1e-6)
        img = (img - mean)/std
        img = tf.clip_by_value(img, self.low_clip, self.high_clip)
        label_idx = tf.where(self.classes == label)[0][0]
        return img, label_idx

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    logger.info("=== Building preprocessing artifact ===")
    config = load_config(CONFIG_PATH)
    preproc_layer = PreprocessingLayer(
        config["data_preprocessing"]["low_clip"],
        config["data_preprocessing"]["high_clip"],
        config["general"]["classes"],
        target_size=(config["data_preprocessing"]["img_size"], config["data_preprocessing"]["img_size"])
    )
    artefact_dir = Path(config["path"]["models_dir"]) / "preproc_pipeline"
    artefact_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(preproc_layer, str(artefact_dir))
    logger.info(f"Saved preprocessing artifact at: {artefact_dir}")

if __name__ == "__main__":
    main()
