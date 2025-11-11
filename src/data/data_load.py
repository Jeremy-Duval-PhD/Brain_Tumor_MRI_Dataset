#!/usr/bin/env python3
"""
data_load.py

Downloads a Kaggle dataset using the kagglehub API.
Reads configuration from config.yaml and credentials from .secrets/kaggle.json.

This script is designed for integration into an MLOps pipeline.
"""

import os
import yaml
import logging
from pathlib import Path
import kagglehub
import shutil


# --- Logger Configuration ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Display logs in console
        logging.FileHandler(LOG_DIR / "data_load.log", mode="a")  # Save logs to a file
    ]
)

logger = logging.getLogger(__name__)


# --- Global Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SECRETS_PATH = PROJECT_ROOT / ".secrets" / "kaggle.json"


def load_config(config_path: Path) -> dict:
    """Load configuration parameters from config.yaml."""
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Missing configuration file: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration successfully loaded.")
    return config


def setup_kaggle_credentials(secret_path: Path):
    """Set Kaggle credentials environment variables for kagglehub."""
    if not secret_path.exists():
        logger.error(f"Kaggle credentials file not found: {secret_path}")
        raise FileNotFoundError(f"Missing Kaggle credentials: {secret_path}")

    os.environ["KAGGLE_CONFIG_DIR"] = str(secret_path.parent)
    os.environ["KAGGLE_CONFIG_FILE"] = str(secret_path)
    logger.info("Kaggle credentials configured from .secrets/kaggle.json")


def download_data(dataset_name: str, output_dir: Path):
    """Download the dataset using kagglehub and move it to the desired output directory."""
    logger.info(f"Starting download for dataset: {dataset_name}")

    try:
        download_path = kagglehub.dataset_download(dataset_name)
        logger.info(f"Dataset downloaded to temporary path: {download_path}")
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")
        raise

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files to data/raw
    logger.info(f"Copying files to {output_dir}")
    try:
        for item in Path(download_path).iterdir():
            dest = output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        logger.info("Dataset successfully copied to raw data directory.")
    except Exception as e:
        logger.error(f"Error while copying dataset files: {e}")
        raise


def main():
    """Main entry point for the data loading process."""
    logger.info("=== Data Load Script Started ===")

    config = load_config(CONFIG_PATH)
    setup_kaggle_credentials(SECRETS_PATH)

    dataset_name = config["data_loading"]["kaggle_dataset"]
    output_dir = PROJECT_ROOT / config["data_loading"]["raw_dir"]

    download_data(dataset_name, output_dir)

    logger.info("=== Data Load Script Completed Successfully ===")


if __name__ == "__main__":
    main()
