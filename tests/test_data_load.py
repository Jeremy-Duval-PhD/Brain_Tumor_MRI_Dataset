
import os
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch

from src.data.data_load import (
    load_config,
    setup_kaggle_credentials,
    download_data,
)


# ---------------------------
# load_config tests
# ---------------------------
@pytest.mark.unit
def test_load_config_success(tmp_path):
    """
    Test that a valid YAML config file is correctly loaded.
    """
    config_content = {"key": "value"}
    config_file = tmp_path / "config.yaml"

    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    config = load_config(config_file)

    assert config == config_content


def test_load_config_file_not_found():
    """
    Test that loading a missing config file raises FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        load_config(Path("non_existent.yaml"))


# ---------------------------
# setup_kaggle_credentials tests
# ---------------------------
@pytest.mark.unit
def test_setup_kaggle_credentials_success(tmp_path):
    """
    Test that environment variables are correctly set when credentials exist.
    """
    secret_file = tmp_path / "kaggle.json"
    secret_file.write_text("{}")

    setup_kaggle_credentials(secret_file)

    assert os.environ["KAGGLE_CONFIG_DIR"] == str(secret_file.parent)
    assert os.environ["KAGGLE_CONFIG_FILE"] == str(secret_file)


def test_setup_kaggle_credentials_missing_file():
    """
    Test that missing credentials file raises FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        setup_kaggle_credentials(Path("missing.json"))


# ---------------------------
# download_data tests
# ---------------------------
@pytest.mark.integration
@patch("src.data.data_load.kagglehub.dataset_download")
def test_download_data_success(mock_download, tmp_path):
    """
    Test that dataset is downloaded and files are copied to output directory.
    """
    # Create fake downloaded directory
    fake_download_dir = tmp_path / "downloaded"
    fake_download_dir.mkdir()

    # Create fake files
    file1 = fake_download_dir / "file1.txt"
    file1.write_text("data")

    subdir = fake_download_dir / "subdir"
    subdir.mkdir()
    file2 = subdir / "file2.txt"
    file2.write_text("data")

    mock_download.return_value = str(fake_download_dir)

    output_dir = tmp_path / "output"

    download_data("dummy_dataset", output_dir)

    # Check files copied
    assert (output_dir / "file1.txt").exists()
    assert (output_dir / "subdir").exists()
    assert (output_dir / "subdir" / "file2.txt").exists()


@pytest.mark.unit
@patch("src.data.data_load.kagglehub.dataset_download")
def test_download_data_download_failure(mock_download, tmp_path):
    """
    Test that an exception is raised if dataset download fails.
    """
    mock_download.side_effect = Exception("Download error")

    with pytest.raises(Exception):
        download_data("dummy_dataset", tmp_path)


@pytest.mark.unit
@patch("src.data.data_load.kagglehub.dataset_download")
def test_download_data_copy_failure(mock_download, tmp_path):
    """
    Test that an exception is raised if file copying fails.
    """
    fake_download_dir = tmp_path / "downloaded"
    fake_download_dir.mkdir()

    mock_download.return_value = str(fake_download_dir)

    # Force copy failure by mocking shutil.copy2
    with patch("shutil.copy2", side_effect=Exception("Copy error")):
        with pytest.raises(Exception):
            download_data("dummy_dataset", tmp_path / "output")