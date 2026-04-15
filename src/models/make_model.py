#!/usr/bin/env python3
"""
make_model.py

Functions to make new model.
Reads configuration from config.yaml and credentials from .secrets/kaggle.json.

This script is designed for integration into an MLOps pipeline.
"""

import os
import yaml
import logging
from pathlib import Path
import json

import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from datetime import datetime
import pandas as pd
from sklearn.metrics import f1_score

from commons import load_tfrecord_dataset, split_labels
from model_archi import  get_model_built, compile_model, masked_sparse_cce


# --- Logger Configuration ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Display logs in console
        logging.FileHandler(LOG_DIR / "make_model.log", mode="a")  # Save logs to a file
    ]
)

logger = logging.getLogger(__name__)


# --- Global Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SECRETS_PATH = PROJECT_ROOT / ".secrets" / "ngrok.json"



""" Config """
def load_config(config_path: Path) -> dict:
    """Load configuration parameters from config.yaml."""
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Missing configuration file: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration successfully loaded.")
    return config


def setup_MLFlow(traking_url, experiment_name="Brain_Tumor_Training"):
    mlflow.set_tracking_uri(traking_url)
    mlflow.set_experiment(experiment_name)
    
    
def get_mlflow_link():
    if not SECRETS_PATH.exists():
        logger.error(f"Ngrok file not found: {SECRETS_PATH}")
        raise FileNotFoundError(f"Missing Ngrok file: {SECRETS_PATH}")

    with open(SECRETS_PATH, "r") as f:
        data = json.load(f)

    link = data.get("forward_link", None)
    
    if link == None:
        logger.error(f"forward_link not found in: {SECRETS_PATH}")
        raise FileNotFoundError(f"Missing Ngrok forward link in: {SECRETS_PATH}")
        
    return link


def setup_tensorflow(debug=False):
    if tf.config.list_physical_devices('GPU'):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        if debug:
            print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
        tf.config.list_physical_devices('GPU')








""" Training """
def config_callbacks(checkpoint_dir, to_monitor = "val_tumor_type_loss", mode = "min"):
    reduce_lr = ReduceLROnPlateau(
        monitor=to_monitor,
        mode=mode,
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor=to_monitor,
        mode=mode,
        min_delta=0.00001,
        patience=10,
        restore_best_weights=False,
        verbose=1,
    )
    
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/epoch_{epoch:02d}.weights.h5",
        monitor=to_monitor,
        mode=mode,
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
    
    terminate_nan = keras.callbacks.TerminateOnNaN()
    
    return [reduce_lr, early_stopping, checkpoint_cb, terminate_nan]


def run_training_heads(model, loss_weight_presence, loss_weight_type,\
                 config, train_ds, val_ds):
    
    freeze_backbone = config["model"]["epochs"]
    project_name = config["model"]["project_name"]
    mask_tumor_type_loss = config["model"]["mask_tumor_type_loss"]
    model_type = config["model"]["model_type"]
    two_head = config["model"]["two_head"]
    data_augmentation = config["model"]["data_augmentation"]
    checkpoint_dir = config["path"]["checkpoint_dir"]
    model_name = f"{project_name}_{model_type}_{'2Head' if two_head else '1Head'}"
    epochs = config["model"]["epochs"]
    
    
    RUN_NAME = (
        f"{model_type}"
        f"freeze={freeze_backbone}_"
        f"mask={mask_tumor_type_loss}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    
    print(f"Run name: {RUN_NAME}\n")
    
    with mlflow.start_run(run_name=RUN_NAME):
    
        mlflow.tensorflow.autolog(registered_model_name=model_name)
    
        mlflow.log_params({
            "project": project_name,
            "model_type": model_type,
            "pretrained_weights": "RadImageNet",
            "backbone_frozen": freeze_backbone,
            "head_1": "tumor_presence_binary",
            "head_2": "tumor_type_softmax",
            "loss_weight_presence": loss_weight_presence,
            "loss_weight_type": loss_weight_type,
            "mask_tumor_type_loss": mask_tumor_type_loss,
            "label_0_means": "no_tumor",
            "data_augmentation": data_augmentation,
        })
    
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            #steps_per_epoch=count_tfrecord_batches(TRAIN_DIR, BATCH_SIZE),
            #validation_steps=count_tfrecord_batches(VAL_DIR, BATCH_SIZE),
            callbacks= config_callbacks(checkpoint_dir),
            verbose=1,
        )
    
        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name)[-1].version
    
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging"
        )
        
        return history
    

def normalize(col):
    return (col - col.min()) / (col.max() - col.min() + 1e-8)
    

def keep_best_epoch_heads(history):
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = history_df.index
    
    metrics_cols = [
        "epoch",
        "val_tumor_presence_recall",
        "val_tumor_type_accuracy",
        "val_tumor_presence_loss",
        "val_tumor_type_loss"
    ]
    
    df = history_df[metrics_cols].copy()
    
    df = df[
        (df["val_tumor_presence_recall"] >= 0.94) &
        (df["val_tumor_type_accuracy"] >= 0.55)
    ]
    
    df["pres_rec_norm"] = normalize(df["val_tumor_presence_recall"])
    df["type_accu_norm"] = normalize(df["val_tumor_type_accuracy"])
    df["pres_loss_norm"] = 1 - normalize(df["val_tumor_presence_loss"])
    df["type_loss_norm"] = 1 - normalize(df["val_tumor_type_loss"])
    
    df["S"] = (
        0.40 * df["pres_rec_norm"]
      + 0.35 * df["type_accu_norm"]
      + 0.15 * df["pres_loss_norm"]
      + 0.10 * df["type_loss_norm"]
    )

    if df.empty:
        raise ValueError("No valid epoch found with given constraints")
    best_row = df.sort_values("S", ascending=False).iloc[0]
    best_epoch = int(best_row["epoch"])
    
    print(f"✅ Best epoch selected from S: {best_epoch}")
    print(best_row)
    
    return best_row, best_epoch


def save_best_model(model, best_row, best_epoch, \
                    checkpoint_dir, model_name, models_dir):
    mlflow.set_tags({
        "model_stage": "best_manual_epoch",
        "best_epoch": best_epoch,
        "selection_method": "composite_score_S",
    })
    
    mlflow.log_metric("S", best_row.iloc[-1])
    mlflow.log_metric("Best epoch", best_row.iloc[0])
    
    model.load_weights(f"{checkpoint_dir}/epoch_{best_epoch:02d}.weights.h5")
    print(f"✅ Loaded best epoch: {best_epoch}")
    
    mlflow.tensorflow.log_model(
        model,
        name=f"best_epoch_{best_epoch}_manual",
        registered_model_name=model_name
    )
    print(f"✅ Registered best model: {model_name}")
    
    #model.save(f"{models_dir}/brain_tumor_model_best_epoch_{best_epoch}.keras")
    model.save_weights(
        f"{models_dir}/brain_tumor_weights_epoch_{best_epoch}.weights.h5"
    )
    mlflow.log_artifacts(models_dir, artifact_path="exported_model_files")
    
    return model


""" Fine-tuning """
def unfreeze_layers(backbone, unfreeze_layer):
    for layer in backbone.layers:
        if layer.name.startswith(unfreeze_layer):
            layer.trainable = True
        else:
            layer.trainable = False
    return backbone

class MaskedMacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset):
        super().__init__()
        self.val_dataset = val_dataset

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        y_true_all = []
        y_pred_all = []

        for x_batch, y_batch in self.val_dataset:
            preds = self.model.predict(x_batch, verbose=0)

            y_true = y_batch["tumor_type"].numpy()
            y_pred = tf.argmax(preds["tumor_type"], axis=1).numpy()

            mask = y_true != 0

            y_true_all.extend(y_true[mask])
            y_pred_all.extend(y_pred[mask])

        f1 = f1_score(y_true_all, y_pred_all, average="macro")

        logs["val_masked_macro_f1"] = f1

        mlflow.log_metric("val_masked_macro_f1", f1, step=epoch)
        
        
def finetuning_callback(checkpoint_dir, val_ds):
    tmp = config_callbacks(checkpoint_dir)
    masked_macro_f1 = MaskedMacroF1Callback(val_ds)
    tmp.append(masked_macro_f1)
    return tmp


def run_training_finetuning(model, loss_weight_presence, loss_weight_type,\
                 config, train_ds, val_ds):
    
    freeze_backbone = config["model"]["freeze_backbone"]
    unfreeze_layer = config["model"]["unfreeze_layer"]
    project_name = config["model"]["project_name"]
    mask_tumor_type_loss = config["model"]["mask_tumor_type_loss"]
    model_type = config["model"]["model_type"]
    two_head = config["model"]["two_head"]
    data_augmentation = config["model"]["data_augmentation"]
    checkpoint_dir = config["path"]["checkpoint_dir"]
    model_name = f"{project_name}_{model_type}_{'2Head' if two_head else '1Head'}"
    epochs = config["model"]["epochs"]
    
    
    RUN_NAME = (
        f"{model_type}"
        f"freeze={freeze_backbone}_"
        f"unfreeze_layers={unfreeze_layer}_"
        f"mask={mask_tumor_type_loss}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    
    print(f"Run name: {RUN_NAME}\n")
    
    with mlflow.start_run(run_name=RUN_NAME):
    
        mlflow.tensorflow.autolog(registered_model_name=model_name)
    
        mlflow.log_params({
            "project": project_name,
            "model_type": model_type,
            "pretrained_weights": "RadImageNet",
            "backbone_frozen": freeze_backbone,
            "backfone_not_frozen": unfreeze_layer,
            "head_1": "tumor_presence_binary",
            "head_2": "tumor_type_softmax",
            "loss_weight_presence": loss_weight_presence,
            "loss_weight_type": loss_weight_type,
            "mask_tumor_type_loss": mask_tumor_type_loss,
            "label_0_means": "no_tumor",
            "data_augmentation": data_augmentation,
        })
    
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=finetuning_callback(checkpoint_dir, val_ds),
            verbose=1,
        )
    
        client = MlflowClient()
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise ValueError("No model versions found in MLflow")
        latest_version = versions[-1].version
    
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging"
        )
        
        return history
        

def keep_best_epoch_finetuning(history):
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = history_df.index
    
    metrics_cols = [
        "epoch",
        "val_tumor_presence_recall",
        "val_tumor_presence_f1_score",
        "val_tumor_type_meningioma_recall",
        "val_masked_macro_f1", # head tumor type
        "val_tumor_type_masked_accuracy",
    ]
    
    df = history_df[metrics_cols].copy()
    
    df = df[
        (df["val_tumor_presence_recall"] >= 0.98) &
        (df["val_tumor_type_masked_accuracy"] >= 0.84)  &
        (df["val_tumor_type_meningioma_recall"] >= 0.54) 
    ].copy()
    
    df["S"] = (
        0.60 * df["val_tumor_presence_recall"]
      + 0.20 * df["val_tumor_type_meningioma_recall"]
      + 0.15 * df["val_masked_macro_f1"]
    )
    
    if df.empty:
        raise ValueError("No valid epoch found with given constraints")
    best_row = df.sort_values("S", ascending=False).iloc[0]
    best_epoch = int(best_row["epoch"])
    
    print(f"✅ Best epoch selected from S: {best_epoch}")
    print(best_row)
    
    return best_row, best_epoch



""" dataset preprox """


def get_datasets(processed_dir, dataset_name='Training', batch_size=32):
    """
    dataset_name : Training, Validation, Testing
    """
    path = processed_dir + "/" + dataset_name
    
    ds = load_tfrecord_dataset(
        path,
        shuffle=True,
        batch_size=batch_size,
        repeat=False
    ).prefetch(tf.data.AUTOTUNE)
    
    ds = ds.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds






def main():
    """Main entry point for the process."""
    logger.info("=== Use Model Script Started ===")

    config = load_config(CONFIG_PATH)
    
    os.makedirs(config["path"]["output_dir"] + "/correct", exist_ok=True)
    os.makedirs(config["path"]["output_dir"] + "/errors", exist_ok=True)
    
    processed_dir = config["path"]["processed_dir"]
    batch_size = config["model"]["batch_size"]
    img_size = config["data_preprocessing"]["img_size"]
    model_dir = config["model"]["models_dir"]
    seed = config["general"]["seed"]
    
    setup_MLFlow(traking_url=get_mlflow_link())
    setup_tensorflow()
    
    unfreeze_layer = config["model"]["unfreeze_layer"]
    model_type = config["model"]["model_type"]
    project_name = config["model"]["project_name"]
    model_type = config["model"]["model_type"]
    two_head = config["model"]["two_head"]
    checkpoint_dir = config["path"]["checkpoint_dir"]
    model_name = f"{project_name}_{model_type}_{'2Head' if two_head else '1Head'}"
    
    """Model rebuild"""
    model = get_model_built(
        img_size,
        model_dir,
        freeze_backbone=True,
        seed=seed
    )
    model, loss_weight_presence, loss_weight_type = compile_model(model, masked_sparse_cce)
    
    """ datasets """
    train_ds = get_datasets(processed_dir, dataset_name='Training', batch_size=batch_size)
    val_ds = get_datasets(processed_dir, dataset_name='Validation', batch_size=batch_size)
    
    
    """ Head training """
    history = run_training_heads(model, loss_weight_presence, loss_weight_type,\
                     config, train_ds, val_ds)
    best_row, best_epoch = keep_best_epoch_heads(history)
    save_best_model(model, best_row, best_epoch, \
                        checkpoint_dir, model_name, model_dir)
        
    """Model rebuild"""
    model = get_model_built(
        img_size,
        model_dir,
        freeze_backbone=False
    )
    
    model.load_weights(os.path.join(model_dir, "brain_tumor_heads.weights.h5"))
    
    """ Fine-tuning """
    backbone = model.get_layer("densenet121")
    backbone = unfreeze_layers(backbone, unfreeze_layer)
    
    model, loss_weight_presence, loss_weight_type = compile_model(model, masked_sparse_cce)
    history = run_training_finetuning(model, loss_weight_presence, loss_weight_type,\
                     config, train_ds, val_ds)
    best_row, best_epoch = keep_best_epoch_finetuning(history)
    save_best_model(model, best_row, best_epoch, \
                        checkpoint_dir, model_name, model_dir)
    
    

    logger.info("=== Use Model Script Completed Successfully ===")


if __name__ == "__main__":
    main()

