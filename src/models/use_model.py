#!/usr/bin/env python3
"""
use_model.py

Functions to load and use model.
Reads configuration from config.yaml and credentials from .secrets/kaggle.json.

This script is designed for integration into an MLOps pipeline.
"""

import os
import yaml
import logging
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from scipy.stats import norm

import shap

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
import math
import random

import cv2
from collections import defaultdict
from typing import Tuple, Optional, Union


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


def setup_tensorflow(debug=False):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if debug:
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    tf.config.list_physical_devices('GPU')


def get_backbone(img_size, models_dir, freeze_backbone, debug=False):
    # 1. Create DenseNet121 WITHOUT weights
    backbone = DenseNet121(
        include_top=False,
        weights=None,
        input_shape=(img_size, img_size, 3)
    )
    
    # 2. Load RadImageNet weights
    backbone.load_weights(models_dir + "/RadImageNet-DenseNet121_notop.h5")
    
    # 3. Freeze the backbone for firsts training
    backbone.trainable = not freeze_backbone
    
    if debug:
        print("✅ RadImageNet DenseNet121 loaded successfully")
    
    return backbone


def get_model_data_augmentation(x, seed):
    x = layers.RandomFlip("horizontal", seed=seed)(x)
    x = layers.RandomZoom((-0.03,0.03),(-0.03,0.03), seed=seed)(x)
    x = layers.RandomTranslation((-0.01,0.01),(-0.01,0.01), seed=seed)(x)
    return x


def get_model_head_presence(x):
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1,activation='sigmoid',name="tumor_presence")(x)
    return x


def get_model_head_type(x):
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(4,activation='softmax',name="tumor_type")(x)
    return x


def shared_head_part(inputs, backbone):
    # Data augmentation (training only)
    x = get_model_data_augmentation(inputs)
    # Backbone - force into inference
    x = backbone(x, training=False)

    # Copy backbone output exactly as a proxy layer
    x = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name='Top_Conv_Layer', trainable=False)(x)
    
    # Shared head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    return x


def assemble_heads(img_size, backbone):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    
    x = shared_head_part(inputs, backbone)
    
    #Heads
    output_presence = get_model_head_presence(x)
    output_type = get_model_head_type(x)
    
    model = keras.Model(
        inputs=inputs,
        outputs={
            "tumor_presence": output_presence,
            "tumor_type": output_type
        },
        name='densenet_two_head'
    )

    return model


def get_loss_presence():
    return keras.losses.BinaryFocalCrossentropy(
        gamma=2.0,
        alpha=0.25 # to favorize tumor detection (penalize false negatives), but taking account that tumors are 75% of data
    )


@tf.keras.utils.register_keras_serializable()
def masked_sparse_cce(y_true, y_pred):
    tumor_present = tf.cast(y_true != 0, tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    loss = loss * tumor_present
    return tf.reduce_sum(loss) / (tf.reduce_sum(tumor_present) + 1e-6)


@tf.keras.utils.register_keras_serializable()
class MaskedSparseCategoricalAccuracy(tf.keras.metrics.Metric): # calculate accuracy, excluding "no_tumor" (because of mask)
    def __init__(self, name="masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # mask: only tumor
        mask = tf.cast(y_true != 0, tf.float32)

        y_pred_labels = tf.argmax(y_pred, axis=-1)
        matches = tf.cast(tf.equal(tf.cast(y_true, tf.int64), y_pred_labels), tf.float32)

        matches = matches * mask

        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total / (self.count + 1e-6)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)



@tf.keras.utils.register_keras_serializable()
class MeningiomaRecall(tf.keras.metrics.Metric):
    def __init__(self, name="meningioma_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = sparse labels (batch,)
        y_true = tf.cast(y_true, tf.int32)

        # y_pred = logits ou probabilités, shape (batch, num_classes)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        # Meningioma class index = 2
        meningioma_mask = tf.equal(y_true, 2)

        tp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(y_pred, 2), meningioma_mask), tf.float32)
        )
        fn = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.not_equal(y_pred, 2), meningioma_mask), tf.float32)
        )

        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        return self.true_positives / (self.true_positives + self.false_negatives + 1e-8)

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)
        
        
@tf.keras.utils.register_keras_serializable()
class BinaryF1(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        return 2 * precision * recall / (precision + recall + 1e-7)

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
        
        
def compile_model(model, masked_sparse_cce):
    loss_presence = get_loss_presence()
    
    loss_weight_presence = 1.0
    loss_weight_type = 1.3 # we give a little more weight to the classification of the type
    
    model.compile(
        optimizer=keras.optimizers.Adam(), # change learning_rate for 1e-4 in fine-tuning steps
        loss={
            "tumor_presence": loss_presence,
            "tumor_type": masked_sparse_cce,
        },
        
        loss_weights={
            "tumor_presence": loss_weight_presence,
            "tumor_type": loss_weight_type, 
        },
        
        metrics={
            "tumor_presence": [
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.Precision(name="precision"),
                BinaryF1(name="f1_score"),
                keras.metrics.AUC(name="auc")
            ],
            "tumor_type": [
                #"accuracy", 
                MaskedSparseCategoricalAccuracy(name="masked_accuracy"),
                MeningiomaRecall(),
            ],
        }
    )

    return model, loss_weight_presence, loss_weight_type


def get_model_built(img_size, models_dir, freeze_backbone):
    
    backbone = get_backbone(img_size, models_dir, freeze_backbone)
    model = assemble_heads(img_size, backbone)

    return model


""" Get y """
def get_y_presence(model, dataset):
    y_true_pres = []
    y_pred_pres = []
    y_prob_pres = []
    
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_true_pres.extend(y_batch["tumor_presence"].numpy().astype(int).flatten())
        y_pred_pres.extend((preds['tumor_presence'] > 0.5).astype(int).flatten())
        y_prob_pres.extend(preds["tumor_presence"].flatten())
        
    return y_true_pres, y_pred_pres, np.array(y_prob_pres)


def get_y_type(model, dataset):
    y_true_type = []
    y_pred_type = []
    
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_true_type.extend(y_batch['tumor_type'].numpy())
        y_pred_type.extend(preds['tumor_type'].argmax(axis=-1))
            
    return y_true_type, y_pred_type


def get_y_for_class(model, dataset, class_index):
    y_true = []
    y_prob = []

    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)

        y_true_batch = y_batch["tumor_type"].numpy()
        prob_batch = preds["tumor_type"]

        y_true.extend((y_true_batch == class_index).astype(int))
        y_prob.extend(prob_batch[:, class_index])

    return np.array(y_true), np.array(y_prob)
    

""" Confusion matrix """
def get_presence_confusion_matrix_plot(y_true_pres, y_pred_pres, presence_classes, path):
    cm = confusion_matrix(y_true_pres, y_pred_pres)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=presence_classes)
    name = 'Presence confusion matrix'
    disp.plot(cmap='Blues').savefig(path+'/'+name)


def get_type_confusion_matrix_plot(y_true_type, y_pred_type, type_classes, path):
    cm = confusion_matrix(y_true_type, y_pred_type)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=type_classes)
    name = 'Type confusion matrix'
    disp.plot(cmap='Blues').savefig(path+'/'+name)
    

""" Confidence intervals """

def wilson_ci(successes, n, confidence=0.95):
    if n == 0:
        return (np.nan, np.nan)

    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = (
        z * np.sqrt((p*(1-p) + z**2/(4*n)) / n)
    ) / denominator
    
    return center - margin, center + margin


def bootstrap_ci(y_true, y_pred, metric_fn, 
                 n_boot=1000, confidence=0.95, random_state=42):
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    rng = np.random.RandomState(random_state)
    scores = []
    n = len(y_true)
    
    for _ in range(n_boot):
        indices = rng.choice(n, n, replace=True)
        score = metric_fn(
            y_true[indices],
            y_pred[indices]
        )
        scores.append(score)
    
    lower = np.percentile(scores, (1 - confidence) / 2 * 100)
    upper = np.percentile(scores, (1 + confidence) / 2 * 100)
    
    return lower, upper


def evaluate_with_ci(y_true, y_pred,
    average="macro",   # "binary", "macro", "weighted"
    confidence=0.95):
    
    results = {}
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    n = len(y_true)
    acc_ci = wilson_ci(int(acc*n), n, confidence)
    
    results["accuracy"] = {
        "value": acc,
        "ci": acc_ci
    }
    
    # Precision
    prec = precision_score(y_true, y_pred, average=average)
    prec_ci = bootstrap_ci(
        y_true, y_pred,
        lambda yt, yp: precision_score(yt, yp, average=average),
        confidence=confidence
    )
    
    results["precision"] = {
        "value": prec,
        "ci": prec_ci
    }
    
    # Recall
    rec = recall_score(y_true, y_pred, average=average)
    rec_ci = bootstrap_ci(
        y_true, y_pred,
        lambda yt, yp: recall_score(yt, yp, average=average),
        confidence=confidence
    )
    
    results["recall"] = {
        "value": rec,
        "ci": rec_ci
    }
    
    # F1
    f1 = f1_score(y_true, y_pred, average=average)
    f1_ci = bootstrap_ci(
        y_true, y_pred,
        lambda yt, yp: f1_score(yt, yp, average=average),
        confidence=confidence
    )
    
    results["f1"] = {
        "value": f1,
        "ci": f1_ci
    }
    
    return results


def mask_no_tumor(y_true, y_pred, no_tumor_label=0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != no_tumor_label

    return y_true[mask], y_pred[mask]


def per_class_recall_ci(y_true, y_pred, confidence=0.95):
    
    cm = confusion_matrix(y_true, y_pred)
    recalls = {}
    
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        n = tp + fn
        
        recalls[i] = {
            "value": tp / n,
            "ci": wilson_ci(tp, n, confidence)
        }
    
    return recalls


def get_y_masked(y_true, y_pred):
    """
    y are from type head.
    """
    
    y_true_type_masked, y_pred_type_masked = mask_no_tumor(
        y_true,
        y_pred
    )
    return y_true_type_masked, y_pred_type_masked


def print_CI(results_dict, classes=None):
    for stat_name, stats in results_dict.items():
        if classes:
            print(f"\nClass: {classes[stat_name+1]}") # stat_name == class_idx
        else:
            print(f"\nStat: {stat_name}")
        print(f"Recall: {stats['value']:.4f}")
        print(f"95% CI: [{stats['ci'][0]:.4f}, {stats['ci'][1]:.4f}]")


def get_presence_CI(y_true, y_pred, print_results=False):
    presence_results = evaluate_with_ci(
        y_true,
        y_pred,
        average="binary"
    )
    
    if print_results:
        print_CI(presence_results)
        
    return presence_results


def get_type_CI(y_true, y_pred, print_results=False):
    y_true_type_masked, y_pred_type_masked = get_y_masked(y_true, y_pred)
    
    type_results = evaluate_with_ci(
        y_true_type_masked,
        y_pred_type_masked,
        average="macro"
    )
    
    if print_results:
        print_CI(type_results)
        
    return type_results


def get_classes_CI(y_true, y_pred, classes, print_results=False):
    y_true_type_masked, y_pred_type_masked = get_y_masked(y_true, y_pred)
    
    recall_stats = per_class_recall_ci(
        y_true_type_masked,
        y_pred_type_masked,
        confidence=0.95
    )
    
    if print_results:
        print_CI(recall_stats, classes)
        
    return recall_stats


""" Calibration """
def compute_calibration_metrics(y_true, y_prob, n_bins=10):

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="uniform"
    )

    # ECE
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    ece = 0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(y_true)

    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "ece": ece,
        "brier": brier
    }


def plot_reliability_diagram(prob_true, prob_pred, ece, brier, path, variable=""):

    plt.figure()

    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1], linestyle="--")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")

    plt.title(f"Reliability Diagram for {variable}")

    plt.text(0, 0.9, f'ECE={round(ece, 3)}\nBrier={round(brier,3)}', fontsize=10)

    name = f'Reliability Diagram for {variable}'
    plt.savefig(path+'/'+name)
    
    
def run_reliability_stats(y_true, y_prob, path, variable=""):
    results = compute_calibration_metrics(
        y_true,
        y_prob
    )
    
    fig = plot_reliability_diagram(
        results["prob_true"],
        results["prob_pred"],
        results["ece"],
        results["brier"],
        path,
        variable
    )
    
    return results, fig


def run_reliability_by_class(classes, model, dataset, path):
    for i in range(1, len(classes)):
        y_true, y_prob = get_y_for_class(model, dataset, i)
        run_reliability_stats(y_true, y_prob, path, variable=classes[i])


""" Grad-CAM """

def build_gradcam_model(model, head_name='tumor_presence'):
    """
    Build a Grad-CAM model directly connected to the backbone + frozen Conv2D layer.
    
    Args:
        model: Original multi-head model
        head_name: 'tumor_presence' or 'tumor_type'
    
    Returns:
        gradcam_model: tf.keras.Model with outputs [Top_Conv_Layer, selected head]
    """
    # Output of frozen Conv2D
    conv_layer = model.get_layer('Top_Conv_Layer').output

    # Output of the selected head
    if head_name == 'tumor_presence':
        head_output = model.get_layer('tumor_presence').output
    elif head_name == 'tumor_type':
        head_output = model.get_layer('tumor_type').output
    else:
        raise ValueError(f"Unknown head_name {head_name}")
    
    gradcam_model = Model(inputs=model.inputs, outputs=[conv_layer, head_output])
    return gradcam_model


def compute_gradcam(gradcam_model, img, target_head_name):
    """
    Compute standard Grad-CAM heatmap.
    """

    if len(img.shape) == 3:
        img = tf.expand_dims(img, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, preds = gradcam_model(img, training=False)

        if target_head_name == 'tumor_presence':
            loss = preds[:, 0]
        else:
            loss = tf.reduce_max(preds, axis=-1)

    # Gradient w.r.t. conv layer
    grads = tape.gradient(loss, conv_outputs)

    # 1️⃣ Global Average Pooling of gradients
    weights = tf.reduce_mean(grads, axis=(1, 2))

    # 2️⃣ Weighted sum of feature maps
    cam = tf.reduce_sum(
        weights[:, tf.newaxis, tf.newaxis, :] * conv_outputs,
        axis=-1
    )

    # 3️⃣ ReLU
    cam = tf.nn.relu(cam)

    # 4️⃣ Normalize
    cam = cam[0].numpy()
    p_low, p_high = np.percentile(cam, (5, 95))
    cam = np.clip((cam - p_low) / (p_high - p_low + 1e-8), 0, 1)

    return cam


def compute_gradcam_pp(gradcam_model, img, target_head_name):
    """
    Compute Grad-CAM++ heatmap for a single image.
    
    Args:
        gradcam_model: model with outputs [Top_Conv_Layer, target_head_output]
        img: tf.Tensor, shape (H,W,3) or (1,H,W,3)
    
    Returns:
        cam: numpy array, normalized heatmap
    """
    # Add batch dimension if necessary
    if len(img.shape) == 3:
        img = tf.expand_dims(img, axis=0)
    
    with tf.GradientTape() as tape:
        tape.watch(img)
        # Compute forward pass
        conv_outputs, preds = gradcam_model(img, training=False)
        
        # Select target for gradient
        if target_head_name == 'tumor_presence':
            loss = preds[:, 0]
        else:
            # For multi-class, pick max logit
            loss = tf.reduce_max(preds, axis=-1)
    
    # Gradients w.r.t. conv layer
    grads = tape.gradient(loss, conv_outputs)
    
    # Grad-CAM++ alpha weights
    alpha_num = grads ** 2
    alpha_denom = 2 * grads ** 2 + tf.reduce_sum(conv_outputs * grads ** 3, axis=(1,2), keepdims=True)
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    alpha = alpha_num / alpha_denom
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1,2))
    cam = tf.reduce_sum(weights[:, tf.newaxis, tf.newaxis, :] * conv_outputs, axis=-1)
    
    # Normalize
    cam = tf.nn.relu(cam)
    cam = cam - tf.reduce_min(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = cam[0].numpy()  # remove batch dimension
    
    return cam


def overlay_gradcam(original_image, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image, supports single or batched images.

    Args:
        original_image: numpy array or tf.Tensor, shape (H, W, 3) or (1, H, W, 3), values in [0,1]
        heatmap: numpy array or tf.Tensor, shape (h, w), values in [0,1]
        alpha: blending factor for overlay

    Returns:
        overlayed image, uint8, shape (H, W, 3)
    """
    # Convert TensorFlow tensors to numpy
    if isinstance(original_image, tf.Tensor):
        original_image = original_image.numpy()
    if isinstance(heatmap, tf.Tensor):
        heatmap = heatmap.numpy()

    # Remove batch dimension if present
    if original_image.ndim == 4:
        original_image = original_image[0]

    # Ensure float32 and clip values
    original_image = np.clip(original_image.astype(np.float32), 0, 1)
    heatmap = np.clip(heatmap.astype(np.float32), 0, 1)

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # Apply color map
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    #original_image -= original_image.min()
    #original_image /= (original_image.max() + 1e-8)
    # Overlay
    overlay = cv2.addWeighted(
        np.uint8(255 * original_image),
        1 - alpha,
        heatmap_colored,
        alpha,
        0
    )

    return overlay


def get_grad_cam_overlay_img(model, tensor_img, head_name='tumor_presence', grad_cam_function=compute_gradcam):
    model.training = False
    gradcam_model = build_gradcam_model(model, head_name)
    
    # Ensure batch dimension
    if len(tensor_img.shape) == 3:
        tensor_img = tf.expand_dims(tensor_img, 0)
    
    image_tensor = tf.cast(tensor_img, tf.float32)
    image_tensor = tf.Variable(image_tensor)  # watch for gradients
    
    heatmap = grad_cam_function(
        gradcam_model,
        image_tensor,
        target_head_name=head_name
    )

    orig_img = image_tensor[0].numpy()
    overlay = overlay_gradcam(orig_img, heatmap)

    plt.imshow(orig_img)
    plt.title("Original")
    plt.axis("off")
    plt.show()
    
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.show()












def run_medical_XAI_pipeline(model, dataset, config):
    """
    Run medical pipeline for the whole dataset
    """
    classes = config["general"]["classes"]
    output_dir = config['path']['output_dir']
    
    y_true_pres, y_pred_pres, y_prob_pres = get_y_presence(model, dataset)
    y_true_type, y_pred_type = get_y_type(model, dataset)
    
    get_presence_confusion_matrix_plot(y_true_pres, y_pred_pres, config["general"]["presence_cat"])
    get_type_confusion_matrix_plot(y_true_type, y_pred_type, classes)
    
    ci_pres_dct = get_presence_CI(y_true_pres, y_pred_pres, print_results=True)
    ci_type_dct = get_type_CI(y_true_type, y_pred_type, print_results=True)
    ci_classes_dct = get_classes_CI(y_true_type, y_pred_type, classes=config["general"]["classes"], print_results=True)
    
    run_reliability_stats(y_true_pres, y_prob_pres,output_dir, variable="presence")
    run_reliability_by_class(classes, model, dataset, output_dir)
    



def main():
    """Main entry point for the process."""
    logger.info("=== Use Model Script Started ===")

    config = load_config(CONFIG_PATH)
    
    os.makedirs(config["path"]["output_dir"] + "/correct", exist_ok=True)
    os.makedirs(config["path"]["output_dir"] + "/errors", exist_ok=True)
    
    debug = config["general"]["debug"]
    img_size = config["data_preprocessing"]["img_size"]
    model_dir = config["model"]["models_dir"]
    
    """Model rebuild"""
    model = get_model_built(
        img_size,
        model_dir,
        FREEZE_BACKBONE=False
    )
    
    model.load_weights(
        model_dir + "/brain_tumor_heads.weights.h5"
    )
    
    if debug:
        print("✅ Model reconstructed + weights loaded")
    
    
    

    logger.info("=== Use Model Script Completed Successfully ===")


if __name__ == "__main__":
    main()
