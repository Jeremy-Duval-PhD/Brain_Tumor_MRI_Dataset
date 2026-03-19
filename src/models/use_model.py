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
from tensorflow.keras.models import Model

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from scipy.stats import norm
import shap

import matplotlib.pyplot as plt
import random

import cv2


from make_model import get_model_built, setup_tensorflow, get_datasets


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
    

""" SHAP """
def build_shap_head_model(model):
    """
    Model that takes Top_Conv_Layer feature maps as input
    and outputs tumor_presence prediction.
    """

    conv_layer = model.get_layer("Top_Conv_Layer")

    x = conv_layer.output

    # continue the model after the conv layer
    for layer in model.layers[model.layers.index(conv_layer)+1:]:
        x = layer(x)

    head_model = tf.keras.Model(
        inputs=conv_layer.output,
        outputs=x
    )

    return head_model


def compute_shap_map(shap_values, img_size):

    shap_values = np.array(shap_values)

    # remove batch dim if present
    if shap_values.ndim == 5:
        shap_values = shap_values[0]

    # remove output dim if present
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    # now should be (H,W,3)
    shap_map = np.mean(np.abs(shap_values), axis=-1)

    # normalization (GradCAM style)
    epsilon = 1e-6
    shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + epsilon)

    shap_map = np.power(shap_map, 0.5)  # amplify small contributions

    shap_map = cv2.resize(shap_map, (img_size, img_size))

    return shap_map


def build_stratified_background(
    dataset,
    samples_per_class=10,
    save_path="background.npy",
    use_save_background=False,
    debug=False
):
    """
    Build a stratified SHAP background dataset based on tumor_type.

    Args:
        dataset: tf.data.Dataset
        samples_per_class: number of images per class
        save_path: path to save numpy file

    Returns:
        background_images: numpy array
    """
    if use_save_background:
        background_images = np.load("background.npy")
        print(f"\n✅ Background loaded from {save_path}")
    else:
        class_buckets = {}
    
        # -------------------------
        # Collect images per class
        # -------------------------
        for x_batch, y_batch in dataset:
            images = x_batch.numpy()
            labels = y_batch["tumor_type"].numpy()
    
            for i in range(len(images)):
                cls = int(labels[i])
    
                if cls not in class_buckets:
                    class_buckets[cls] = []
    
                class_buckets[cls].append(images[i])
    
        # -------------------------
        # Sampling
        # -------------------------
        background_images = []
    
        for cls in sorted(class_buckets.keys()):
            imgs = class_buckets[cls]
    
            n = min(samples_per_class, len(imgs))
            selected = random.sample(imgs, n)
    
            background_images.extend(selected)
            if debug:
                print(f"Class {cls}: {len(selected)} samples")
    
        background_images = np.array(background_images)
    
        # -------------------------
        # Save
        # -------------------------
        np.save(save_path, background_images)
        if debug:
            print(f"\n✅ Background saved to {save_path}")
            print("Shape:", background_images.shape)

    return background_images


def get_presence_explainer(model, background_images):
    presence_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.output["tumor_presence"]
    )
    
    explainer_presence = shap.GradientExplainer(
        presence_model,
        background_images
    )
    
    return explainer_presence


def get_type_explainer(model, background_images, pred_class, type_explainers_cache={}):

    if pred_class in type_explainers_cache:
        return type_explainers_cache[pred_class], type_explainers_cache

    type_class_model = tf.keras.Model(
        inputs=model.input,
        outputs=tf.keras.layers.Lambda(
            lambda x: x[:, pred_class:pred_class+1]
        )(model.get_layer("tumor_type").output)
    )

    explainer = shap.GradientExplainer(
        type_class_model,
        background_images
    )

    type_explainers_cache[pred_class] = explainer

    return explainer, type_explainers_cache


""" Gathered visualization """

def normalize_for_display(img):
    img = img.astype(np.float32)
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img


def compute_agreement_map(grad_map, shap_map):

    # -------------------------
    # Resize Grad-CAM
    # -------------------------
    grad_map_resized = cv2.resize(
        grad_map,
        (shap_map.shape[1], shap_map.shape[0])
    )

    # -------------------------
    # Normalisation (CRUCIALE)
    # -------------------------
    grad_map_resized = grad_map_resized.astype(np.float32)
    grad_map_resized = grad_map_resized - grad_map_resized.min()
    grad_map_resized = grad_map_resized / (grad_map_resized.max() + 1e-8)

    shap_map = shap_map.astype(np.float32)
    shap_map = shap_map - shap_map.min()
    shap_map = shap_map / (shap_map.max() + 1e-8)

    # -------------------------
    # Agreement
    # -------------------------
    agreement = grad_map_resized * shap_map

    # amplification
    agreement = np.power(agreement, 0.5)

    return agreement


def visualize_explanations(
    model,
    img,
    img_size,
    background_images,
    explainer_presence=None,
    head="tumor_presence",
    nsamples=100,
    true_label=None,
    classes=None,
    path="",
    type_explainers_cache={}
):

    img_display = normalize_for_display(img)

    # ----------------
    # Prediction
    # ----------------
    pred = model.predict(img[np.newaxis,...], verbose=0)

    if head == "tumor_presence":
        pred_prob = float(pred["tumor_presence"][0][0])
        pred_class = int(pred_prob > 0.5)

    else:
        pred_probs = pred["tumor_type"][0]
        pred_class = int(np.argmax(pred_probs))
        pred_prob = float(pred_probs[pred_class])

    # ----------------
    # Title
    # ----------------
    if classes is not None:
        pred_name = classes[pred_class]

        if true_label is not None:
            true_name = classes[true_label]
            title_text = f"TRUE: {true_name}\nPRED: {pred_name} ({pred_prob:.2f})"
        else:
            title_text = f"PRED: {pred_name} ({pred_prob:.2f})"
    else:
        title_text = f"PRED: {pred_class} ({pred_prob:.2f})"

    # ----------------
    # Plot
    # ----------------
    fig, axes = plt.subplots(1,4, figsize=(16,4))

    # ----------------
    # Original
    # ----------------
    axes[0].imshow(img_display)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # ----------------
    # GradCAM
    # ----------------
    gradcam_model = build_gradcam_model(model, head)

    grad_map = compute_gradcam(
        gradcam_model,
        img[np.newaxis,...],
        head
    )

    grad_overlay = overlay_gradcam(img_display, grad_map)

    axes[1].imshow(grad_overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    # ----------------
    # SHAP
    # ----------------
    if head == "tumor_presence":

        with tf.device("/CPU:0"):
            shap_values = explainer_presence.shap_values(
                img[np.newaxis,...],
                nsamples=nsamples
            )

    else:
        pred_class = int(np.argmax(pred["tumor_type"][0]))

        explainer_type, type_explainers_cache = get_type_explainer(
            model,
            background_images,
            pred_class,
            type_explainers_cache=type_explainers_cache
        )

        with tf.device("/CPU:0"):
            shap_values = explainer_type.shap_values(
                img[np.newaxis,...],
                nsamples=nsamples
            )

    shap_map = compute_shap_map(shap_values, img_size)

    shap_overlay = overlay_gradcam(img_display, shap_map)

    axes[2].imshow(shap_overlay)
    axes[2].set_title("SHAP")
    axes[2].axis("off")

    # ----------------
    # Agreement
    # ----------------
    agreement_map = compute_agreement_map(grad_map, shap_map)

    agreement_overlay = overlay_gradcam(img_display, agreement_map)

    axes[3].imshow(agreement_overlay)
    axes[3].set_title("GradCAM × SHAP")
    axes[3].axis("off")
    plt.suptitle(title_text, fontsize=14)
    plt.tight_layout()
    
    safe_name = title_text.replace("\n", "_").replace(" ", "_")
    plt.savefig(os.path.join(path, safe_name + ".png"))
    
    return type_explainers_cache


def get_explanations_for_confusion_mtrx_presence(
    model,
    dataset,
    background_images,
    img_size,
    explainer_presence,
    presence_cat,
    nb_ex_by_cat=1,
    path="",
    type_explainers_cache={}
):
    confusion_mtrx_elm = {
        "TP": [],
        "TN": [],
        "FP": [],
        "FN": []
    }
    
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        
        y_true_batch = y_batch["tumor_presence"].numpy().astype(int).flatten()
        y_pred_batch = (preds["tumor_presence"] > 0.5).astype(int).flatten()
        
        for i in range(len(x_batch)):
            true = y_true_batch[i]
            pred = y_pred_batch[i]
            img = x_batch[i].numpy()
            
            if true == 1 and pred == 1:
                confusion_mtrx_elm["TP"].append((img, true))
            elif true == 0 and pred == 0:
                confusion_mtrx_elm["TN"].append((img, true))
            elif true == 0 and pred == 1:
                confusion_mtrx_elm["FP"].append((img, true))
            elif true == 1 and pred == 0:
                confusion_mtrx_elm["FN"].append((img, true))
    
    # -------------------------
    # Visualisation
    # -------------------------
    for key in confusion_mtrx_elm:
        val_lst = confusion_mtrx_elm[key]
        exemples = random.sample(val_lst, min(nb_ex_by_cat, len(val_lst)))


        for img, true in exemples:
            type_explainers_cache = visualize_explanations(
                model,
                img,
                img_size,
                background_images,
                explainer_presence,
                head="tumor_presence",
                true_label=true,
                classes=presence_cat,
                path=path,
                type_explainers_cache=type_explainers_cache
            )
            
    return type_explainers_cache
        
        
def get_explanations_for_confusion_mtrx_type(
    model,
    dataset,
    background_images,
    classes,
    nb_ex_by_cat=1,
    skip_no_tumor_cat=True,
    path="",
    type_explainers_cache={}
):    
    head_name = "tumor_type"
    confusion_examples = {}

    # -------------------------
    # Collecte
    # -------------------------
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        
        y_true_batch = y_batch[head_name].numpy()
        y_pred_batch = preds[head_name].argmax(axis=-1)

        for i in range(len(x_batch)):
            true_class = int(y_true_batch[i])

            if true_class == 0 and skip_no_tumor_cat:
                continue

            pred_class = int(y_pred_batch[i])
            img = x_batch[i].numpy()
            
            key = f"{classes[true_class]}__{classes[pred_class]}"
            
            if key not in confusion_examples:
                confusion_examples[key] = []

            confusion_examples[key].append((img, true_class))

    # -------------------------
    # Visualisation
    # -------------------------
    for key in confusion_examples:
        val_lst = confusion_examples[key]
        exemples = random.sample(val_lst, min(nb_ex_by_cat, len(val_lst)))

        true_name, pred_name = key.split("__")

        for img, true_class in exemples:

            type_explainers_cache = visualize_explanations(
                model,
                img,
                background_images,
                head="tumor_type",
                true_label=true_class,
                classes=classes,
                path=path,
                type_explainers_cache=type_explainers_cache
            )

    return type_explainers_cache






def run_medical_XAI_one_image(img, model, background_images, explainer_presence, \
                              output_dir, classes, type_explainers_cache={}):
    _ = visualize_explanations(
        model,
        img,
        background_images,
        explainer_presence,
        head="tumor_presence",
        true_label=1,
        classes=["no_tumor", "tumor"],
        path=output_dir
    )
    
    type_explainers_cache = visualize_explanations(
        model,
        img,
        background_images,
        head="tumor_type",
        true_label=2,
        classes=classes,
        path=output_dir,
        type_explainers_cache=type_explainers_cache
    )
    
    return type_explainers_cache



def run_medical_XAI_pipeline(model, dataset, config, type_explainers_cache={}):
    """
    Run medical pipeline for the whole dataset
    """
    classes = config["general"]["nb_img_shap"]
    presence_cat = config["general"]["presence_cat"]
    output_dir = config['path']['output_dir']
    img_size = config['data_preprocessing']['img_size']
    nb_img_shap = config["model"]["nb_img_shap"]
    
    y_true_pres, y_pred_pres, y_prob_pres = get_y_presence(model, dataset)
    y_true_type, y_pred_type = get_y_type(model, dataset)
    
    get_presence_confusion_matrix_plot(y_true_pres, y_pred_pres, config["general"]["presence_cat"], output_dir)
    get_type_confusion_matrix_plot(y_true_type, y_pred_type, classes)
    
    ci_pres_dct = get_presence_CI(y_true_pres, y_pred_pres, print_results=True)
    ci_type_dct = get_type_CI(y_true_type, y_pred_type, print_results=True)
    ci_classes_dct = get_classes_CI(y_true_type, y_pred_type, classes=config["general"]["classes"], print_results=True)
    
    run_reliability_stats(y_true_pres, y_prob_pres,output_dir, variable="presence")
    run_reliability_by_class(classes, model, dataset, output_dir)
    
    background_images = build_stratified_background(
        dataset,
        samples_per_class=10,
        save_path="background.npy",
        use_save_background=True
    )
    explainer_presence = get_presence_explainer(model, background_images)
    
    count=0
    for x_batch, _ in dataset:
        for img in x_batch:
            if count >= nb_img_shap:
                break
            img = img.numpy()
            type_explainers_cache = run_medical_XAI_one_image(img, model,\
                                          background_images, explainer_presence, \
                                          output_dir, classes, type_explainers_cache)
            count+=1
    
    mtrx_path = output_dir + "/matrix"
    os.makedirs(mtrx_path, exist_ok=True)
    
    type_explainers_cache = get_explanations_for_confusion_mtrx_presence(
        model,
        dataset,
        background_images,
        img_size,
        explainer_presence,
        presence_cat,
        nb_ex_by_cat=1,
        path=mtrx_path,
        type_explainers_cache=type_explainers_cache
    )
    
    type_explainers_cache = get_explanations_for_confusion_mtrx_type(
        model,
        dataset,
        background_images,
        classes,
        nb_ex_by_cat=1,
        path=mtrx_path,
        type_explainers_cache=type_explainers_cache
    )
    
    
    return type_explainers_cache
    



def main():
    """Main entry point for the process."""
    logger.info("=== Use Model Script Started ===")

    config = load_config(CONFIG_PATH)
    
    os.makedirs(config["path"]["output_dir"] + "/correct", exist_ok=True)
    os.makedirs(config["path"]["output_dir"] + "/errors", exist_ok=True)
    
    debug = config["general"]["debug"]
    img_size = config["data_preprocessing"]["img_size"]
    model_dir = config["model"]["models_dir"]
    
    processed_dir = config["path"]["processed_dir"]
    batch_size = config["model"]["batch_size"]
    
    setup_tensorflow()
    
    """Model rebuild"""
    model = get_model_built(
        img_size,
        model_dir,
        freeze_backbone=False
    )
    
    model.load_weights(os.path.join(model_dir, "brain_tumor_heads.weights.h5"))
    
    if debug:
        print("✅ Model reconstructed + weights loaded")
    
    type_explainers_cache={}
    dataset = get_datasets(processed_dir, dataset_name='Testing', batch_size=batch_size)
    run_medical_XAI_pipeline(model, dataset, config, type_explainers_cache=type_explainers_cache)
    

    logger.info("=== Use Model Script Completed Successfully ===")


if __name__ == "__main__":
    main()
