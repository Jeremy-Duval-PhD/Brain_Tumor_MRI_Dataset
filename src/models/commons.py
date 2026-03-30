import tensorflow as tf
from tensorflow.keras.models import Model
import shap
import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings



def split_labels(image, label):
    """
    Create labels for a 2-head model.
    """
    tumor_present = tf.cast(label != 0, tf.float32)
    tumor_type = tf.cast(label, tf.int32)

    return image, {
        "tumor_presence": tumor_present,
        "tumor_type": tumor_type
    }


def parse_tfrecord(example_proto):
    """
    Parse a single TFRecord example and convert grayscale → RGB.
    """
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    # Deserialize image
    image = tf.io.parse_tensor(example["image"], out_type=tf.float32)

    # Shape after loading: (260, 260, 3)
    image.set_shape((260, 260, 3))

    label = tf.cast(example["label"], tf.int32)

    return image, label


def load_tfrecord_dataset(tfrecord_dir, shuffle=False, batch_size=1, repeat=False):
    """
    Load a TFRecord dataset from a directory.

    Args:
        tfrecord_dir (str or Path): Folder containing .tfrecord files
        shuffle (bool): Whether to shuffle files and samples
        batch_size (int): Batch size (can stay 1)
        repeat (bool): Repeat dataset indefinitely (for training)

    Returns:
        tf.data.Dataset
    """
    tfrecord_files = tf.io.gfile.glob(
        str(tfrecord_dir) + "/*.tfrecord"
    )

    ds = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=tf.data.AUTOTUNE
    )

    ds = ds.map(
        parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=512)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


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


def get_shap_values(pred,
                    model,
                    img,
                    background_images,
                    explainer_presence=None,
                    type_explainers_cache={},
                    head="tumor_presence",
                    nsamples=100,
                    low_memory=False):
    
    if low_memory:
        nsamples=5
        warnings.warn(f"Reducing nsamples to {nsamples} due to low memory.")

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
            
    return shap_values , type_explainers_cache


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
    img_id="",
    type_explainers_cache={},
    low_memory=False
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
    
    shap_values , type_explainers_cache = get_shap_values(pred,
                                model,
                                img,
                                background_images,
                                explainer_presence=explainer_presence,
                                type_explainers_cache=type_explainers_cache,
                                head=head,
                                nsamples=nsamples,
                                low_memory=low_memory)

    shap_map = compute_shap_map(shap_values, img_size)
    
    import streamlit as st
    if low_memory:
        std = round(shap_map.std(),4)
        if std == 0:
            st.write(f"SHAP can't explane anything due to low samples (SHAP std = {std}")
        else:
            st.write(f"std={std}")

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
    
    name = str(img_id).replace("\n", "_").replace(" ", "_")
    safe_name = title_text.replace("\n", "_").replace(" ", "_")
    name = name + safe_name + ".png"
    plt.savefig(path / name)
    
    return type_explainers_cache


def run_medical_XAI_one_image(img, img_size, model, background_images, explainer_presence, \
                              output_dir, classes, presence_cat=["no_tumor", "tumor"],\
                              type_explainers_cache={}, true_label=None,\
                              img_id="", low_memory=False):
    _ = visualize_explanations(
        model,
        img,
        img_size,
        background_images,
        explainer_presence,
        head="tumor_presence",
        true_label=true_label,
        classes=presence_cat,
        path=output_dir,
        low_memory=low_memory,
        img_id=img_id
    )
    
    type_explainers_cache = visualize_explanations(
        model,
        img,
        img_size,
        background_images,
        head="tumor_type",
        true_label=true_label,
        classes=classes,
        path=output_dir,
        type_explainers_cache=type_explainers_cache,
        low_memory=low_memory,
        img_id=img_id
    )
    
    return type_explainers_cache


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
