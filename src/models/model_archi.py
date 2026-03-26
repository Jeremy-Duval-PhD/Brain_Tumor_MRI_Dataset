import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121


""" Model """
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


def shared_head_part(inputs, backbone, seed):
    # Data augmentation (training only)
    x = get_model_data_augmentation(inputs, seed)
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


def assemble_heads(img_size, backbone, seed):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    
    x = shared_head_part(inputs, backbone, seed)
    
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


def get_model_built(img_size, models_dir, freeze_backbone, seed):
    
    backbone = get_backbone(img_size, models_dir, freeze_backbone)
    model = assemble_heads(img_size, backbone, seed)

    return model
