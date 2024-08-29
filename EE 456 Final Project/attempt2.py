

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers
import load

import keras_cv
from keras_cv import bounding_box
import os
from luketils import visualization
from keras_cv import layers as cv_layers


BATCH_SIZE = 16
EPOCHS = int(os.getenv("EPOCHS", "2"))
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint/")
INFERENCE_CHECKPOINT_PATH = os.getenv("INFERENCE_CHECKPOINT_PATH", CHECKPOINT_PATH)

#https://github.com/keras-team/keras-cv/blob/master/keras_cv/datasets/pascal_voc/load.py
train_ds, train_ds_info = load.load(split="train", bounding_box_format="xywh", batch_size=BATCH_SIZE)
val_ds, val_ds_info = load.load(split="validation", bounding_box_format="xywh", batch_size=BATCH_SIZE)

#https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/retina_net_overview.py
class_ids = ["Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car","Cat","Chair","Cow","Dining Table","Dog","Horse",
             "Motorcycle","Human","Potted Plant","Sheep","Sofa","Train","TV Monitor","Total",]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

def visualize_dataset(dataset, bounding_box_format):
    example = next(iter(dataset))
    images, boxes = example["images"], example["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=boxes,
        scale=4,
        rows=3,
        cols=3,
        show=True,
        thickness=4,
        font_scale=1,
        class_mapping=class_mapping,
    )

#visualize_dataset(train_ds, bounding_box_format="xywh")

#augment data to make more images by flipping and r
random_flip = keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh")
rand_augment = keras_cv.layers.RandAugment(value_range=(0, 255),augmentations_per_image=2,geometric=False)
def augment(inputs):
    inputs["images"] = rand_augment(inputs["images"])
    inputs = random_flip(inputs)
    return inputs

train_ds = train_ds.map(augment,num_parallel_calls=tf.data.AUTOTUNE)
#batch train set, cannot run augmentation with batched dataset
#visualize_dataset(train_ds, bounding_box_format="xywh")

#unpack inputs into tuples
def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]

train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

#create model
model = keras_cv.models.RetinaNet(
    classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
    evaluate_train_time_metrics=False,
)
model.backbone.trainable = False

#compile model
optimizer = tf.optimizers.SGD(global_clipnorm=10.0)
model.compile(
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    optimizer=optimizer,
    metrics=[
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=range(20),
            bounding_box_format="xywh",
            name="Mean Average Precision",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=range(20),
            bounding_box_format="xywh",
            max_detections=100,
            name="Recall",
        ),
    ],
)

callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.ReduceLROnPlateau(patience=5),
    # Uncomment to train your own RetinaNet
    keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True),
]

model.fit(
    train_ds,
    validation_data=val_ds.take(20),
    epochs=EPOCHS,
    callbacks=callbacks,
)