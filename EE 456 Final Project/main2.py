

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
EPOCHS = int(os.getenv("EPOCHS", "1"))
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint/")
INFERENCE_CHECKPOINT_PATH = os.getenv("INFERENCE_CHECKPOINT_PATH", CHECKPOINT_PATH)

#https://github.com/keras-team/keras-cv/blob/master/keras_cv/datasets/pascal_voc/load.py
train_ds, train_ds_info = load.load(split="train", bounding_box_format="xywh", batch_size=BATCH_SIZE)
val_ds, val_ds_info = load.load(split="validation", bounding_box_format="xywh", batch_size=BATCH_SIZE)

#https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/retina_net_overview.py
class_ids = ["Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car","Cat","Chair","Cow","Dining Table","Dog","Horse",
             "Motorcycle","Person","Potted Plant","Sheep","Sofa","Train","TV Monitor","Total",]
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

visualize_dataset(train_ds, bounding_box_format="xywh")