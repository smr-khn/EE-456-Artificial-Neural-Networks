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
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
import fiftyone


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score

BATCH_SIZE = 16
#EPOCHS = int(os.getpipenv("EPOCHS", "1"))
#CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint/")
#INFERENCE_CHECKPOINT_PATH = os.getenv("INFERENCE_CHECKPOINT_PATH", CHECKPOINT_PATH)

#https://github.com/keras-team/keras-cv/blob/master/keras_cv/datasets/pascal_voc/load.py
train_ds, train_ds_info = load.load(split="train", bounding_box_format="xywh", batch_size=4)
val_ds, val_ds_info = load.load(split="validation", bounding_box_format="xywh", batch_size=4)

#https://github.com/keras-team/keras-io/blob/master/guides/keras_cv/retina_net_overview.py
class_ids = ["Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car","Cat","Chair","Cow","Dining Table","Dog","Horse",
             "Motorcycle","Person","Potted Plant","Sheep","Sofa","Train","TV Monitor","Total",]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# def visualize_dataset(dataset, bounding_box_format):
#     example = next(iter(dataset))
#     images, boxes = example["images"], example["bounding_boxes"]
#     visualization.plot_bounding_box_gallery(
#         images,
#         value_range=(0, 255),
#         bounding_box_format=bounding_box_format,
#         y_true=boxes,
#         scale=4,
#         rows=3,
#         cols=3,
#         show=True,
#         thickness=4,
#         font_scale=1,
#         class_mapping=class_mapping,
#     )

# visualize_dataset(train_ds, bounding_box_format="xywh")

class Yolo_Reshape(tf.keras.layers.Layer):
  def __init__(self, target_shape):
    super(Yolo_Reshape, self).__init__()
    self.target_shape = tuple(target_shape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'target_shape': self.target_shape
    })
    return config

  def call(self, input):
    # grids 7x7
    S = [self.target_shape[0], self.target_shape[1]]
    # classes
    C = 20
    # no of bounding boxes per grid
    B = 2

    idx1 = S[0] * S[1] * C
    idx2 = idx1 + S[0] * S[1] * B
   
    # class probabilities
    class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))
    class_probs = K.softmax(class_probs)

    #confidence
    confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
    confs = K.sigmoid(confs)

    # boxes
    boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
    boxes = K.sigmoid(boxes)

    outputs = K.concatenate([class_probs, confs, boxes])
    return outputs

def yolo_loss(y_true, y_pred):
    label_class = y_true[..., :20]  # ? * 7 * 7 * 20
    label_box = y_true[..., 20:24]  # ? * 7 * 7 * 4
    response_mask = y_true[..., 24]  # ? * 7 * 7
    response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
    predict_trust = y_pred[..., 20:22]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
    box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss


nb_boxes=1
grid_w=7
grid_h=7
cell_w=64
cell_h=64
img_w=grid_w*cell_w
img_h=grid_h*cell_h
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size= (7, 7), strides=(1, 1), input_shape =(img_h, img_w, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'),

    tf.keras.layers.Conv2D(filters=192, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'),

    tf.keras.layers.Conv2D(filters=128, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=256, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'),

    tf.keras.layers.Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'),

    tf.keras.layers.Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=1024, kernel_size= (3, 3), strides=(2, 2), padding = 'same'),

    tf.keras.layers.Conv2D(filters=1024, kernel_size= (3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)),
    tf.keras.layers.Conv2D(filters=1024, kernel_size= (3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1470, activation='sigmoid'),
    Yolo_Reshape(target_shape=(7,7,30))

])
#summary of all layers and weights of CNN
model.summary()

mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')


#This creates the modle.png file, do not need to run again
#tf.keras.utils.plot_model(model,to_file="model.png",show_shapes=True,show_layer_names=True,rankdir="TB",expand_nested=True,dpi=96,)

#compiles layers into a cohesive model
model.compile(optimizer='adam',loss=yolo_loss)

#creates a stopping criterion if the model does not improve validation loss after x epochs
callbacks = mcp_save



class My_Custom_Generator(keras.utils.Sequence) :
 
  def __init__(self, train, batch_size) :
    self.train = train
    self.batch_size = batch_size
   
   
  def __len__(self) :
    example = next(iter1(dataset))
   
    return (np.ceil(len(example))).astype(np.int)
 
 
  def __getitem__(self, idx) :
    thisOne =next(iter(self.train))
    images = thisOne["images"]
    labels = thisOne["bounding_boxes"]
    #batch_x = images[idx * self.batch_size : (idx+1) * self.batch_size]
    #batch_y = labels[idx * self.batch_size : (idx+1) * self.batch_size]

    batch_x=images
    batch_y=labels

    train_image = []
    train_label = []

    for i in range(0, len(batch_x)):
        img_path = batch_x[i]
        label = batch_y[:,i]
     
        image_w =448
        image_h = 448
        label_matrix = np.zeros([7, 7, 30])
        #for l in label:
            # l = l.split(',')
        #l = np.array(l, dtype=np.int)
        l = label
        xGet = l[0]
        yGet = l[1]
        wGet = l[2]
        hGet = l[3]
        cls = l[4]
        x = (xGet) / 2 / image_w
        y = (yGet) / 2 / image_h
        w = (wGet) / image_w
        h = (hGet) / image_h
        loc = [7 * x, 7 * y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        y = loc[1] - loc_i
        x = loc[0] - loc_j

        if label_matrix[loc_i, loc_j, 24] == 0:
            label_matrix[loc_i, loc_j, cls] = 1
            label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
            label_matrix[loc_i, loc_j, 24] = 1  # response


        train_image.append(images[i])
        train_label.append(label_matrix)
    return np.array(train_image), np.array(train_label)


batch_size = 4

my_training_batch_generator = My_Custom_Generator(train_ds, batch_size)

history = model.fit(x=my_training_batch_generator, epochs=10,callbacks = callbacks)
