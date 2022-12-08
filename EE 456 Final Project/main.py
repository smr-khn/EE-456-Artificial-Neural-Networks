# YOLOv1 original paper https://arxiv.org/pdf/1506.02640.pdf  
# YOLOv1 in Tensorflow //Reference 2: This one looks good https://www.maskaravivek.com/post/yolov1/  
# YOLOv3 keras https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
#              https://towardsdatascience.com/yolo-v3-object-detection-with-keras-461d2cfccef6 
# Explains YOLO (and loss function) https://hackernoon.com/understanding-yolo-f5a74bbc7967 

import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score
#Libraries used: tensorflow, matplotlib, numpy, pandas, graphviz, pydot, pydotplus, scikit learn

#This is all from reference 2
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


#load data set 
#Classes: 1-airplane, 2-automobile, 3-bird, 4-cat, 5-deer, 6-dog, 7-frog, 8-horse, 9-ship, 10-truck

(data_train, label_train), (data_test, label_test) = tf.keras.datasets.cifar10.load_data()
#normalize color channels of images
data_train, data_test = data_train / 255.0, data_test / 255.0

#model architecture for R-CNN based on AlexNet
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
    tf.keras.layers.YOLO_Reshape(target_shape=(7,7,30))

])
#summary of all layers and weights of CNN
model.summary()


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh


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



mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')


#This creates the modle.png file, do not need to run again
#tf.keras.utils.plot_model(model,to_file="model.png",show_shapes=True,show_layer_names=True,rankdir="TB",expand_nested=True,dpi=96,)

#compiles layers into a cohesive model
model.compile(optimizer='adam',loss=yolo_loss)

#creates a stopping criterion if the model does not improve validation loss after x epochs
callbacks = mcp_save

#trains model for x epochs using training data and also validates it against validation data every epoch
#!!!!comment next line if running pretrained model!!!!
history = model.fit(data_train,label_train, epochs=10,validation_data=,callbacks = callbacks)


#!!!!uncomment next two lines when running pretrained model!!!!
#model = tf.keras.models.load_model("model.h5")
#history = np.load('my_history.npy',allow_pickle='TRUE').item()

#get class probabilities of each training data image
test_predictions = model.predict(data_test)
#chose the index with the highest probability, this is the prediction
test_predictions = np.argmax(test_predictions, axis = 1)

#use pandas to create Loss vs Epoch and Accuracy vs Epoch for both training and validation test sets
#!!!!change to history.history if training model!!!!
metrics_df = pd.DataFrame(history.history)
metrics_df[["accuracy","val_accuracy"]].plot()
metrics_df[["loss","val_loss"]].plot()
plt.show()


#graph confusion matrix in terminal
results = tf.math.confusion_matrix(label_test, test_predictions, 10)
print("Confusion Matrix:", results)

#Use sklearn library to compute th precision and recall of the model
precision, recall, _, _ = score(label_test, test_predictions)
print("Model Precision: ", precision)
print("Model Recall: " , recall)

#Random images to test
summary_images = np.array([data_test[1],data_test[800],data_test[2000],data_test[4000],data_test[6000],data_test[8000],data_test[9000]])
summary_labels = np.array([label_test[1],label_test[800],label_test[2000],label_test[4000],label_test[6000],label_test[8000],label_test[9000]])
#get class probabilities for each class
summary_class_probability = model.predict(summary_images)
#get the max probability of each image form the class probability
summary_prediction_probability = np.max(summary_class_probability, axis = 1)
#get the corresponding index which si the class of highest probability
summary_prediction = np.argmax(summary_class_probability, axis = 1)
print("Class Predictions: ", summary_prediction)
print("Class Probability: ", summary_prediction_probability)

#Saving themodel and history 
model.save("model.h5")
np.save('my_history.npy',history.history)

