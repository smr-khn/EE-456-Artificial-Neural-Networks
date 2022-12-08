# YOLOv1 original paper https://arxiv.org/pdf/1506.02640.pdf  
# YOLOv1 in Tensorflow // This one looks good https://www.maskaravivek.com/post/yolov1/  
# YOLOv3 keras https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
#              https://towardsdatascience.com/yolo-v3-object-detection-with-keras-461d2cfccef6 
# Explains YOLO (and loss function) https://hackernoon.com/understanding-yolo-f5a74bbc7967 

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score
#Libraries used: tensorflow, matplotlib, numpy, pandas, graphviz, pydot, pydotplus, scikit learn


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

#This creates the modle.png file, do not need to run again
#tf.keras.utils.plot_model(model,to_file="model.png",show_shapes=True,show_layer_names=True,rankdir="TB",expand_nested=True,dpi=96,)

#compiles layers into a cohesive model
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#creates a stopping criterion if the model does not improve validation loss after x epochs
callbacks = tf.keras.callbacks.EarlyStopping(patience=10)

#trains model for x epochs using training data and also validates it against validation data every epoch
#!!!!comment next line if running pretrained model!!!!
history = model.fit(data_train,label_train, epochs=10,validation_data=(data_test,label_test),callbacks = callbacks)


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

