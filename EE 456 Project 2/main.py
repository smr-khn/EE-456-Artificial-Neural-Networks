import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#load data set 
(data_train, label_train), (data_test, label_test) = tf.keras.datasets.cifar10.load_data()
#normalize color channels of images
data_train, data_test = data_train / 255.0, data_test / 255.0

#model architecture for RCNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same',input_shape=(32, 32, 3)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3,3), padding='same'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3,3), padding='same'),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(64, (3,3), padding='same'),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(64, (3,3), padding='same'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    #tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation="softmax")
])
#summary of all layers and weights of CNN
model.summary()

#??
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#??
from tensorflow.keras.callbacks import EarlyStopping
callbacks = [
             EarlyStopping(patience=2)
]
#trains model for x epochs using training data and also validates it against validation data every epoch
history = model.fit(data_train,label_train, epochs=5,validation_data=(data_test,label_test),callbacks=callbacks)
#get class probabilities of each training data image
test_predictions = model.predict(data_test)
#chose the index with the highest probability, this is the prediction
test_predictions = np.argmax(test_predictions, axis = 1)

import pandas as pd
#use pandas to create Loss vs Epoch and Accuracy vs Epoch for both training and validation test sets
metrics_df = pd.DataFrame(history.history)
metrics_df[["accuracy","val_accuracy"]].plot()
metrics_df[["loss","val_loss"]].plot()
plt.show()

#graph confusion matrix in terminal
results = tf.math.confusion_matrix(label_test, test_predictions, 10)
print("Confusion Matrix:", results)

#Saving the Model for use later
model.save("model.h5")
load_saved_model = tf.keras.models.load_model("model.h5")
