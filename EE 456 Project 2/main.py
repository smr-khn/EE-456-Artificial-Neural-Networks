import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

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
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
callbacks = [
             EarlyStopping(patience=2)
]

history = model.fit(x_train,y_train, epochs=1,validation_data=(x_test,y_test),callbacks=callbacks)

import pandas as pd
metrics_df = pd.DataFrame(history.history)

metrics_df[["accuracy","val_accuracy"]].plot();

model.save("model.h5")

load_saved_model = tf.keras.models.load_model("model.h5")