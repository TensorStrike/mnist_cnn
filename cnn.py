import tensorflow.keras as keras
from keras import layers
import numpy as np

number_classes = 10
input_shape = (28,28,1)

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
print("X-trrain shape is ", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, number_classes)
y_test = keras.utils.to_categorical(y_test, number_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(number_classes, activation="softmax")
    ]
)

model.summary()

batch_size = 64
epoch = 100
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=3)

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, validation_split=0.15, callbacks=[callback])

score = model.evaluate(X_test, y_test, verbose=1)
print("test loss is ", score[0])
print("test acc is ", score[1])