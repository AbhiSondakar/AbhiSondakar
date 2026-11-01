"""CNN image-classification demonstration based on the manual's Program 4 code.
Note: the source manual labels this image captioning, but its code trains a CIFAR-10
classifier and does not generate natural-language captions.
"""
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (4, 4), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"CIFAR-10 test accuracy: {accuracy * 100:.2f}%")
