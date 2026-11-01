"""Program 1(b): CIFAR-10 classification using a 3-layer neural network."""
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Flatten and normalize images
x_train = x_train.reshape((-1, 32 * 32 * 3)).astype("float32") / 255.0
x_test = x_test.reshape((-1, 32 * 32 * 3)).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Three Dense layers: 512, 256 and 10 output units
model = models.Sequential([
    layers.Input(shape=(32 * 32 * 3,)),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"MLP Test Accuracy: {accuracy:.4f}")
