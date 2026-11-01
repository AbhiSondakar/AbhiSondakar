"""IMDB sentiment analysis using a SimpleRNN."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE, MAX_WORDS = 5000, 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
print(f"Loaded {len(x_train)} training and {len(x_test)} test reviews")
x_train = pad_sequences(x_train, maxlen=MAX_WORDS)
x_test = pad_sequences(x_test, maxlen=MAX_WORDS)

model = keras.Sequential([
    layers.Embedding(VOCAB_SIZE, 32),
    layers.SimpleRNN(100),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.4f}")
