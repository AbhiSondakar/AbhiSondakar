"""Tiny next-word chatbot demo using a bidirectional LSTM.
The manual's heading mentions image captions, but its code is a chatbot language-model demo.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

training_data = [
    "hello", "hi", "how are you", "i am doing well thanks",
    "what is your name", "i am a chatbot", "tell me a joke",
    "why do scientists trust atoms", "because they make up everything", "exit"
]
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(training_data)
total_words = len(tokenizer.word_index) + 1
sequences = []
for line in training_data:
    ids = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(ids)):
        sequences.append(ids[:i + 1])
if not sequences:
    raise RuntimeError("Not enough text to build training sequences.")
max_len = max(len(s) for s in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")
X, y = sequences[:, :-1], sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 64, input_length=max_len - 1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(total_words, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=100, verbose=0)
id_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

def generate_response(text):
    ids = tokenizer.texts_to_sequences([text])[0]
    if not ids:
        return "I don't know that phrase yet."
    seq = pad_sequences([ids], maxlen=max_len - 1, padding="pre")
    probs = model.predict(seq, verbose=0)[0]
    word_id = int(np.argmax(probs))
    return id_to_word.get(word_id, "I'm still learning.")

if __name__ == "__main__":
    print("Tiny demo chatbot. Type 'exit' to stop.")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() == "exit":
            break
        print("Chatbot:", generate_response(user_text))
