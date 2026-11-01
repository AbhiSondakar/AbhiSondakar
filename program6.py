"""Time-series forecasting using an LSTM on a synthetic series."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# Replace this synthetic series with your own chronological measurements if available.
t = np.arange(0, 600, dtype=np.float32)
series = (np.sin(t * 0.04) + 0.3 * np.sin(t * 0.13) + 0.001 * t).reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)

INPUT_LEN, HORIZON = 64, 1
def make_windows(values, input_len=INPUT_LEN, horizon=HORIZON):
    X, y = [], []
    for i in range(len(values) - input_len - horizon + 1):
        X.append(values[i:i + input_len])
        y.append(values[i + input_len:i + input_len + horizon, 0])
    return np.asarray(X), np.asarray(y)

X, y = make_windows(scaled)
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

model = keras.Sequential([
    layers.Input(shape=(INPUT_LEN, 1)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.1),
    layers.LSTM(32),
    layers.Dropout(0.1),
    layers.Dense(HORIZON),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="mse", metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {loss:.4f}; scaled MAE: {mae:.4f}")
last_window = scaled[-INPUT_LEN:].reshape(1, INPUT_LEN, 1)
forecast_scaled = model.predict(last_window, verbose=0)
print("Next forecast:", scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).ravel())
