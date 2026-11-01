"""Program 1(a): CIFAR-10 classification using K-Nearest Neighbours."""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Flatten and normalize images. A subset keeps KNN training manageable.
x_train = x_train[:5000].reshape(5000, -1).astype("float32") / 255.0
x_test = x_test[:1000].reshape(1000, -1).astype("float32") / 255.0
y_train = y_train[:5000].ravel()
y_test = y_test[:1000].ravel()

# Train KNN and evaluate
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(f"KNN Test Accuracy (on 1000 samples): {accuracy_score(y_test, y_pred):.4f}")
