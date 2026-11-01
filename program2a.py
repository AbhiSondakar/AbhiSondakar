"""Program 2(a): Pretrained MobileNetV2 image classification."""
import argparse
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Path to the image to classify")
args = parser.parse_args()

model = MobileNetV2(weights="imagenet")
img = image.load_img(args.image_path, target_size=(224, 224))
arr = image.img_to_array(img)
arr = preprocess_input(np.expand_dims(arr, axis=0))
predictions = model.predict(arr, verbose=0)
for rank, (_, label, probability) in enumerate(decode_predictions(predictions, top=3)[0], 1):
    print(f"{rank}. {label}: {probability * 100:.2f}%")
