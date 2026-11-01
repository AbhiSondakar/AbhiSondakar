"""Object detection using pretrained Faster R-CNN from torchvision."""
import argparse
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Path to input image")
parser.add_argument("--threshold", type=float, default=0.80)
parser.add_argument("--output", default="detected_objects.jpg")
args = parser.parse_args()

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
categories = weights.meta["categories"]
image = Image.open(args.image_path).convert("RGB")
with torch.no_grad():
    prediction = model([to_tensor(image)])[0]

draw = ImageDraw.Draw(image)
count = 0
for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
    if float(score) < args.threshold:
        continue
    x1, y1, x2, y2 = map(float, box.tolist())
    name = categories[int(label)]
    draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    draw.text((x1, max(0, y1 - 15)), f"{name}: {float(score):.2f}", fill="red")
    print(f"{name}: {float(score):.3f}")
    count += 1
image.save(args.output)
print(f"Detected {count} objects. Annotated image saved to {args.output}")
