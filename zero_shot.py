# -*- coding: utf-8 -*-
"""zero-shot.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1I01RZ4nGqiZ6zm82uuJwrPLdNHu5Vl9h
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("emmarex/plantdisease")

print("Path to dataset files:", path)

import os

# Replace 'path' with the actual variable where your dataset path is stored
dataset_path = path  # This is the path variable from your kagglehub download

# List top-level files and folders
for root, dirs, files in os.walk(dataset_path):
    print(f"Directory: {root}")
    for name in dirs:
        print(f"Folder: {name}")
    for name in files:
        print(f"File: {name}")
    # Break after the first directory level to avoid too much output
    break

# Path to the PlantVillage folder
plant_village_path = os.path.join(dataset_path, "PlantVillage")

# List all folders inside PlantVillage
for root, dirs, files in os.walk(plant_village_path):
    print(f"Directory: {root}")
    for name in dirs:
        print(f"Folder: {name}")
    # Print a sample of files in each folder
    for name in files[:5]:  # Show first 5 files as a sample
        print(f"File: {name}")
    # Break after the first directory level to avoid too much output
    break

# Initialize lists to store image paths and labels
image_paths = []
labels = []

# Loop through each category folder in PlantVillage
for category_folder in os.listdir(plant_village_path):
    category_path = os.path.join(plant_village_path, category_folder)
    if os.path.isdir(category_path):  # Ensure it's a directory
        for image_file in os.listdir(category_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Image file extensions
                image_paths.append(os.path.join(category_path, image_file))
                labels.append(category_folder)  # Folder name as label (y)

# Verify the data collection
print(f"Collected {len(image_paths)} images across {len(set(labels))} categories.")
print("Sample labels and paths:")
for i in range(10):  # Show 10 samples for verification
    print(f"Label: {labels[i]}, Path: {image_paths[i]}")

# Mapping from original folder names to better, more readable category names
label_map = {
    "Tomato_Early_blight": "Tomato Early Blight",
    "Pepper__bell___Bacterial_spot": "Pepper Bell Bacterial Spot",
    "Pepper__bell___healthy": "Pepper Bell Healthy",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___healthy": "Potato Healthy",
    "Potato___Late_blight": "Potato Late Blight",
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Spider Mites",
    "Tomato__Target_Spot": "Tomato Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato_healthy": "Tomato Healthy",  # Fixed mapping
    "Tomato_Late_blight": "Tomato Late Blight"
}

# Initialize lists to store image paths and mapped labels
image_paths = []
labels = []

# Loop through each category folder in PlantVillage and apply label mapping
for category_folder in os.listdir(plant_village_path):
    category_path = os.path.join(plant_village_path, category_folder)
    if os.path.isdir(category_path):  # Ensure it's a directory
        # Use mapped label or default to original folder name if not in label_map
        label = label_map.get(category_folder, category_folder)
        for image_file in os.listdir(category_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Image file extensions
                image_paths.append(os.path.join(category_path, image_file))
                labels.append(label)  # Mapped label as y

# Verify the data collection with mapped labels
print(f"Collected {len(image_paths)} images across {len(set(labels))} categories.")
print("Sample labels and paths:")
for i in range(10):  # Show 10 samples for verification
    print(f"Label: {labels[i]}, Path: {image_paths[i]}")

"""## Zero-Shot Classification with ViT"""

from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load pre-trained ViT model and image processor
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Select 300 images for zero-shot classification
zero_shot_image_paths = image_paths[:300]
zero_shot_labels = labels[:300]

# Perform zero-shot classification
correct = 0
total = 0

for idx, (image_path, true_label) in enumerate(zip(zero_shot_image_paths, zero_shot_labels)):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_index = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_index]

    # Check if prediction is correct
    if predicted_label == true_label:
        correct += 1
    total += 1

    # Print progress every 50 images
    if idx % 50 == 0:
        print(f"Processed {idx + 1}/{len(zero_shot_image_paths)} images.")

# Calculate accuracy
zero_shot_accuracy = (correct / total) * 100
print(f"Zero-Shot Classification Accuracy on 300 images: {zero_shot_accuracy:.2f}%")

"""## Zero-Shot Classification with ResNet"""

from torchvision import models, transforms
from PIL import Image
import torch

# Step 1: Load the pre-trained ResNet model (ResNet50 in this case)
model = models.resnet50(pretrained=True)  # You can use resnet18, resnet101, etc.
model.eval()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Define image transformation (must match ImageNet preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizes the image to 224x224 pixels
    transforms.ToTensor(),  # Converts image to tensor format (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
])

# Step 3: Select 300 images for zero-shot classification
zero_shot_image_paths = image_paths[:300]  # List of image file paths
zero_shot_labels = labels[:300]  # List of ground truth labels (make sure these match ImageNet labels, or map them)

# Step 4: Load ImageNet class labels (if available)
# This will map the ResNet model's output indices (0-999) to class names
import json
import urllib.request

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
imagenet_classes = json.loads(urllib.request.urlopen(url).read().decode())

# Step 5: Zero-shot classification loop
correct = 0
total = 0

for idx, (image_path, true_label) in enumerate(zip(zero_shot_image_paths, zero_shot_labels)):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Get predictions from the model
    with torch.no_grad():
        outputs = model(inputs)
        predicted_index = outputs.argmax(-1).item()  # Get the index of the class with the highest logit
        predicted_label = imagenet_classes[predicted_index]  # Map index to human-readable label

    # Check if prediction is correct (Note: ImageNet classes may not match your custom labels)
    if predicted_label == true_label:
        correct += 1
    total += 1

    # Print progress every 50 images
    if idx % 50 == 0:
        print(f"Processed {idx + 1}/{len(zero_shot_image_paths)} images. Current Accuracy: {(correct / total) * 100:.2f}%")

# Step 6: Calculate and display the final zero-shot accuracy
zero_shot_accuracy = (correct / total) * 100
print(f"Zero-Shot Classification Accuracy on 300 images: {zero_shot_accuracy:.2f}%")