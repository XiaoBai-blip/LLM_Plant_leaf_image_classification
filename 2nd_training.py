# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wy48Q3Hy8g6dN9eGFDeyF7dv8yqY32vL

# Load the original dataset (20k)
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

import os
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

import os
from collections import defaultdict

# Make sure to define the path to the PlantVillage dataset
plant_village_path = os.path.join(dataset_path, "PlantVillage")  # Update with your actual dataset path

# Initialize a dictionary to count the number of files in each folder
folder_counts = defaultdict(int)

# Iterate through the PlantVillage dataset folders and count files
for root, dirs, files in os.walk(plant_village_path):
    folder_name = os.path.basename(root)  # Get the folder name
    if files:  # Check if there are any files in the folder
        folder_counts[folder_name] += len(files)

# Print the count of items in each folder
print("\nNumber of items in each folder:")
for folder, count in folder_counts.items():
    print(f"Folder: {folder}, Count: {count}")

import os

# Initialize lists to store image paths and labels
image_paths = []
labels = []

# Loop through each category folder in PlantVillage
for category_folder in os.listdir(plant_village_path):
    category_path = os.path.join(plant_village_path, category_folder)
    if os.path.isdir(category_path):  # Ensure it's a directory
        # Get all images in the folder and enumerate them for renaming
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for position, image_file in enumerate(image_files, start=1):
            old_path = os.path.join(category_path, image_file)
            # Rename the image file
            new_filename = f"{category_folder}_{position}.jpg"
            new_path = os.path.join(category_path, new_filename)
            os.rename(old_path, new_path)

            # Append to the lists for later use
            image_paths.append(new_path)
            labels.append(category_folder)

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

from sklearn.model_selection import train_test_split

# Split the dataset into 80% train, 20% test
original_train_paths, original_test_paths, original_train_labels, original_test_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"Training set: {len(original_train_paths)} images")
print(f"Testing set: {len(original_test_paths)} images")

# Extract unique identifiers from the file paths of the test set
original_test_identifiers = set([os.path.basename(path).rsplit('.', 1)[0] for path in original_test_paths])

print(f"Number of unique image identifiers for original test set: {len(original_test_identifiers)}")

# Print the first ten unique identifiers from the original test set
print("Sample unique identifiers from the original test set:")
for identifier in list(original_test_identifiers)[:10]:
    print(identifier)











"""# Load the SAM dataset (20k) and match the 20% test set with the 20% of the original dataset"""

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Set the path to the folder containing segmented images
segmented_images_path = "/content/drive/My Drive/ViT-plant/seg_images"

import os

# Define the label mapping again
label_map = {
    "segmented_Tomato_Early_blight": "Tomato Early Blight",
    "segmented_Pepper__bell___Bacterial_spot": "Pepper Bell Bacterial Spot",
    "segmented_Pepper__bell___healthy": "Pepper Bell Healthy",
    "segmented_Potato___Early_blight": "Potato Early Blight",
    "segmented_Potato___healthy": "Potato Healthy",
    "segmented_Potato___Late_blight": "Potato Late Blight",
    "segmented_Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "segmented_Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "segmented_Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "segmented_Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Spider Mites",
    "segmented_Tomato__Target_Spot": "Tomato Target Spot",
    "segmented_Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "segmented_Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "segmented_Tomato_healthy": "Tomato Healthy",
    "segmented_Tomato_Late_blight": "Tomato Late Blight"
}

# Path to the folder where all images are combined into a single folder
segmented_images_path = "/content/drive/My Drive/ViT-plant/seg_images"  # Update with your actual path

# Prepare the dataset
segmented_image_paths = []
segmented_labels = []

# Iterate through all images in the folder
for image_file in os.listdir(segmented_images_path):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check valid image file extensions
        image_path = os.path.join(segmented_images_path, image_file)

        # Extract the category key (remove last underscore and number)
        category_key = '_'.join(image_file.split('_')[:-1])  # Extract everything before the last underscore
        label = label_map.get(category_key, "Unknown")  # Map to human-readable label using label_map

        # Log error if category is not in label_map
        if label == "Unknown":
            print(f"Warning: Category '{category_key}' not found in label_map for image '{image_file}'")

        segmented_image_paths.append(image_path)
        segmented_labels.append(label)

# Verify the dataset
print(f"Collected {len(segmented_image_paths)} segmented images across {len(set(segmented_labels))} categories.")
print("Sample labels and paths:")
for i in range(5):
    print(f"Label: {segmented_labels[i]}, Path: {segmented_image_paths[i]}")

from collections import Counter

# Count the number of images for each label
label_counts = Counter(segmented_labels)

# Print the counts for each label
print("\nNumber of images in each label:")
for label, count in label_counts.items():
    print(f"Label: {label}, Count: {count}")

# Calculate the total number of items across all labels
total_items = sum(Counter(segmented_labels).values())

# Print the total number of items
print(f"\nTotal number of items across all labels: {total_items}")

"""### match to use the same testing dataset."""

# 🟢 Step 3: Filter the SAM-segmented dataset to create the SAM test set
sam_test_paths = []
sam_test_labels = []

for path, label in zip(segmented_image_paths, segmented_labels):  # `segmented_image_paths` contains SAM dataset paths
    sam_identifier = os.path.basename(path).rsplit('.', 1)[0].replace('segmented_', '')
  # Extract the identifier
    if sam_identifier in original_test_identifiers:  # Check if the identifier exists in the original test set
        sam_test_paths.append(path)
        sam_test_labels.append(label)

print(f"Number of images in the SAM test set: {len(sam_test_paths)}")

# 🟢 Step 4: Create the SAM training set (exclude the SAM test set)
sam_train_paths = [path for path in segmented_image_paths if path not in sam_test_paths]
sam_train_labels = [label for path, label in zip(segmented_image_paths, segmented_labels) if path not in sam_test_paths]

print(f"Number of images in the SAM training set: {len(sam_train_paths)}")

# 🟢 Step 5: Verify the SAM test dataset by listing the first 10 samples
print("\nSample SAM test set labels and paths (first 10):")
for i in range(min(10, len(sam_test_paths))):  # Ensure we don't exceed the dataset size
    print(f"{i+1}. Label: {sam_test_labels[i]}, Path: {sam_test_paths[i]}")

import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
# 🟢 Step 4: Define Transformations
sam_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
    transforms.RandomRotation(degrees=30),  # Random rotation from -30 to 30 degrees
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Random Gaussian blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 🟢 Step 5: Define the Custom Dataset
class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = categories.index(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label

# 🟢 Step 6: Create DataLoaders
train_dataset_sam = PlantDataset(sam_train_paths, sam_train_labels, transform=sam_train_transform)
test_dataset_sam = PlantDataset(sam_test_paths, sam_test_labels, transform=sam_train_transform)

train_loader_sam = DataLoader(train_dataset_sam, batch_size=64, shuffle=True, num_workers=4)
test_loader_sam = DataLoader(test_dataset_sam, batch_size=64, shuffle=False, num_workers=4)

print(f"SAM Train Set: {len(train_loader_sam.dataset)} images")
print(f"SAM Test Set: {len(test_loader_sam.dataset)} images")

"""# Load the ViT we fine-turned last time (used original 20k) and continue training on new SAM data + flip, rotate, blur"""

from google.colab import drive
from transformers import ViTForImageClassification, ViTImageProcessor
import torch

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the saved model on Google Drive
model_path = '/content/drive/MyDrive/ViT-plant/fine_tuned_vit'

# Load the processor and model
processor = ViTImageProcessor.from_pretrained(model_path)
Vit_finetuned_model = ViTForImageClassification.from_pretrained(model_path)

print("Model and processor loaded successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Vit_finetuned_model.to(device)

from transformers import ViTForImageClassification
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from tqdm import tqdm

categories = list(label_map.values())

# ------------------------------------------
# 🟢 Step 2: Define Loss, Optimizer, and Scheduler
# ------------------------------------------
# Define optimizer, loss function, and scheduler
optimizer = AdamW(Vit_finetuned_model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()
num_training_steps = len(train_loader_sam) * 5
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)



# Training loop
num_epochs = 5
Vit_finetuned_model.train()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader_sam, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = Vit_finetuned_model(images)
        loss = criterion(outputs.logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    print(f"Epoch {epoch + 1} completed. Loss: {running_loss / len(train_loader_sam):.4f}, Accuracy: {100 * correct / total:.2f}%")

"""This result is for: 原先的模型，用原先traing dataset对应的SAMed training，做了一轮training,

"""

import shutil

# 🟢 Step 1: Mount Google Drive
drive.mount('/content/drive')

# 🟢 Step 2: Create a directory to save the fine-tuned model
save_directory = "./SAM_fine_tuned_vit"
os.makedirs(save_directory, exist_ok=True)

# 🟢 Step 3: Save the model weights and configuration
# This saves the model's weights and configuration so it can be loaded later
Vit_finetuned_model.save_pretrained(save_directory)

# 🟢 Step 4: Copy the saved model directory to Google Drive
drive_save_path = '/content/drive/MyDrive/ViT-plant/SAM_fine_tuned_vit'  # Update this path as needed
shutil.copytree(save_directory, drive_save_path)

print(f"✅ Fine-tuned ViT model saved to Google Drive at: {drive_save_path}")

import time

categories = list(label_map.values())
# 🟢 Test Accuracy Calculation
Vit_finetuned_model.eval()  # Turn off dropout, batch norm, etc.
correct = 0  # Number of correct predictions
total = 0  # Total number of samples

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in test_loader_sam:  # Loop through the test DataLoader
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Forward pass through the model
        outputs = Vit_finetuned_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate accuracy as a percentage
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


# 🟢 Inference Time Calculation
Vit_finetuned_model.eval()
inference_times = []

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Calculating Inference Time"):
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Measure inference time
        start_time = time.time()
        outputs = Vit_finetuned_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)
        inference_time = time.time() - start_time
        inference_times.append(inference_time / images.size(0))  # Time per image

# Calculate average inference time
avg_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {avg_inference_time * 1000:.2f} ms/image")


# 🟢 Robustness Test Calculation
Vit_finetuned_model.eval()
correct = 0  # Number of correct predictions
total = 0  # Total number of samples
noise_level = 0.1

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Testing Robustness"):
        # Add random noise to the images
        noise = torch.randn_like(images) * noise_level  # Create random noise
        noisy_images = torch.clamp(images + noise, 0, 1)  # Clamp pixel values to [0, 1]

        # Move images and labels to the GPU (if available)
        noisy_images, labels = noisy_images.to(device), labels.to(device)

        # Forward pass with noisy images
        outputs = Vit_finetuned_model(noisy_images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate robustness accuracy as a percentage
robustness_accuracy = 100 * correct / total
print(f"Robustness Test Accuracy with {noise_level * 100}% noise: {robustness_accuracy:.2f}%")

"""这个是新模型在test daatset上的accuracy."""

from transformers import AutoModelForImageClassification

# 🟢 Step 2: Path to the saved model directory
drive_save_path = '/content/drive/MyDrive/ViT-plant/SAM_fine_tuned_vit'

# 🟢 Step 3: Load the fine-tuned model from the saved directory
SAM_Vit_finetuned_model = AutoModelForImageClassification.from_pretrained(drive_save_path)

print(f"✅ Successfully loaded the fine-tuned ViT model from: {drive_save_path}")

# 🟢 (Optional) Print model architecture
print(SAM_Vit_finetuned_model)

import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
from tqdm import tqdm

categories = list(label_map.values())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAM_Vit_finetuned_model.to(device)
# 🟢 Ensure model is in evaluation mode
SAM_Vit_finetuned_model.eval()

# 🟢 Lists to store true and predicted labels for the entire test set
all_labels = []
all_predictions = []
all_probabilities = []

# 🟢 Disable gradient computation for faster inference
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Evaluating Model"):
        # Move data to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = SAM_Vit_finetuned_model(images)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probabilities = torch.nn.functional.softmax(logits, dim=1)  # Convert logits to probabilities
        _, predicted = torch.max(logits, 1)

        # Store true labels, predictions, and probabilities
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# Convert lists to NumPy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)

# 🟢 Calculate Metrics
precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')
accuracy = accuracy_score(all_labels, all_predictions)

print(f"Precision (Macro): {precision:.4f}")
print(f"Recall (Macro): {recall:.4f}")
print(f"F1-Score (Macro): {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# 🟢 Top-K Accuracy (Top-1, Top-3, and Top-5)
top_k_accuracies = {k: 0 for k in [1, 3, 5]}
total_samples = len(all_labels)
for i in range(total_samples):
    top_k_preds = np.argsort(all_probabilities[i])[-5:]  # Get the indices of the top 5 predictions
    if all_labels[i] in top_k_preds[-1:]:  # Top-1
        top_k_accuracies[1] += 1
    if all_labels[i] in top_k_preds[-3:]:  # Top-3
        top_k_accuracies[3] += 1
    if all_labels[i] in top_k_preds[-5:]:  # Top-5
        top_k_accuracies[5] += 1

top_1_accuracy = top_k_accuracies[1] / total_samples * 100
top_3_accuracy = top_k_accuracies[3] / total_samples * 100
top_5_accuracy = top_k_accuracies[5] / total_samples * 100

print(f"Top-1 Accuracy: {top_1_accuracy:.2f}%")
print(f"Top-3 Accuracy: {top_3_accuracy:.2f}%")
print(f"Top-5 Accuracy: {top_5_accuracy:.2f}%")

# 🟢 ROC-AUC Curve (for multi-class classification)
roc_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')


print(f"ROC-AUC (OVR): {roc_auc:.4f}")

# 🟢 Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 🟢 Visualization of Key Metrics (Bar Chart)
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'ROC-AUC']
metric_values = [precision, recall, f1, accuracy, top_1_accuracy / 100, top_3_accuracy / 100, top_5_accuracy / 100, roc_auc]

plt.figure(figsize=(10, 6))
plt.bar(metrics, metric_values, color='lightseagreen')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Score')
plt.title('Evaluation Metrics')
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Assuming `all_labels` contains the true labels and `all_probabilities` contains probabilities for each class.
# Example:
# all_labels = [0, 1, 2, ...]  # Ground truth for each sample
# all_probabilities = [[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], ...]  # Model probabilities for each class

# Number of classes in the dataset
num_classes = all_probabilities.shape[1]

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for class_idx in range(num_classes):
    # Get true binary labels for the current class (One-vs-Rest approach)
    binary_labels = (all_labels == class_idx).astype(int)
    # Get probabilities for the current class
    class_probabilities = all_probabilities[:, class_idx]
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(binary_labels, class_probabilities)
    # Compute the AUC (Area Under the Curve)
    class_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'Class {class_idx} (AUC = {class_auc:.2f})')

# Plot diagonal line for random guessing
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess')

# Add plot details
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves for Multi-Class Classification')
plt.legend(loc='lower right')
plt.grid()
plt.show()













"""# Load fine-tuned ResNet and continue training on SAM 20k"""

# 🟢 Step 1: Mount Google Drive
drive.mount('/content/drive')

# 🟢 Step 2: Define the path to the saved models on Google Drive
full_model_path = '/content/drive/MyDrive/ViT-plant/fine_tuned_resnet/resnet50_fine_tuned_full.pth'


# -------------------------------------------------------
# 🔥 Option 1: Load the **Full Saved Model** (weights + config)
# -------------------------------------------------------
print("\n🔹 Loading Full ResNet Model...")
full_resnet_model = torch.load(full_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_resnet_model.to(device)
full_resnet_model.eval()  # Set to evaluation mode

print("✅ Full ResNet model loaded successfully from:", full_model_path)

"""## 用20k original trained过的RESNET zeroshot on SAM test set"""

import time

categories = list(label_map.values())
# 🟢 Test Accuracy Calculation
full_resnet_model.eval()  # Turn off dropout, batch norm, etc.
correct = 0  # Number of correct predictions
total = 0  # Total number of samples

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in test_loader_sam:  # Loop through the test DataLoader
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Forward pass through the model
        outputs = full_resnet_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate accuracy as a percentage
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


# 🟢 Inference Time Calculation
full_resnet_model.eval()
inference_times = []

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Calculating Inference Time"):
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Measure inference time
        start_time = time.time()
        outputs = full_resnet_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)
        inference_time = time.time() - start_time
        inference_times.append(inference_time / images.size(0))  # Time per image

# Calculate average inference time
avg_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {avg_inference_time * 1000:.2f} ms/image")


# 🟢 Robustness Test Calculation
full_resnet_model.eval()
correct = 0  # Number of correct predictions
total = 0  # Total number of samples
noise_level = 0.1

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Testing Robustness"):
        # Add random noise to the images
        noise = torch.randn_like(images) * noise_level  # Create random noise
        noisy_images = torch.clamp(images + noise, 0, 1)  # Clamp pixel values to [0, 1]

        # Move images and labels to the GPU (if available)
        noisy_images, labels = noisy_images.to(device), labels.to(device)

        # Forward pass with noisy images
        outputs = full_resnet_model(noisy_images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate robustness accuracy as a percentage
robustness_accuracy = 100 * correct / total
print(f"Robustness Test Accuracy with {noise_level * 100}% noise: {robustness_accuracy:.2f}%")

"""## Continue training。。。"""

import torch.optim as optim
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from tqdm import tqdm



# 🟢 Step 2: Define Loss, Optimizer, and Scheduler
# ------------------------------------------
# Define optimizer, loss function, and scheduler
optimizer = optim.AdamW(full_resnet_model.parameters(), lr=5e-5)  # Use AdamW optimizer (same as ViT)
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
num_training_steps = len(train_loader_sam) * 5  # Assuming 5 epochs
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)

# ------------------------------------------
# 🟢 Step 3: Training Loop
# ------------------------------------------
num_epochs = 8  # Number of epochs
full_resnet_model.train()  # Set the model to training mode

for epoch in range(num_epochs):
    print(f"🔹 Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader_sam, leave=True)  # Training progress bar
    for images, labels in loop:
        # Move images and labels to GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = full_resnet_model(images)  # Get predictions from the ResNet model
        loss = criterion(outputs, labels)  # Calculate the loss

        # Backward pass
        optimizer.zero_grad()  # Zero out previous gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model weights
        scheduler.step()  # Update learning rate scheduler

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted class for each image
        total += labels.size(0)  # Count total images in the batch
        correct += (predicted == labels).sum().item()  # Count correct predictions

        # Update the progress bar with loss and accuracy
        loop.set_description(f"🔹 Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    # Print epoch summary
    print(f"🔹 Epoch {epoch + 1} completed. Loss: {running_loss / len(train_loader_sam):.4f}, Accuracy: {100 * correct / total:.2f}%")


# ------------------------------------------
# 🟢 Step 4: Save the Fine-tuned ResNet Model
# ------------------------------------------
# Save the fine-tuned model as a new version (both weights and full model)
fine_tuned_resnet_path = '/content/drive/MyDrive/ViT-plant/SAM_fine_tuned_resnet'

# Save the full model (including architecture + weights)
torch.save(full_resnet_model, fine_tuned_resnet_path)

print(f"✅ Fine-tuned ResNet model saved successfully to: {fine_tuned_resnet_path}")

import time

categories = list(label_map.values())
# 🟢 Test Accuracy Calculation
full_resnet_model.eval()  # Turn off dropout, batch norm, etc.
correct = 0  # Number of correct predictions
total = 0  # Total number of samples

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in test_loader_sam:  # Loop through the test DataLoader
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Forward pass through the model
        outputs = full_resnet_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate accuracy as a percentage
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


# 🟢 Inference Time Calculation
full_resnet_model.eval()
inference_times = []

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Calculating Inference Time"):
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Measure inference time
        start_time = time.time()
        outputs = full_resnet_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)
        inference_time = time.time() - start_time
        inference_times.append(inference_time / images.size(0))  # Time per image

# Calculate average inference time
avg_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {avg_inference_time * 1000:.2f} ms/image")


# 🟢 Robustness Test Calculation
full_resnet_model.eval()
correct = 0  # Number of correct predictions
total = 0  # Total number of samples
noise_level = 0.1

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Testing Robustness"):
        # Add random noise to the images
        noise = torch.randn_like(images) * noise_level  # Create random noise
        noisy_images = torch.clamp(images + noise, 0, 1)  # Clamp pixel values to [0, 1]

        # Move images and labels to the GPU (if available)
        noisy_images, labels = noisy_images.to(device), labels.to(device)

        # Forward pass with noisy images
        outputs = full_resnet_model(noisy_images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate robustness accuracy as a percentage
robustness_accuracy = 100 * correct / total
print(f"Robustness Test Accuracy with {noise_level * 100}% noise: {robustness_accuracy:.2f}%")



