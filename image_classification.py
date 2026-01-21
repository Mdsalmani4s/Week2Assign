import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define dataset transformation (normalize and convert to tensors)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define labels subset (3 classes)
selected_classes = [0, 1, 2]  # Example: Airplane, Automobile, Bird

# Filter dataset
trainset.data = trainset.data[[i for i in range(len(trainset.targets)) if trainset.targets[i] in selected_classes]]
trainset.targets = [trainset.targets[i] for i in range(len(trainset.targets)) if trainset.targets[i] in selected_classes]

testset.data = testset.data[[i for i in range(len(testset.targets)) if testset.targets[i] in selected_classes]]
testset.targets = [testset.targets[i] for i in range(len(testset.targets)) if testset.targets[i] in selected_classes]

# Display a sample image
plt.imshow(trainset.data[0])
plt.title(f"Sample Image - Class {trainset.targets[0]}")
plt.axis("off")
plt.show()



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import numpy as np

# Flatten images for SVM (from 32x32x3 to 1D array)
print("Preparing data...")
X_train = trainset.data.reshape(len(trainset.data), -1)
y_train = np.array(trainset.targets)  # Convert to numpy array

X_test = testset.data.reshape(len(testset.data), -1)
y_test = np.array(testset.targets)  # Convert to numpy array

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Option 1: Train on a subset first to see if it works
print("\n⚠️  Training SVM on full dataset can take 20-60 minutes!")
use_subset = input("Train on smaller subset for faster results? (y/n): ").lower() == 'y'

if use_subset:
    subset_size = 5000
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]
    print(f"\nUsing subset of {subset_size} samples")
else:
    X_train_subset = X_train
    y_train_subset = y_train

# Train SVM classifier
print("\n🚀 Training SVM classifier...")
start_time = time.time()

svm = SVC(kernel='linear', verbose=True)
svm.fit(X_train_subset, y_train_subset)

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")

# Predict and evaluate
print("\n📊 Making predictions...")
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print(f"\n✓ SVM Accuracy: {svm_accuracy:.4f}")