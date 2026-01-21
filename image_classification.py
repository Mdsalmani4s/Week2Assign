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



########32. Softmax Classifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

print("\n" + "="*50)
print("SOFTMAX CLASSIFIER")
print("="*50)

# Train Softmax classifier
print("\n🚀 Training Softmax classifier...")
start_time = time.time()

softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, verbose=1)
softmax.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds")

# Predict and evaluate
print("\n📊 Making predictions...")
y_pred_softmax = softmax.predict(X_test)
softmax_accuracy = accuracy_score(y_test, y_pred_softmax)

print(f"\n✓ Softmax Accuracy: {softmax_accuracy:.4f}")




#########3. Two-layer Neural Network

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

print("\n" + "="*50)
print("TWO-LAYER NEURAL NETWORK")
print("="*50)

# Define neural network model
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define parameters
input_size = 32 * 32 * 3
hidden_size = 100
output_size = len(selected_classes)

print(f"\nModel Architecture:")
print(f"  Input size: {input_size}")
print(f"  Hidden size: {hidden_size}")
print(f"  Output size: {output_size}")

# Initialize model
model = TwoLayerNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Normalize data for better neural network performance
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0

# Convert data to tensors
X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create data loader for batch training
batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train neural network
print("\n🚀 Training Neural Network...")
num_epochs = 100
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")

# Evaluate on test set
print("\n📊 Evaluating Neural Network...")
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    nn_accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"✓ Neural Network Accuracy: {nn_accuracy:.4f}")



#######📊 Step 5: Compare Classifier Performance

import matplotlib.pyplot as plt

print("\n" + "="*50)
print("CLASSIFIER PERFORMANCE COMPARISON")
print("="*50)

# Display all accuracies
print(f"\n📊 Final Results:")
print(f"  SVM Accuracy:            {svm_accuracy:.4f}")
print(f"  Softmax Accuracy:        {softmax_accuracy:.4f}")
print(f"  Neural Network Accuracy: {nn_accuracy:.4f}")

# Find best classifier
classifiers = ["SVM", "Softmax", "Neural Network"]
accuracies = [svm_accuracy, softmax_accuracy, nn_accuracy]
best_idx = accuracies.index(max(accuracies))
print(f"\n🏆 Best Classifier: {classifiers[best_idx]} ({accuracies[best_idx]:.4f})")

# Plot performance comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel("Accuracy", fontsize=12)
plt.title("Classifier Performance Comparison", fontsize=14, fontweight='bold')
plt.ylim([0, 1.0])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✓ Comparison complete!")