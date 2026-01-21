##### Step 3: Load and Prepare Dataset

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define dataset transformation (normalize and convert to tensors)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define labels subset (3 classes)
selected_classes = [0, 1, 2]  # Airplane, Automobile, Bird
class_names = ['Airplane', 'Automobile', 'Bird']

# Filter dataset
trainset.data = trainset.data[[i for i in range(len(trainset.targets)) if trainset.targets[i] in selected_classes]]
trainset.targets = [trainset.targets[i] for i in range(len(trainset.targets)) if trainset.targets[i] in selected_classes]

testset.data = testset.data[[i for i in range(len(testset.targets)) if testset.targets[i] in selected_classes]]
testset.targets = [testset.targets[i] for i in range(len(testset.targets)) if testset.targets[i] in selected_classes]

print(f"Training samples: {len(trainset.data)}")
print(f"Test samples: {len(testset.data)}")

# VISUAL 1: Display sample images from each class
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, class_idx in enumerate(selected_classes):
    # Find first image of this class
    idx = trainset.targets.index(class_idx)
    axes[i].imshow(trainset.data[idx])
    axes[i].set_title(f"{class_names[i]}", fontsize=14, fontweight='bold')
    axes[i].axis('off')

plt.suptitle("Sample Images from Each Class", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: dataset_samples.png")



######  SVM Classifier (with confusion matrix)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import time
import numpy as np

print("\n" + "="*50)
print("SVM CLASSIFIER")
print("="*50)

# Flatten images for SVM (from 32x32x3 to 1D array)
print("Preparing data...")
X_train = trainset.data.reshape(len(trainset.data), -1)
y_train = np.array(trainset.targets)

X_test = testset.data.reshape(len(testset.data), -1)
y_test = np.array(testset.targets)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Option to use subset for faster training
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

svm_training_time = time.time() - start_time
print(f"\n✓ Training completed in {svm_training_time:.2f} seconds ({svm_training_time/60:.1f} minutes)")

# Predict and evaluate
print("\n📊 Making predictions...")
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print(f"\n✓ SVM Accuracy: {svm_accuracy:.4f}")

# VISUAL 2: SVM Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('SVM Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: svm_confusion_matrix.png")


########2. Softmax Classifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import time

print("\n" + "="*50)
print("SOFTMAX CLASSIFIER")
print("="*50)

# Train Softmax classifier
print("\n🚀 Training Softmax classifier...")
start_time = time.time()

softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, verbose=1)
softmax.fit(X_train, y_train)

softmax_training_time = time.time() - start_time
print(f"\n✓ Training completed in {softmax_training_time:.2f} seconds")

# Predict and evaluate
print("\n📊 Making predictions...")
y_pred_softmax = softmax.predict(X_test)
softmax_accuracy = accuracy_score(y_test, y_pred_softmax)

print(f"\n✓ Softmax Accuracy: {softmax_accuracy:.4f}")

# VISUAL 3: Softmax Confusion Matrix
cm_softmax = confusion_matrix(y_test, y_pred_softmax)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_softmax, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Softmax Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('softmax_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: softmax_confusion_matrix.png")


#########3. Two-layer Neural Network


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
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

# Store loss history for plotting
loss_history = []

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
    
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

nn_training_time = time.time() - start_time
print(f"\n✓ Training completed in {nn_training_time:.2f} seconds ({nn_training_time/60:.1f} minutes)")

# VISUAL 4: Training Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_history, linewidth=2, color='#2ecc71')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Neural Network Training Loss', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('nn_training_loss.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: nn_training_loss.png")

# Evaluate on test set
print("\n📊 Evaluating Neural Network...")
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    nn_accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"✓ Neural Network Accuracy: {nn_accuracy:.4f}")

# VISUAL 5: Neural Network Confusion Matrix
y_pred_nn = predicted.numpy()
cm_nn = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Neural Network Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('nn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: nn_confusion_matrix.png")



#######📊 Step 5: Compare Classifier Performance
import matplotlib.pyplot as plt
import numpy as np

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
training_times = [svm_training_time, softmax_training_time, nn_training_time]
best_idx = accuracies.index(max(accuracies))
print(f"\n🏆 Best Classifier: {classifiers[best_idx]} ({accuracies[best_idx]:.4f})")

# VISUAL 6: Accuracy Comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel("Accuracy", fontsize=12)
plt.title("Classifier Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylim([0, 1.0])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: accuracy_comparison.png")

# VISUAL 7: Training Time Comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, training_times, color=['#9b59b6', '#f39c12', '#1abc9c'])

for i, (bar, time_val) in enumerate(zip(bars, training_times)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.02, 
             f'{time_val:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel("Training Time (seconds)", fontsize=12)
plt.title("Classifier Training Time Comparison", fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: training_time_comparison.png")

# VISUAL 8: Per-Class Accuracy Comparison
from sklearn.metrics import classification_report

print("\n📊 Per-Class Performance:")
for i, classifier_name in enumerate(classifiers):
    print(f"\n{classifier_name}:")
    if i == 0:
        y_pred = y_pred_svm
    elif i == 1:
        y_pred = y_pred_softmax
    else:
        y_pred = y_pred_nn
    
    print(classification_report(y_test, y_pred, target_names=class_names))

# Create per-class accuracy visualization
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(class_names))
width = 0.25

# Calculate per-class accuracies
def per_class_accuracy(y_true, y_pred, num_classes):
    accuracies = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            acc = (y_pred[mask] == c).sum() / mask.sum()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    return accuracies

svm_per_class = per_class_accuracy(y_test, y_pred_svm, len(selected_classes))
softmax_per_class = per_class_accuracy(y_test, y_pred_softmax, len(selected_classes))
nn_per_class = per_class_accuracy(y_test, y_pred_nn, len(selected_classes))

bars1 = ax.bar(x - width, svm_per_class, width, label='SVM', color='#3498db')
bars2 = ax.bar(x, softmax_per_class, width, label='Softmax', color='#e74c3c')
bars3 = ax.bar(x + width, nn_per_class, width, label='Neural Network', color='#2ecc71')

ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: per_class_accuracy.png")
