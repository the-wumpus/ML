# Thomas Ehret
# CS519 Applied ML
# Dr Cao
# NMSU Sp23
# project 5

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load the preprocessed image and label datasets
images = np.load('dataset_images.npy')
labels = np.load('dataset_labels.npy')

# Split the data into training and validation sets
split_ratio = 0.8
split_idx = int(split_ratio * len(images))
train_images, train_labels = images[:split_idx], labels[:split_idx]
val_images, val_labels = images[split_idx:], labels[split_idx:]

# Define the CNN model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the CNN model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the CNN model
num_epochs = 10
batch_size = 64
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, len(train_images), batch_size):
        # Prepare a batch of data
        batch_images = train_images[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        inputs = torch.from_numpy(batch_images).float()
        targets = torch.from_numpy(batch_labels).long()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item() * batch_size
    epoch_loss = running_loss / len(train_images)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

# Evaluate the CNN model on the validation set
with torch.no_grad():
    inputs = torch.from_numpy(val_images).float()
    targets = torch.from_numpy(val_labels).long()
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    accuracy = correct / total
    print('Validation Accuracy: {:.2f}%'.format(accuracy * 100))


