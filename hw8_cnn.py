# Thomas Ehret
# CS519 Applied ML
# Dr Cao
# NMSU Sp23
# Homework 8

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
num_epochs = 2
batch_size = 64
learning_rate = 0.001

# Define transform to normalize data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

# Load MNIST dataset
print('Loading MNIST dataset...')
train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define CNN architecture
# adapted from Sebastian Raschka's PyTorch Youtube tutorial
# https://www.youtube.com/watch?v=B5GHmm3KN2A
# Machine Learning with PyTorch and Scikit-Learn Book

# default parameters
# kernel_size=3, stride=1, padding_mode='valid' exp. zeroes
# the forward function is adpated directly frorm the PyTorch documentation.
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding_mode='zeros')
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv2 = nn.Conv2d(4, 2, kernel_size=3, stride=3, padding_mode='zeros')
        #self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc = nn.Linear(64*1*1, 128)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Initialize model and optimizer
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define loss function
# this is the softmax cross entropy loss
criterion = nn.CrossEntropyLoss()

# Train the model
total_time = time.time()
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss every 100 batches
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
    cnn_fit_time = time.time() - epoch_start_time
    print(f"CNN epoch fit time: {cnn_fit_time:.3f} seconds")

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
model_time = time.time() - total_time
print(f"CNN total time: {model_time:.3f} seconds")
