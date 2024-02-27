import torch
import torch.nn as nn
import torch.optim as optim
import xml.etree.ElementTree as ET
import os

# Define a custom dataset class for reading shredded XML data
class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, xml_folder):
        self.xml_folder = xml_folder
        self.xml_files = os.listdir(xml_folder)

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, idx):
        xml_file = os.path.join(self.xml_folder, self.xml_files[idx])
        # Parse XML and extract relevant information for commercial insurance policies
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Example: Extracting policy type and premium amount
        policy_type = root.find('policy_type').text
        premium_amount = float(root.find('premium_amount').text)
        
        # Convert policy type to a numerical representation (assuming categorical classification)
        if policy_type == 'auto':
            label = 0
        elif policy_type == 'home':
            label = 1
        elif policy_type == 'health':
            label = 2
        else:
            label = 3  # Other category
        
        return premium_amount, label

# Define a simple MLP (Multi-Layer Perceptron) model for classification
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)  # Output layer for 4 categories

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define paths to your shredded XML data
xml_folder = "path/to/your/shredded/xml/data"

# Create custom dataset
dataset = XMLDataset(xml_folder)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# Instantiate the MLP model
model = MLP()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for premium_amount, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(premium_amount.unsqueeze(1))  # Unsqueezing to match expected input shape
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Validation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for premium_amount, labels in val_loader:
        outputs = model(premium_amount.unsqueeze(1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on validation set: {(100 * correct / total):.2f}%")
