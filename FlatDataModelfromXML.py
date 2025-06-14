import os
import torch
import torch.nn as nn
import torch.optim as optim
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# -----------------------------
# Dataset
# -----------------------------
class XMLDataset(Dataset):
    def __init__(self, xml_folder):
        self.xml_files = [os.path.join(xml_folder, f) for f in os.listdir(xml_folder) if f.endswith('.xml')]
        self.premiums = []
        self.labels = []
        self.policy_types = []
        
        # Parse and collect raw data
        for file in self.xml_files:
            try:
                tree = ET.parse(file)
                root = tree.getroot()
                policy_type = root.find('policy_type').text.strip().lower()
                premium = float(root.find('premium_amount').text)
                self.policy_types.append(policy_type)
                self.premiums.append(premium)
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.policy_types)

        # Normalize premiums
        self.scaler = StandardScaler()
        self.premiums = self.scaler.fit_transform(np.array(self.premiums).reshape(-1, 1)).astype(np.float32)

    def __len__(self):
        return len(self.premiums)

    def __getitem__(self, idx):
        return torch.tensor(self.premiums[idx]), torch.tensor(self.labels[idx])

# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden1=64, hidden2=32, output_dim=4):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Training Pipeline
# -----------------------------
def train_model(xml_folder, num_epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = XMLDataset(xml_folder)
    num_classes = len(set(dataset.labels))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = MLP(output_dim=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    print("\nValidation Results:")
    print(classification_report(all_labels, all_preds, target_names=dataset.label_encoder.classes_))

# -----------------------------
# Run Training
# -----------------------------
if __name__ == "__main__":
    xml_folder = "path/to/your/shredded/xml/data"
    train_model(xml_folder)
