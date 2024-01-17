import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# validation loop
def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():  # Ensure no gradients are computed during validation
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            labels = labels.long()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Compute accuracy
            _, predictions = torch.max(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Accumulate total loss
            total_loss += loss.item()

    # Calculate accuracy and average loss for the validation set
    accuracy = 100.0 * total_correct / total_samples
    average_loss = total_loss / len(val_loader)

    # Print or log validation metrics
    print(f'Validation Loss: {average_loss:.4f} | Validation Accuracy: {accuracy:.2f}%')

    return accuracy, average_loss


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = torch.tensor(self.dataframe.drop('class', axis=1).iloc[idx, :].values, dtype=torch.float32)
        target = torch.tensor(self.dataframe.loc[idx, 'class'], dtype=torch.float32)
        return features, target
    
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        
        return out

def main():
    # Load dataset 
    df = pd.read_csv('.\data\RT_IOT2022_sanitised.csv')

    # Split into training and testing dataframes
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Test using the first 1000 rows of data
    df = df.iloc[:1000]

    train_size = int(0.6 * len(df))
    validation_size = int(0.8 * len(df))

    train_df = df.iloc[:train_size, :]
    train_df = train_df.reset_index(drop=True)

    validation_df = df.iloc[train_size:validation_size, :]
    validation_df = train_df.reset_index(drop=True)

    test_df = df.iloc[validation_size:, :]
    test_df = test_df.reset_index(drop=True)

    # Define hyperparamters
    input_size = train_df.shape[1] - 1
    hidden_size = 128
    num_classes = 2
    num_epochs = 20
    batch_size = 10
    learning_rate = 0.01

    # Load data into a data loader 
    train_dataset = CustomDataset(train_df)
    validation_dataset = CustomDataset(validation_df)
    test_dataset = CustomDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model 
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # loss and optimiser 
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        total_loss = 0
        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            labels = labels.long()

            # forward
            outputs = model(features)
            loss = criterion(outputs, labels)

            # backward
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Compute accuracy
            _, predictions = torch.max(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Accumulate the loss 
            total_loss += loss.item()

            # if (i+1) % 5 == 0:
            #     print(f'Epoch {epoch+1} / {num_epochs} | Step {i+1} / {n_total_steps} | Loss = {loss.item():.4f}')

        # Caculate the accuracy and average loss for the epoch
        epoch_accuracy = 100.0 * total_correct / total_samples
        epoch_loss = total_loss / len(train_loader)
        
        # Append epoch loss and accuracy to the lists
        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)
        
        # Print and/or log the epoch metrics
        print(f'Epoch {epoch+1} / {num_epochs} | Loss = {epoch_loss:.4f} | Accuracy = {epoch_accuracy:.2f}%') 

        # Model validation 
        val_accuracy, val_loss = validate(model, validation_loader, criterion, device)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)

    # test
    test_losses = []
    test_accuracies = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            

            # value, index
            _, predictions = torch.max(outputs, dim=1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
        
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy = {acc}')

    return


main()