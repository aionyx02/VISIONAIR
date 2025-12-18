# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.gesture_recognition import GestureRecognition
import numpy as np

def train_model():
    data_dir = "data"
    gestures = os.listdir(data_dir)
    
    X = []
    y = []
    
    for i, gesture in enumerate(gestures):
        gesture_dir = os.path.join(data_dir, gesture)
        for filename in os.listdir(gesture_dir):
            df = pd.read_csv(os.path.join(gesture_dir, filename), header=None)
            X.append(df.values.flatten())
            y.append(i)
            
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    input_size = X_train.shape[1]
    num_classes = len(gestures)
    
    model = GestureRecognition(input_size, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(10):
        for landmarks, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
    # Save the model
    torch.save(model.state_dict(), os.path.join("models", "gesture_model.pth"))
    
    # Evaluate the model
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    train_model()
