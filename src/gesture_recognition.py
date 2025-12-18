# src/gesture_recognition.py
import torch
import torch.nn as nn

class GestureRecognition(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureRecognition, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
