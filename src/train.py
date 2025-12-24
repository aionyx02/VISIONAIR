import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from gesture_recognition import GestureRecognition
import numpy as np

def train_model():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        return

    gestures = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not gestures:
        print("No gesture data found.")
        return

    X = []
    y = []

    # Assuming each CSV file represents a sequence of frames for one gesture instance
    # We need to pad sequences to a fixed length or use a fixed window size.
    # For simplicity, let's assume we take a fixed number of frames (e.g., 30)
    # If a sample has fewer frames, we pad; if more, we truncate or sample.
    SEQUENCE_LENGTH = 30
    NUM_LANDMARKS = 21 * 2 # 21 landmarks * (x, y) - assuming z is not used or handled differently.
                           # Actually the collector saves [id, cx, cy], so 3 columns per landmark?
                           # Let's check data_collector.py. It writes `writer.writerows(lm_list)`.
                           # lm_list is [[id, cx, cy], ...]. So each row in CSV is one landmark.
                           # A single frame has 21 rows.
                           # A gesture sequence is multiple frames?
                           # Wait, data_collector.py saves ONE frame per file currently?
                           # "if key == ord('s') ... writer.writerows(lm_list)" -> This saves ONE frame's landmarks into a CSV.
                           # The project description mentions "Dynamic Gestures" and LSTM.
                           # But the current data collector seems to collect static frames (one CSV per sample).
                           # If we want to train LSTM for dynamic gestures, we need sequences.
                           # However, if the user wants to stick to the current structure where one CSV = one static pose,
                           # then we should treat it as sequence length = 1 or just use a dense network.
                           # BUT, the user asked to "Implement dynamic gesture recognition using LSTM/GRU".
                           # This implies we should probably treat the input as a sequence.
                           # If the current data is static images, we can't really train a dynamic model effectively unless we simulate sequences.
                           # Let's assume for now we are upgrading the model to LSTM but the data might still be static frames
                           # (which is a bit contradictory, but I must follow the "Implement LSTM" instruction).
                           # OR, maybe the user intends to collect sequences later.
                           # Let's adjust the data loading to be compatible with the LSTM input shape (Batch, Seq, Feature).
                           # If we have static frames, Seq=1.

    # Let's look at data_collector.py again.
    # It saves `lm_list` which is 21 rows of [id, cx, cy].
    # So one CSV file = one frame.
    # To use LSTM, we usually feed a sequence of frames.
    # If we only have single-frame CSVs, we can treat them as a sequence of length 1.

    for i, gesture in enumerate(gestures):
        gesture_dir = os.path.join(data_dir, gesture)
        for filename in os.listdir(gesture_dir):
            if not filename.endswith('.csv'): continue

            file_path = os.path.join(gesture_dir, filename)
            try:
                df = pd.read_csv(file_path, header=None)
                # df shape should be (21, 3) -> 21 landmarks, [id, x, y]
                # We only care about x and y usually for the model, id is just index.
                # Let's extract x and y.
                # Column 0 is ID, 1 is X, 2 is Y.
                coords = df.iloc[:, 1:].values # Shape (21, 2)

                # Flatten to (42,)
                features = coords.flatten()

                X.append(features)
                y.append(i)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not X:
        print("No valid data found.")
        return

    X = np.array(X)
    y = np.array(y)

    # Reshape X for LSTM: (Batch, Sequence_Length, Input_Size)
    # Since we currently have static frames, Sequence_Length = 1
    X = X.reshape(X.shape[0], 1, X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    input_size = X_train.shape[2] # Should be 42
    num_classes = len(gestures)

    model = GestureRecognition(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"Input shape: {X_train.shape}")

    for epoch in range(20): # Increased epochs slightly
        model.train()
        running_loss = 0.0
        for landmarks, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", "gesture_model.pth"))
    print("模型已儲存至 models/gesture_model.pth")

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        if len(y_test) > 0:
            accuracy = (predicted == y_test).sum().item() / len(y_test)
            print(f"測試集準確度: {accuracy * 100:.2f}%")
        else:
            print("測試集為空。")

if __name__ == "__main__":
    train_model()