import torch.nn as nn

class GestureRecognition(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        初始化網路結構
        :param input_size: 輸入特徵的數量 (例如：21個點的 X,Y 座標，總共 42 個數值)
        :param num_classes: 要辨識的手勢類別數量 (例如：點擊、移動、滾動，共 3 種)
        """
        super(GestureRecognition, self).__init__()

        # LSTM layer
        # input_size: The number of expected features in the input x
        # hidden_size: The number of features in the hidden state h
        # num_layers: Number of recurrent layers
        # batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        self.hidden_size = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        定義數據在網路中的傳遞流程 (前向傳播)
        :param x: 輸入的手部座標數據序列 (batch_size, seq_len, input_size)
        :return: 各個手勢的機率預測值
        """
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out