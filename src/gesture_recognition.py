import torch.nn as nn

class GestureRecognition(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        初始化網路結構
        :param input_size: 輸入特徵的數量 (例如：21個點的 X,Y 座標，總共 42 個數值)
        :param num_classes: 要辨識的手勢類別數量 (例如：點擊、移動、滾動，共 3 種)
        """
        super(GestureRecognition, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(128, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        定義數據在網路中的傳遞流程 (前向傳播)
        :param x: 輸入的手部座標數據
        :return: 各個手勢的機率預測值
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out