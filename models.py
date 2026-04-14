import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class Net1(nn.Module):
    """两层隐藏层的全连接网络 [cite: 16]"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 180 * 180, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, Config.NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(self.flatten(x)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Net2(nn.Module):
    """卷积网络 (Figure 1) [cite: 17]"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 45 * 45, 256), nn.ReLU(),
            nn.Linear(256, Config.NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class Net3(nn.Module):
    """带 BN 的卷积网络 [cite: 18, 19]"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 45 * 45, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, Config.NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class Net4(Net3):
    """与 Net3 结构相同，使用 RMSProp 优化器 [cite: 20]"""
    pass

class Net5(nn.Module):
    """LSTM 网络 [cite: 21]"""
    def __init__(self, input_size=40, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, Config.NUM_CLASSES)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class Net6(Net5):
    """与 Net5 结构相同，使用 GAN 增强数据 """
    pass

# --- GAN 模块用于数据增强 ---
class Generator(nn.Module):
    def __init__(self, latent_dim=100, seq_len=1300, feature_dim=40):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, seq_len * feature_dim),
            nn.Tanh()
        )
        self.seq_len, self.feature_dim = seq_len, feature_dim

    def forward(self, z):
        return self.model(z).view(-1, self.seq_len, self.feature_dim)

class Discriminator(nn.Module):
    def __init__(self, seq_len=1300, feature_dim=40):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * feature_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, img): return self.model(img)