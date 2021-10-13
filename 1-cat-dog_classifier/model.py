import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 112
            nn.MaxPool2d(2, 2),  # 56
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # 56
            nn.MaxPool2d(2, 2),  # 28
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 28
            nn.MaxPool2d(2, 2),  # 14
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # 14
            nn.MaxPool2d(2, 2),  # 7
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*128, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, input):
        x = input['tens']
        x = self.cnn(x)
        output = self.fc(x)
        return output

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(SimpleDNN, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.output_dim = num_classes
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.batchnorm = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, input):
        x = input['tens']
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.batchnorm(x)
        output = self.fc3(x)
        return output

class SimplePatchRNN(nn.Module):
    def __init__(self, num_classes=2, batch_size=16):
        super(SimplePatchRNN, self).__init__()
        self.batch_size = batch_size
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 48
            nn.MaxPool2d(2, 2),  # 24
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # 24
            nn.MaxPool2d(2, 2),  # 12
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 12
            nn.MaxPool2d(2, 2),  # 6
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 6
            nn.MaxPool2d(2, 2),  # 3
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(512)
        )
        self.gru = nn.GRU(512, 512, 2, bidirectional=True)
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, input):
        x = input['tens']
        patch_feat_list = torch.zeros((9, self.batch_size, 512))
        # patch
        for p in range(9):
            patch = x[:, p]
            patch_feat = self.cnn(patch)  # [B, 512]
            patch_feat = self.fc1(patch_feat)  # [B, 256]
            patch_feat_list[p] = patch_feat  # [9 ,B, 256]
        output, hn = self.gru(patch_feat_list)  # torch.Size([9, 16, 1024])
        output = self.fc2(output[-1])  #[B, 1024]
        return output

