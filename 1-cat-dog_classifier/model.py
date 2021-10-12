import torch
import torch.nn as nn
import torch.nn.functional as F

# class VGGNet16(nn.Module):
#     def __init__(self, num_classes):
#         super(VGGNet16, self).__init__()
#         self.num_classes = num_classes
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(7*7*512, 4096),
#             nn.Linear(4096, 4096),
#             nn.Linear(4096, self.num_classes)
#         )
#
#     def forward(self, input_data):
#         x = input_data['tens']
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = x.view(x.size(0), -1)
#         output = self.fc(x)  # dim = 2
#         # output = F.softmax(x, dim=1)
#
#         return output

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 112
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # 56
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # 14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 7
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*128, 512),
            nn.Linear(512, num_classes)
        )

    def forward(self, input):
        x = input['tens']
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(SimpleDNN, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.output_dim = num_classes
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, input):
        x = input['tens']
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output

class SimpleRNN(nn.Module):
    def __init__(self, input_size=56, hidden_size=20, num_classes=2):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(input_size*hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input):
        x = input['tens']
        output, hn = self.gru1(x)
        # print(output.shape)
        output = output.contiguous().view(output.size(0), -1)  # 56*20
        # print(output.shape)  # torch.Size([20, 3136])
        x = self.fc1(output)
        x = self.fc2(x)
        return x






