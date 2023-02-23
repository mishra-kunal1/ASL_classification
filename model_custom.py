import torch
import torch.nn as nn

class MySeqModel(nn.Module):
    def __init__(self, n_classes):
        super(MySeqModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.dropout1 = nn.Dropout2d(0.5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.dropout2 = nn.Dropout2d(0.5)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=4, stride=2)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(2304 , 256)
        self.fc2 = nn.Linear(256, n_classes)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.dropout1(x)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.dropout2(x)
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        x = self.dropout3(x)
        #print('after dropout layer shape')
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.softmax(x, dim=1)

# n_classes = 29
# seq_model = MySeqModel(n_classes)