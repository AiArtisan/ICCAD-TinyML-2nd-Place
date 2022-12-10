import torch.nn as nn
import torch.nn.functional as F

class SunNet(nn.Module):
    def __init__(self):
        super(SunNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(10, 1), stride=(6,1), padding=(1,0)),
            nn.ReLU(True),
            # nn.BatchNorm2d(2, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(9, 1), stride=(5,1), padding=(1,0)),
            nn.ReLU(True),
            # nn.BatchNorm2d(4, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(8, 1), stride=(4,1), padding=(1,0)),
            nn.ReLU(True),
            # nn.BatchNorm2d(8, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.25),
            nn.Linear(in_features=32, out_features=16)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=16, out_features=2)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv3_output = conv3_output.view(-1,32)
        
        fc1_output = F.relu(self.fc1(conv3_output))
        fc2_output = self.fc2(fc1_output)

        return fc2_output