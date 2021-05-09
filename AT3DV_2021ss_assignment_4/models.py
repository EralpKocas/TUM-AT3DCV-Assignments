import torch
import torch.nn as nn
import torch.nn.functional as F


class point_model(nn.Module):
    ### Complete for task 1
    def __init__(self, num_classes):
        super(point_model, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)

        #self.mlp1 = nn.Linear(3, 64)
        self.mlp1 = nn.Conv1d(3, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)

        #self.mlp2 = nn.Linear(64, 64)
        self.mlp2 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)

        #self.mlp3 = nn.Linear(64, 64)
        self.mlp3 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)

        #self.mlp4 = nn.Linear(64, 128)
        self.mlp4 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(128)

        #self.mlp5 = nn.Linear(128, 1024)
        self.mlp5 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(1024)

        self.max_pool = nn.MaxPool1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 128)
        self.bn7 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 3, -1))
        x = self.relu(self.bn1(self.mlp1(x)))
        x = self.relu(self.bn2(self.mlp2(x)))
        x = self.relu(self.bn3(self.mlp3(x)))
        x = self.relu(self.bn4(self.mlp4(x)))
        x = self.relu(self.bn5(self.mlp5(x)))
        x = self.max_pool(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.relu(self.bn6(self.fc1(x)))
        x = self.relu(self.bn7(self.fc2(x)))
        x = self.fc3(self.dropout(x))

        #x = self.relu(self.mlp1(x))
        #x = self.relu(self.mlp2(x))
        #x = self.relu(self.mlp3(x))
        #x = self.relu(self.mlp4(x))
        #x = self.relu(self.mlp5(x))
        #x = self.max_pool(x)
        #x = torch.reshape(x, (x.shape[0], -1))
        #x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        #x = self.fc3(self.dropout(x))
        return x
        

class voxel_model(nn.Module):
    ### Complete for task 2
    def __init__(self,num_classes):
        super(voxel_model, self).__init__()
        self.conv1=nn.Conv3d(1,8,3)

    def forward(self,x):

        return x
        

class spectral_model(nn.Module):
    ### Complete for task 3
    def __init__(self,num_classes):
        super(spectral_model, self).__init__()
        self.mlp1=nn.Conv1d(6,64,1)

    def forward(self,x):

        return x