import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class BottomModelA(nn.Module):
    def __init__(self):
        super(BottomModelA, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(5, 10), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class BottomModelB(nn.Module):
    def __init__(self):
        super(BottomModelB, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(5, 10), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Interactive(nn.Module):
    def __init__(self):
        super(Interactive, self).__init__()
        self.layerA = nn.Sequential(nn.Linear(20, 10))
        self.layerB = nn.Sequential(nn.Linear(20, 10))
        self.activation = nn.ReLU(True)

    def forward(self, a, b):
        a = self.layerA(a)
        b = self.layerB(b)
        x = self.activation(a + b)
        return x


class TopModel(nn.Module):
    def __init__(self):
        super(TopModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(10, 15), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(15, 20), nn.ReLU(True))
        self.layer3 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


if __name__ == '__main__':
    bottom_model_A = BottomModelA()
    bottom_model_B = BottomModelB()
    interactive = Interactive()
    top_model = TopModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(top_model.parameters(), lr=0.01, momentum=0.9)
    optimizer2 = optim.SGD(interactive.parameters(), lr=0.01, momentum=0.9)
    optimizer3 = optim.SGD(bottom_model_A.parameters(), lr=0.01, momentum=0.9)
    optimizer4 = optim.SGD(bottom_model_B.parameters(), lr=0.01, momentum=0.9)

    guest_data = pd.read_csv('give_credit_a.csv', usecols=['x0', 'x1', 'x2', 'x3', 'x4'])
    guest_data = torch.from_numpy(np.array(guest_data, dtype='float32'))
    host_data = pd.read_csv('give_credit_b.csv', usecols=['x0', 'x1', 'x2', 'x3', 'x4'])
    host_data = torch.from_numpy(np.array(host_data, dtype='float32'))
    label = pd.read_csv('give_credit_b.csv', usecols=['y'])
    label = np.array(label).reshape(150000)
    label = torch.from_numpy(label)

    host_dataset = TensorDataset(host_data, label)
    host_loader = DataLoader(dataset=host_dataset, batch_size=64, shuffle=False)
    guest_dataset = TensorDataset(guest_data)
    guest_loader = DataLoader(dataset=guest_dataset, batch_size=64, shuffle=False)

    num_of_epochs = 10
    for epoch in range(num_of_epochs):
        running_loss = 0.0
        running_corrrect = 0
        print("Epoch  {}/{}".format(epoch + 1, num_of_epochs))
        host_iter = iter(host_loader)
        guest_iter = iter(guest_loader)
        while True:
            try:
                dataA = guest_iter.next()[0]
                dataA = Variable(dataA)
                dataB, data_label = host_iter.next()
                dataB = Variable(dataB)
                data_label = Variable(data_label)
            except StopIteration:
                break
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()

            PartyA_output = bottom_model_A(dataA)
            PartyB_output = bottom_model_B(dataB)
            Interactive_output = interactive(PartyA_output, PartyB_output)
            output = top_model(Interactive_output)
            _, pred = torch.max(output.data, 1)
            loss = criterion(output, data_label)
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            running_loss += float(loss.data)
            # print("label:",data_label)
            # print("prediction:",pred)
            running_corrrect += torch.sum(pred == data_label)

        print("correct:", int(running_corrrect))
        print("Loss is {:.4f}, Train Accuracy is:{:.4f}%"
              .format(running_loss / len(host_data), 100 * int(running_corrrect) / len(host_data)))
