import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class VerticalFL_Model(nn.Module):
    def __init__(self):
        super(VerticalFL_Model, self).__init__()
        self.bottom_layerA = nn.Sequential(nn.Linear(5, 10), nn.ReLU(True))
        self.bottom_layerB = nn.Sequential(nn.Linear(5, 10), nn.ReLU(True))
        self.interactive_layerA = nn.Linear(10, 15)
        self.interactive_layerB = nn.Linear(10, 15)
        self.interactive_activation_layer = nn.ReLU(True)
        self.top_layer = nn.Sequential(nn.Linear(15, 10), nn.ReLU(True), nn.Linear(10, 2))

    def forward(self, inputA, inputB):
        PartyA_output = self.bottom_layerA(inputA)
        PartyB_output = self.bottom_layerB(inputB)
        interactive_activation_input = self.interactive_layerA(PartyA_output) + self.interactive_layerB(PartyB_output)
        interactive_output = self.interactive_activation_layer(interactive_activation_input)
        output = self.top_layer(interactive_output)
        return output


if __name__ == '__main__':
    model = VerticalFL_Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # data preprocessing
    guest_data = pd.read_csv('give_credit_a.csv', usecols=['x0', 'x1', 'x2', 'x3', 'x4'])
    guest_data = np.array(guest_data, dtype='float32')
    guest_train_data = torch.from_numpy(guest_data[:120000])
    guest_test_data = torch.from_numpy(guest_data[120000:])
    host_data = pd.read_csv('give_credit_b.csv', usecols=['x0', 'x1', 'x2', 'x3', 'x4'])
    host_data = np.array(host_data, dtype='float32')
    host_train_data = torch.from_numpy(host_data[:120000])
    host_test_data = torch.from_numpy(host_data[120000:])
    label = pd.read_csv('give_credit_b.csv', usecols=['y'])
    label = np.array(label).reshape(150000)
    train_label = torch.from_numpy(label[:120000])
    test_label = torch.from_numpy(label[120000:])

    # construct datasets
    host_train_dataset = TensorDataset(host_train_data, train_label)
    host_train_loader = DataLoader(dataset=host_train_dataset, batch_size=64, shuffle=False)
    host_test_dataset = TensorDataset(host_test_data, test_label)
    host_test_loader = DataLoader(dataset=host_test_dataset, batch_size=64, shuffle=False)
    guest_train_dataset = TensorDataset(guest_train_data)
    guest_train_loader = DataLoader(dataset=guest_train_dataset, batch_size=64, shuffle=False)
    guest_test_dataset = TensorDataset(guest_test_data)
    guest_test_loader = DataLoader(dataset=guest_test_dataset, batch_size=64, shuffle=False)

    # train the model
    num_of_epochs = 5
    for epoch in range(1, num_of_epochs + 1):
        running_loss = 0.0
        running_correct = 0
        print("Epoch  {}/{}".format(epoch, num_of_epochs))
        host_train_iter = iter(host_train_loader)
        guest_train_iter = iter(guest_train_loader)
        # loss_list = []
        while True:
            try:
                train_dataA = guest_train_iter.next()[0]
                train_dataA = Variable(train_dataA)
                train_dataB, train_data_label = host_train_iter.next()
                train_dataB = Variable(train_dataB)
                train_data_label = Variable(train_data_label)
            except StopIteration:
                break
            optimizer.zero_grad()

            output = model(train_dataA, train_dataB)
            _, pred = torch.max(output.data, 1)
            loss = criterion(output, train_data_label)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.data) * train_dataA.size(0)
            # print("loss:", float(loss.data))
            # loss_list.append(float(loss.data))
            # print("label:",data_label)
            # print("prediction:",pred)
            running_correct += torch.sum(pred == train_data_label)
        print("correct:", int(running_correct))
        print("Loss is {:.4f}, Train Accuracy is:{:.4f}%"
              .format(running_loss / len(host_train_data), 100 * int(running_correct) / len(host_train_data)))

    # evaluation
    testing_correct = 0
    host_test_iter = iter(host_test_loader)
    guest_test_iter = iter(guest_test_loader)
    while True:
        try:
            test_dataA = guest_test_iter.next()[0]
            test_dataA = Variable(test_dataA)
            test_dataB, test_data_label = host_test_iter.next()
            test_dataB = Variable(test_dataB)
            test_data_label = Variable(test_data_label)
        except StopIteration:
            break
        output = model(test_dataA, test_dataB)
        _, pred = torch.max(output.data, 1)
        testing_correct += torch.sum(pred == test_data_label)
    print("Test Accuracy is:{:.4f}%".format(100 * int(testing_correct) / len(host_test_data)))
