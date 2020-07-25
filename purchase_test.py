import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import time

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(600, 1024), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(1024, 512),nn.Dropout(0.2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(512, 256),nn.Dropout(0.2), nn.ReLU(True))
        self.layer4 = nn.Linear(256, 100)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = F.log_softmax(x)
        return x

if __name__ == '__main__':
    start = time.time()
    model = Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # data preprocessing
    filename = 'dataset_purchase.txt'
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, 1:]
    X = np.array(X, dtype='float32')
    y = data[:, 0]
    y = np.array([i - 1 for i in y])
    X_train = X[:150000]
    X_test = X[150000:]
    y_train = y[:150000]
    y_test = y[150000:]
    train_data = torch.from_numpy(X_train)
    test_data = torch.from_numpy(X_test)
    train_label = torch.from_numpy(y_train)
    test_label = torch.from_numpy(y_test)

    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(test_data, test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

    num_of_epochs = 50
    loss_list = []
    for epoch in range(1, num_of_epochs + 1):
        running_loss = 0.0
        running_correct = 0
        print("Epoch  {}/{}".format(epoch, num_of_epochs))
        for data in train_loader:
            X_train, y_train = data
            X_train, y_train = Variable(X_train), Variable(y_train)
            outputs = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train.long())
            loss.backward()
            optimizer.step()
            running_loss += float(loss.data)
            loss_list.append(float(loss.data))
            running_correct += torch.sum(pred == y_train)
        print("Loss is {:.4f}, Train Accuracy is:{:.4f}%"
              .format(running_loss / len(train_data), 100 * int(running_correct) / len(train_data)))
    plt.plot(loss_list)
    plt.show()
    testing_correct = 0
    for data in test_loader:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test)
    print("Test Accuracy is:{:.4f}%".format(100 * int(testing_correct) / len(test_data)))