import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(22, 10),nn.ReLU(),nn.Linear(10,5))
        self.cloud = nn.Sequential(nn.ReLU(),nn.Linear(5,2))
        #self.decoder = nn.Sequential(nn.Linear(5,10),nn.ReLU(),nn.Linear(10,22))

    def forward(self,original_feature):
        encoded_feature = self.encoder(original_feature)
        output = self.cloud(encoded_feature)
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(nn.Linear(5,10),nn.ReLU(),nn.Linear(10,22))

    def forward(self,encoded_feature):
        decoded_feature = self.layers(encoded_feature)
        return decoded_feature

if __name__ == '__main__':
    model = Model()
    decoder = Decoder()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_decoder = torch.nn.MSELoss()
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=0.001)

    filename = 'GroundTruth_and_Features.csv'
    FeatureAmount = 22
    data = np.loadtxt(filename, delimiter=",")[:, :FeatureAmount + 1]
    X = data[:, 1:]
    X = np.array(X, dtype='float32')
    y = data[:, 0]
    X_train = X[:90000]
    X_test = X[90000:]
    y_train = y[:90000]
    y_test = y[90000:]
    train_data = torch.from_numpy(X_train)
    test_data = torch.from_numpy(X_test)
    train_label = torch.from_numpy(y_train)
    test_label = torch.from_numpy(y_test)

    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    test_dataset = TensorDataset(test_data, test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

    #train encoder
    num_of_epochs = 1
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
            #loss_list.append(float(loss.data))
            running_correct += torch.sum(pred == y_train)
        print("Loss is {:.4f}, Train Accuracy is:{:.4f}%"
              .format(running_loss / len(train_data), 100 * int(running_correct) / len(train_data)))
    #plt.plot(loss_list)
    #plt.show()
    testing_correct = 0
    for data in test_loader:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data,1)
        testing_correct += torch.sum(pred == y_test)
    print("Test Accuracy is:{:.4f}%".format(100 * int(testing_correct) / len(test_data)))
    print("Encoder training over!")

    decoder_train_data = model.encoder(train_data)
    decoder_test_data = model.encoder(test_data)
    decoder_train_label = train_data
    decoder_test_label = test_data

    decoder_train_dataset = TensorDataset(decoder_train_data, decoder_train_label)
    decoder_train_loader = DataLoader(dataset=decoder_train_dataset, batch_size=10, shuffle=True)
    decoder_test_dataset = TensorDataset(decoder_test_data, decoder_test_label)
    decoder_test_loader = DataLoader(dataset=decoder_test_dataset, batch_size=1, shuffle=False)

    #train decoder
    loss_list = []
    num_of_epochs_decoder = 10
    for epoch in range(1,num_of_epochs_decoder+1):
        running_loss = 0.0
        print("Epoch  {}/{}".format(epoch, num_of_epochs_decoder))
        for data in decoder_train_loader:
            X_train, Y_train = data
            X_train, Y_train = Variable(X_train), Variable(Y_train)
            outputs = decoder(X_train)
            optimizer_decoder.zero_grad()
            loss = criterion_decoder(outputs, Y_train)
            loss_list.append(float(loss.data))
            loss.backward()
            optimizer_decoder.step()
            running_loss += float(loss.data)
    plt.plot(loss_list)
    plt.show()

    test_loss_list = []
    for data in decoder_test_loader:
        X_test, Y_test = data
        X_test, Y_test = Variable(X_test), Variable(Y_test)
        outputs = decoder(X_test)
        #print("decoded feature:",outputs)
        #print("original feature:",Y_test)
        test_loss = criterion_decoder(outputs, Y_test)
        test_loss_list.append(float(test_loss.data))
    test_loss_list.sort(reverse=True)
    print("test loss:",test_loss_list)
    print("average testing loss: ",sum(test_loss_list)/len(test_loss_list))