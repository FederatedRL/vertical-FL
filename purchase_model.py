import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#data preprocessing
filename = '../../../../data/purchase/dataset_purchase.txt'
data = np.loadtxt(filename, delimiter=",")
X = data[:, 1:]
X = np.array(X, dtype='float32')
print("X shape:",X.shape)
y = data[:, 0]
y = np.array([i - 1 for i in y])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
poison_num = int(0.3*X_train.shape[0])
train_data = torch.from_numpy(X_train)
party_A_train_data = train_data[:, :300]
party_B_train_data = train_data[:, 300:]
train_label = torch.from_numpy(y_train)

A_poison_train = party_A_train_data[:poison_num,:]
A_poison_test = party_A_train_data[poison_num:,:]

test_data = torch.from_numpy(X_test)
party_A_test_data = test_data[:, :300]
party_B_test_data = test_data[:, 300:]
test_label = torch.from_numpy(y_test)


train_dataset = TensorDataset(party_A_train_data, party_B_train_data, train_label)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(party_A_test_data, party_B_test_data, test_label)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()
decoder_criterion = torch.nn.MSELoss()

class PurchaseVerticalModel(nn.Module):
    def __init__(self):
        super(PurchaseVerticalModel, self).__init__()
        self.bottom_layerA = nn.Sequential(nn.Linear(300, 512), nn.ReLU(True))
        self.bottom_layerB = nn.Sequential(nn.Linear(300, 512), nn.ReLU(True))
        self.interactive_layerA = nn.Linear(512, 256)
        self.interactive_layerB = nn.Linear(512, 256)
        self.interactive_activation_layer = nn.ReLU(True)
        self.top_layer = nn.Sequential(nn.Linear(512, 100))

    def forward(self,inputA,inputB):
        PartyA_output = self.bottom_layerA(inputA)
        PartyB_output = self.bottom_layerB(inputB)
        #interactive_activation_input = self.interactive_layerA(PartyA_output)+self.interactive_layerB(PartyB_output)
        interactive_activation_input = torch.cat((self.interactive_layerA(PartyA_output), self.interactive_layerB(PartyB_output)), 1)
        #interactive_activation_input = self.interactive_layerA(PartyA_output)
        interactive_output = self.interactive_activation_layer(interactive_activation_input)
        output = self.top_layer(interactive_output)
        return PartyA_output, output

class PurchaseDecoderInversion(nn.Module):
    def __init__(self):
        super(PurchaseDecoderInversion, self).__init__()
        self.layers = nn.Sequential(nn.ReLU(True),nn.Linear(512,300))

    def forward(self,encoded_feature):
        decoded_feature = self.layers(encoded_feature)
        return decoded_feature

def train_purchase_vertical_model(net, optimizer, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (A_inputs, B_inputs, targets) in enumerate(train_loader):
        A_inputs, B_inputs, targets = A_inputs.to(device), B_inputs.to(device), targets.to(device)
        feature, outputs = net(A_inputs, B_inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_accuracy = correct / float(total)
    print("train accuracy {:.2f}%".format(train_accuracy * 100))
    return train_accuracy


def test_purchase_normal_model(net, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (A_inputs, B_inputs, targets) in enumerate(test_loader):
            A_inputs, B_inputs, targets = A_inputs.to(device), B_inputs.to(device), targets.to(device)
            feature, outputs = net(A_inputs, B_inputs)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        val_accuracy = correct / float(total)
        print("test accuracy = {:.2f}%".format(val_accuracy * 100))

    return val_accuracy

def train_purchase_vertical_inversion_decoder(net, decoder, optimizer, decoder_optimizer, poison_ratio,
                                              decoder_epoch_num,
                                              device):
    """

    Args:
        net:
        decoder:
        optimizer:
        decoder_optimizer:
        poison_ratio:
        decoder_epoch_num:
        device:

    Returns:

    """
    net.train()
    correct = 0
    total = 0
    for batch_idx, (A_inputs, B_inputs, targets) in enumerate(train_loader):
        A_inputs, B_inputs, targets = A_inputs.to(device), B_inputs.to(device), targets.to(device)
        feature, outputs = net(A_inputs, B_inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        # calculate accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_accuracy = correct / float(total)
    print("train accuracy {:.2f}%".format(train_accuracy * 100))
    torch.save(net,'vertical_model.pkl')

    decoder_train_data = net.bottom_layerA(A_poison_train.to(device))
    decoder_train_label = A_poison_train
    decoder_test_data = net.bottom_layerA(A_poison_test.to(device))
    decoder_test_label = A_poison_test
    print("decoder data:",decoder_train_data.shape)
    # decoder dataset
    decoder_train_dataset = TensorDataset(decoder_train_data, decoder_train_label)
    decoder_train_loader = DataLoader(dataset=decoder_train_dataset, batch_size=decoder_train_data.size(0),
                                      shuffle=True)
    decoder_test_dataset = TensorDataset(decoder_test_data, decoder_test_label)
    decoder_test_loader = DataLoader(dataset=decoder_test_dataset, batch_size=decoder_train_data.size(0), shuffle=False)

    train_privacy = 0
    test_privacy = 0
    loss_list = []
    for decoder_epoch in range(decoder_epoch_num):
        print("train decoder {}/{}".format(decoder_epoch, decoder_epoch_num))
        # train decoder
        decoder_train_loss = 0.0
        for batch_idx, (decoder_inputs, decoder_targets) in enumerate(decoder_train_loader):
            decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(device)
            outputs = decoder(decoder_inputs)
            decoder_optimizer.zero_grad()
            loss = decoder_criterion(outputs, decoder_targets)
            loss_list.append(loss)
            loss.backward(retain_graph=True)
            decoder_optimizer.step()
            # calculate accuracy
            decoder_train_loss += loss.item() * decoder_inputs.size(0)
        train_privacy = decoder_train_loss / float(len(decoder_train_loader.dataset))
        print('train privacy', train_privacy)
        # test decoder
        decoder_test_loss = 0.0
        incorrect = 0
        with torch.no_grad():
            for batch_idx, (decoder_inputs, decoder_targets) in enumerate(decoder_test_loader):
                decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(
                    device)
                outputs = decoder(decoder_inputs)
                outputs_new = np.array(outputs.to('cpu'))
                #print("original outputs:",outputs_new)
                targets_new = np.array(decoder_targets.to('cpu'),dtype='int32')
                outputs_new = np.where(outputs_new>=0.5,1,0)
                #print("new outputs:",outputs_new)
                #print("targets:",targets_new)
                #print("shape:",outputs_new.shape)
                #loss = decoder_criterion(outputs, decoder_targets)
                #decoder_test_loss += loss.item() * decoder_inputs.size(0)
                loss = len(np.argwhere(outputs_new!=targets_new))
                incorrect+=loss
        print("incorrect:",incorrect)
        test_privacy = incorrect / (float(len(decoder_test_loader.dataset))*300)
        print('test privacy', test_privacy)
    plt.plot(loss_list)
    plt.show()
    torch.save(decoder,'decoder.pkl')
    return train_accuracy, train_privacy, test_privacy