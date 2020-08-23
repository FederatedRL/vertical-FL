import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle
import math

#data preprocessing
f1 = open('train_data.pickle', 'rb')
f2 = open('train_label.pickle', 'rb')
f3 = open('test_data.pickle', 'rb')
f4 = open('test_label.pickle', 'rb')
X_train = np.array(pickle.load(f1), dtype='float32')
y_train = pickle.load(f2)
X_test = np.array(pickle.load(f3), dtype='float32')
y_test = pickle.load(f4)
f1.close()
f2.close()
f3.close()
f4.close()

train_data = torch.from_numpy(X_train)
party_A_train_data = train_data[:, :5]
party_B_train_data = train_data[:, 5:]
train_label = torch.from_numpy(y_train)

test_data = torch.from_numpy(X_test)
party_A_test_data = test_data[:, :5]
party_B_test_data = test_data[:, 5:]
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
        self.bottom_layerA = nn.Sequential(nn.Linear(5, 32), nn.ReLU(False))
        self.bottom_layerB = nn.Sequential(nn.Linear(6, 32), nn.ReLU(False))
        self.interactive_layerA = nn.Linear(32, 16)
        self.interactive_layerB = nn.Linear(32, 16)
        self.interactive_activation_layer = nn.ReLU(False)
        self.top_layer = nn.Sequential(nn.Linear(32, 2))

    def forward(self,inputA,inputB):
        PartyA_output = self.bottom_layerA(inputA)
        PartyB_output = self.bottom_layerB(inputB)
        #interactive_activation_input = self.interactive_layerA(PartyA_output)+self.interactive_layerB(PartyB_output)
        interactive_activation_input = torch.cat((self.interactive_layerA(PartyA_output), self.interactive_layerB(PartyB_output)), 1)
        interactive_output = self.interactive_activation_layer(interactive_activation_input)
        output = self.top_layer(interactive_output)
        return PartyA_output, output

class PurchaseDecoderInversion(nn.Module):
    def __init__(self):
        super(PurchaseDecoderInversion, self).__init__()
        self.layers = nn.Sequential(nn.ReLU(False),nn.Linear(32,5))

    def forward(self,encoded_feature):
        decoded_feature = self.layers(encoded_feature)
        return decoded_feature

def train_purchase_decoder(net,decoder,decoder_optimizer,poison_ratio,decoder_epoch_num,device):
    net.train()
    for batch_idx, (A_inputs, B_inputs, targets) in enumerate(train_loader):
        A_inputs, B_inputs, targets = A_inputs.to(device), B_inputs.to(device), targets.to(device)
        feature, outputs = net(A_inputs, B_inputs)
        batch_size = A_inputs.size(0)
        decoder_data_divide = int(batch_size * poison_ratio)
        if batch_idx == 0:
            decoder_train_data = feature[:decoder_data_divide].detach()
            decoder_train_label = A_inputs[:decoder_data_divide]
            decoder_test_data = feature[decoder_data_divide:]
            decoder_test_label = A_inputs[decoder_data_divide:]
        else:
            decoder_train_data = torch.cat([decoder_train_data, feature[:decoder_data_divide].detach()], dim=0)
            decoder_train_label = torch.cat([decoder_train_label, A_inputs[:decoder_data_divide].detach()], dim=0)
            decoder_test_data = torch.cat([decoder_test_data, feature[decoder_data_divide:].detach()], dim=0)
            decoder_test_label = torch.cat([decoder_test_label, A_inputs[decoder_data_divide:].detach()], dim=0)

    decoder_train_dataset = TensorDataset(decoder_train_data, decoder_train_label)
    decoder_train_loader = DataLoader(dataset=decoder_train_dataset, batch_size=decoder_train_data.size(0),shuffle=True)
    decoder_test_dataset = TensorDataset(decoder_test_data, decoder_test_label)
    decoder_test_loader = DataLoader(dataset=decoder_test_dataset, batch_size=decoder_train_data.size(0),shuffle=False)

    train_privacy = 0
    test_privacy = 0
    for decoder_epoch in range(decoder_epoch_num):
        print("train decoder {}/{}".format(decoder_epoch, decoder_epoch_num))
        # train decoder
        decoder_train_loss = 0.0
        for batch_idx, (decoder_inputs, decoder_targets) in enumerate(decoder_train_loader):
            decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(device)
            outputs = decoder(decoder_inputs)
            decoder_optimizer.zero_grad()
            loss = decoder_criterion(outputs, decoder_targets)
            loss.backward()
            decoder_optimizer.step()
            # calculate accuracy
            decoder_train_loss += loss.item() * decoder_inputs.size(0)
        train_privacy = decoder_train_loss / float(len(decoder_train_loader.dataset))
        print('train privacy', train_privacy)

        #test decoder
        decoder_test_loss = 0.0
        data_sum = 0.0
        with torch.no_grad():
            for batch_idx, (decoder_inputs, decoder_targets) in enumerate(decoder_test_loader):
                data_sum += torch.sum(decoder_inputs)
                decoder_inputs, decoder_targets = decoder_inputs.to(device), decoder_targets.to(
                    device)
                outputs = decoder(decoder_inputs)
                loss = decoder_criterion(outputs, decoder_targets)
                decoder_test_loss += loss.item() * decoder_inputs.size(0)
        test_loss = decoder_test_loss / float(len(decoder_test_loader.dataset))
        average = float(data_sum) / float(len(decoder_test_loader.dataset))
        test_privacy = math.sqrt(test_loss / 11) / (average / 11)
        print('test privacy', test_privacy)

    return train_privacy, test_privacy

def train_purchase_vertical_model(net,decoder, optimizer, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (A_inputs, B_inputs, targets) in enumerate(train_loader):
        A_inputs, B_inputs, targets = A_inputs.to(device), B_inputs.to(device), targets.to(device)
        feature, outputs = net(A_inputs, B_inputs)
        optimizer.zero_grad()
        decoder_outputs = decoder(feature)
        acc_loss = criterion(outputs,targets.long())
        #print("acc_loss",acc_loss)
        pri_loss = -decoder_criterion(decoder_outputs,A_inputs)
        #print("pri_loss",pri_loss)
        #loss = acc_loss * pri_loss
        loss = acc_loss + 0.1*pri_loss
        #print("loss",loss)
        loss.backward()
        optimizer.step()

        # calculate accuracy
        train_loss += acc_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_accuracy = correct / float(len(train_loader.dataset))
    # calculate similarity
    train_similarity = pri_loss.item()
    # print("train accuracy {:.2f}%".format(train_accuracy), "train similarity = %f" % train_similarity)
    return train_accuracy, train_similarity

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

