from torch import optim, nn
import os
import time
import sys

from output_log import Logger
from accuracy_privacy_adult.adult_vertical_model import PurchaseVerticalModel, PurchaseDecoderInversion, \
    train_purchase_decoder, test_purchase_normal_model,train_purchase_vertical_model
from draw_while_running import draw_while_running
from linear_weight_pruning import apply_weight_pruning_linear

file_name = 'adult'
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
outputSavePath = './test' + file_name + '_' + timestamp
if not os.path.exists(outputSavePath):
    os.mkdir(outputSavePath)
logSavePath = outputSavePath + '/log'
if not os.path.exists(logSavePath):
    os.mkdir(logSavePath)
sys.stdout = Logger(os.path.join(logSavePath, "output.txt"), sys.stdout)
sys.stderr = Logger(os.path.join(logSavePath, "error.txt"), sys.stderr)
rewardSavePath = outputSavePath + '/saveReward'
if not os.path.exists(rewardSavePath):
    os.mkdir(rewardSavePath)
results_name = 'results_log.txt'
# privacy_name = 'privacy_log.txt'
accuracy_file = open(os.path.join(rewardSavePath, results_name), 'w')
# privacy_file = open(os.path.join(rewardSavePath, privacy_name), 'w')

model = PurchaseVerticalModel().to('cuda')
#inversion_decoder = PurchaseDecoderInversion().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#decoder_optimizer = optim.Adam(inversion_decoder.parameters(), lr=0.0001)
device = 'cuda'
#inversion_decoder = PurchaseDecoderInversion().to(device)
#decoder_optimizer = optim.Adam(inversion_decoder.parameters(), lr=0.0001)

num_of_epochs = 20
poison_ratio = 0.3
#perc = [99.2,99.2,10,10]
for epoch in range(num_of_epochs):
    print("Epoch {}".format(epoch))
    # train
    inversion_decoder = PurchaseDecoderInversion().to(device)
    decoder_optimizer = optim.Adam(inversion_decoder.parameters(), lr=0.0001)
    train_privacy, test_privacy = train_purchase_decoder(model,inversion_decoder,decoder_optimizer,poison_ratio,100,device)
    '''
    train_accuracy, train_privacy, test_privacy = train_purchase_vertical_inversion_decoder(model, inversion_decoder,
                                                                                            optimizer,
                                                                                            decoder_optimizer,
                                                                                            poison_ratio, 10, device)
    '''
    train_accuracy , train_similarity = train_purchase_vertical_model(model,inversion_decoder,optimizer,device)
    # test
    test_accuracy = test_purchase_normal_model(model, device)
    # write into file
    # save log
    result_file = open(os.path.join(rewardSavePath, results_name), 'a')

    result_file.write(
        str(epoch) + ' ' + str(train_privacy) + ' ' + str(test_privacy) + ' ' + str(train_accuracy) + ' '
        + str(train_similarity) + ' ' + str(test_accuracy) +'\n')
    result_file.close()
    '''
    # ====== apply weight pruning to linear layer ====
    mask_linear = apply_weight_pruning_linear(model.bottom_layerA[0], perc[int(epoch / 10)])
    model.bottom_layerA[0] = mask_linear
    print('apply pruning, percentage {}'.format(perc[int(epoch / 10)]))
    print(model.bottom_layerA[0])
    print(model.bottom_layerA[0].weight)
    #logging.info('apply pruning, percentage {}'.format(perc[int(epoch / 10)]))
    #logging.info(vertical_model.bottom_layerA[0].weight)
    '''
    #
    # # draw
    if epoch > 0:
        draw_while_running(rewardSavePath, results_name, rewardSavePath, str(epoch) + '_results.svg',
                           'train_vertical_model',
                           'epoch',
                           'results', ['epoch', 'train_privacy', 'test_privacy','train_accuracy', 'train_similarity','test_accuracy'])