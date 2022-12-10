import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from help_code_demo import ToTensor, DownsampleToTensor, IEGM_DataSET, stats_report
from OptModel import SunNet
import numpy as np
import random
from copy import deepcopy
from thop import profile

def main():
    seed = 222
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = 1
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    # Load Model
    net = SunNet()
    input = torch.randn(1, 1, SIZE//2, 1)
    FLOPs, params = profile(net, inputs=(input,))
    print('Model loaded.')
    print('FLOPs ---', FLOPs)
    print('params ---', params)

    net = net.float().to(device)

    # Load train and test dataset
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([DownsampleToTensor()]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([DownsampleToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    train_num = len(trainset)
    test_num = len(testset)

    print("Training Dataset loading finish.")

    CE = nn.CrossEntropyLoss()
    CE = CE.float().to(device)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-4)
    epoch_num = EPOCH

    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []

    os_best_epoch = 0

    print("Start training \n")
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)

        train_loss = 0.0
        train_correct = 0.0
        net.train()

        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['IEGM_seg'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            batch_num = inputs.shape[0]

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = CE(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum()
            train_loss += loss.item() * batch_num

        scheduler.step()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        train_loss = train_loss / train_num
        train_acc = (train_correct / train_num).item()

        print('[Epoch, Batches, lr] is [%d, %5d, %.5f] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, j + 1, lr ,train_acc, train_loss))

        Train_loss.append(train_loss)
        Train_acc.append(train_acc)

        test_loss = 0.0
        test_correct = 0.0
        net.eval()

        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0

        for i, data_test in enumerate(testloader, 0):
            IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
            IEGM_test = IEGM_test.float().to(device)
            labels_test = labels_test.to(device)
            batch_num = IEGM_test.shape[0]
    
            seg_label = deepcopy(labels_test)

            outputs_test = net(IEGM_test)
            loss = CE(outputs_test, labels_test)

            _, predicted_test = torch.max(outputs_test.data, 1)
            test_correct += (predicted_test == labels_test).sum()
            test_loss += loss.item() * batch_num

            if seg_label == 0:
                segs_FP += (labels_test.size(0) - (predicted_test == labels_test).sum()).item()
                segs_TN += (predicted_test == labels_test).sum().item()
            elif seg_label == 1:
                segs_FN += (labels_test.size(0) - (predicted_test == labels_test).sum()).item()
                segs_TP += (predicted_test == labels_test).sum().item()


        test_acc = (test_correct / test_num).item()
        test_loss = test_loss / test_num

        print('Test Acc: %.5f Test Loss: %.5f' % (test_acc, test_loss))
        _, test_fb = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
        print('---------------------------------------------- \n')

        Test_loss.append(test_loss)
        Test_acc.append(test_acc)

        if epoch == 0:
            max_fb = test_fb
            min_loss = test_loss
            best_epoch = epoch + 1
            torch.save(net, './result/best_acc.pkl')
        else:
            if test_fb > max_fb:
                max_fb = test_fb
                min_loss = test_loss
                best_epoch = epoch + 1
                torch.save(net, './result/best_acc.pkl')
            if test_fb == max_fb:
                if test_loss < min_loss:
                    max_fb = test_fb
                    min_loss = test_loss
                    best_epoch = epoch + 1
                    torch.save(net, './result/best_acc.pkl')

    print('best epoch --- ', best_epoch)

    if os_best_epoch > 0:
        print('os best epoch --- ', os_best_epoch)

    torch.save(net, './result/final.pkl')

    file = open('./result/loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')

    print('Finish training')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-2)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=128)
    argparser.add_argument('--cuda', type=int, default=1)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./data_set/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    print("device is --------------", device)

    main()
