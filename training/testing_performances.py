import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import torchvision.transforms as transforms

from help_code_demo import ToTensor,DownsampleToTensor, IEGM_DataSET, stats_report


def main():
    seed = 222
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Hyperparameters
    m = 'test'
    BATCH_SIZE_TEST = 1
    SIZE = args.size
    path_data = args.path_data
    path_records = args.path_record
    path_indices = args.path_indices
    stats_file = open(path_records + m + '_seg_stat.txt', 'w')

    # load trained network
    net = torch.load('./result/best_acc.pkl', map_location='cuda:0')
    net.cuda()
    device = torch.device('cuda:0')

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode=m,
                           size=SIZE,
                           transform=transforms.Compose([DownsampleToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0

    for data_test in testloader:
        IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
        seg_label = deepcopy(labels_test)

        IEGM_test = IEGM_test.float().to(device)
        labels_test = labels_test.to(device)

        outputs_test = net(IEGM_test)
        _, predicted_test = torch.max(outputs_test.data, 1)

        if seg_label == 0:
            segs_FP += (labels_test.size(0) - (predicted_test == labels_test).sum()).item()
            segs_TN += (predicted_test == labels_test).sum().item()
        elif seg_label == 1:
            segs_FN += (labels_test.size(0) - (predicted_test == labels_test).sum()).item()
            segs_TP += (predicted_test == labels_test).sum().item()

    # report metrics
    stats_file.write('segments: TP, FN, FP, TN\n')
    output_segs, _ = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
    stats_file.write(output_segs + '\n')

    del net


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./data_set/')
    argparser.add_argument('--path_record', type=str, default='./result/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))
    print("device is --------------", device)

    main()
