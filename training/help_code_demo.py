import csv, torch, os
import numpy as np

# 总准确率
def ACC(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc

# 正负样本精度
def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then ppv should be 1
    if tp + fn == 0:    # 正样本数量为 0
        ppv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:    # 正样本数量不为 0，模型预测的正样本数量为 0
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv


def NPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then npv should be 1
    if tn + fp == 0:    # 负样本数量为 0
        npv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:    # 负正样本数量不为 0，模型预测的负样本数量为 0
        npv = 0
    else:
        npv = tn / (tn + fn)
    return npv

# 正负样本召回率
def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:    # 正样本数量为 0
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity


def Specificity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then spe should be 1
    if tn + fp == 0:    # 负样本数量为 0
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity

# 正负样本召回率均值
def BAC(mylist):
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc

# 正样本评估函数
def F1(mylist):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# 加入 beta 因子的正样本评估函数
def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return f1


def stats_report(mylist):
    # round 返回 5 位小数
    f1 = round(F1(mylist), 5)     # 正样本评估函数
    fb = round(FB(mylist), 5)     # 加入 beta 因子的正样本评估函数
    se = round(Sensitivity(mylist), 5)    # 正样本召回率
    sp = round(Specificity(mylist), 5)    # 负样本召回率
    bac = round(BAC(mylist), 5)    # 正负样本召回率均值
    acc = round(ACC(mylist), 5)    # 总准确率
    ppv = round(PPV(mylist), 5)    # 正样本精度
    npv = round(NPV(mylist), 5)    # 负样本精度

    output = str(mylist) + '\n' + \
             "F-1 = " + str(f1) + '\n' + \
             "F-B = " + str(fb) + '\n' + \
             "SEN = " + str(se) + '\n' + \
             "SPE = " + str(sp) + '\n' + \
             "BAC = " + str(bac) + '\n' + \
             "ACC = " + str(acc) + '\n' + \
             "PPV = " + str(ppv) + '\n' + \
             "NPV = " + str(npv) + '\n'

    print("F-1 = ", F1(mylist))
    print("F-B = ", FB(mylist))
    print("SEN = ", Sensitivity(mylist))
    print("SPE = ", Specificity(mylist))
    print("BAC = ", BAC(mylist))
    print("ACC = ", ACC(mylist))
    print("PPV = ", PPV(mylist))
    print("NPV = ", NPV(mylist))

    return output, fb    # 打印，返回一个 output 字符串

def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels    # 返回数据集中相同 label 的 filename 集合（字典）


def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat    # 建立 array


class ToTensor(object):
    def __call__(self, sample):
        text = sample
        return torch.from_numpy(text)

class DownsampleToTensor(object):
    def __call__(self, sample):
        text = sample
        text = np.mean(text.reshape((-1,2)), axis=1).reshape((1,-1,1))
        return torch.from_numpy(text)

class IEGM_DataSET():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))    # 数据集中相同 label 的 filename 集合（字典）

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)    # 建立 array
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg, 'label': label}
        if self.transform:
            sample['IEGM_seg'] = self.transform(sample['IEGM_seg'])

        return sample    # 加载数据集，返回 array 与 label 的字典


def pytorch2onnx(net_path, net_name, size):
    net = torch.load(net_path, map_location=torch.device('cpu'))    # 加载模型

    dummy_input = torch.randn(1, 1, size, 1)

    optName = str(net_name)+'.onnx'
    torch.onnx.export(net, dummy_input, optName, verbose=True)
