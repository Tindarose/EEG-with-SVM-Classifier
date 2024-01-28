import os
import time
import numpy
import torch
import pandas
import random
import datetime
import torchmetrics
from loguru import logger
from Model.Baseline_Model import *
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import DEAP_CHANNEL_LIST
from torcheeg.datasets.constants.emotion_recognition.deap import DEAP_CHANNEL_LOCATION_DICT

BATCH_SIZE, EPOCHS, LEARNING_RATE, KFOLD = 512, 50, 1e-1, 5

class Easy_Dataset(torch.utils.data.Dataset):
    # 提取训练集
    def __init__(self, data, sign, fold, key):
        # if sign == 1:
        #     arr = list(pandas.read_csv("./Data/Trail_index.csv")['Train_1'])
        # else:
        #     arr = list(pandas.read_csv("./Data/Trail_index.csv")['Train_2'])
        arr = list(pandas.read_csv("./Data/Random_index.csv")['Train'])
        self.data, self.label = None, []

        j = 0
        temp = None
        if key == "train":
            if fold == 0:
                arr = arr[int(len(arr) * 0.2):]
            elif fold == 4:
                arr = arr[:int(len(arr) * 0.8)]
            else:
                arr = arr[:int(len(arr) * 0.2 * fold)] + arr[int(len(arr) * 0.2 * (1 + fold)):]
        else:
            arr = arr[int(len(arr) * 0.2 * fold): int(len(arr) * 0.2 * (1 + fold))]

        for i in arr:
            j += 1
            self.label.append(float(data[i][1]))

            if temp == None:
                temp = torch.unsqueeze(data[i][0], dim = 0)
            else:
                temp = torch.cat((temp, torch.unsqueeze(data[i][0], dim = 0)))

            if j == 3000:
                if self.data == None:
                    self.data = temp
                else:
                    self.data = torch.cat((self.data, temp))
                j = 0
                temp = None

        self.data, self.label = torch.cat((self.data, temp)), torch.tensor(self.label) # 防止有剩余数据
        print("数据提取完成")

        # for i in range(3):
        #     self.label = torch.unsqueeze(self.label, 0)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def data_choose(model_name, fold):
    sign = None
    if model_name == 'CCNN':
        sign = 1
        dataset = DEAPDataset(io_path = f'./Data/deap_CCNN', root_path = './Data/data_preprocessed_python',
                            io_mode = 'pickle',
                            offline_transform = transforms.Compose([
                                transforms.BandDifferentialEntropy(sampling_rate = 128, apply_to_baseline = True),
                                transforms.BaselineRemoval(),
                                transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                            ]),
                            online_transform = transforms.ToTensor(),
                            label_transform = transforms.Compose([
                                transforms.Select('valence'),
                                transforms.Binary(5.0),
                            ]),
                            chunk_size = 128,
                            baseline_chunk_size = 128,
                            num_baseline = 3,
                            num_worker = 4)
    elif model_name == "EEG_Net" or model_name == "EEG_Conformer":
        dataset = DEAPDataset(io_path=f'./Data/deap_EEG', root_path='./Data/data_preprocessed_python',
            io_mode='pickle',
            online_transform=transforms.Compose([
                transforms.To2d(),
                transforms.ToTensor(),
            ]),
            label_transform=transforms.Compose([
                transforms.Select('valence'),
                transforms.Binary(5.0),
            ]))
    elif model_name == "TSCeption":
        dataset = DEAPDataset(io_path='./Data/deap_TSCeption', root_path='./Data/data_preprocessed_python',
                            io_mode='pickle',
                            offline_transform=transforms.Compose([
                                transforms.PickElectrode(transforms.PickElectrode.to_index_list(
                                ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
                                'PO3','O1', 'FP2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2',
                                'P4', 'P8', 'PO4', 'O2'], DEAP_CHANNEL_LIST)),
                                transforms.To2d()
                            ]),
                            online_transform=transforms.ToTensor(),
                            label_transform=transforms.Compose([
                                transforms.Select('valence'),
                                transforms.Binary(5.0),
                            ]))

    train_dataset = Easy_Dataset(dataset, sign, fold, "train")
    val_dataset = Easy_Dataset(dataset, sign, fold, "val")
    return train_dataset, val_dataset

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

arr = ["CCNN", "EEG_Net", "TSCeption", "EEG_Conformer"]

for a in range(4):
    model_name = arr[a]
    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    logger.add(f"./Model/Pretrain/log/{model_name}_{now}.log", encoding = "utf-8")
    logger.info(f"Batch_Size：{BATCH_SIZE}，Epoch：{EPOCHS}，Learning_Rate：{LEARNING_RATE}，K-Fold：{KFOLD}")
    logger.info("Optimizer：SGD，Scheduler：None，Criterion：Cross-Entropy")
    max_acc = 0
    for fold in range(KFOLD):
        if model_name == "CCNN":
            model = CCNN().cuda()
        elif model_name == "EEG_Net":
            model = EEG_Net().cuda()
        elif model_name == "TSCeption":
            model = TSCeption().cuda()
        elif model_name == "EEG_Conformer":
            model = EEG_Conformer().cuda()

        train_dataset, val_dataset = data_choose(model_name, fold)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, verbose = True)
        train_accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = 2).cuda()
        test_accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes = 2).cuda()

        print('================== 轮次%d ==================' % fold)
        train_loader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
        val_loader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
        for ep in range(1, EPOCHS + 1):
            model.train()
            train_accuracy.reset()
            run_loss = 0

            logger.info(f"轮次{fold + 1}   Epoch{ep}")

            start = time.time()

            for data in train_loader:
                x_train, y_train = data
                optimizer.zero_grad()

                out = model(x_train.cuda())
                if model_name == "EEG_Conformer":
                    out = out[1]

                loss = criterion(out, y_train.long().cuda())
                loss.requires_grad_(True)
                train_accuracy.update(out, y_train.cuda())
                run_loss += loss.item()
                loss.backward()

                optimizer.step()
            time1 = time.time() - start
            train_loss = run_loss / len(train_loader)
            train_acc = 100 * train_accuracy.compute()

            logger.info(f"训练损失：{train_loss}，准确率{train_acc:>0.3f}%, 用时{time1:>0.1f}s")
            scheduler.step()
        for data in val_loader:
            x, y = data

            out = model(x.cuda())
            if model_name == "EEG_Conformer":
                out = out[1]

            test_accuracy.update(out, y.cuda())
        test_acc = 100 * test_accuracy.compute()
        logger.info(f"测试准确率{test_acc:>0.3f}%")

        if (max_acc != 0 and max_acc < test_acc) or max_acc == 0:
            torch.save(model.state_dict(), f'./Model/Pretrain/{model_name}_fold_random.pth')
            max_acc = test_acc