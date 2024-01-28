import os
import pandas
import random
import numpy
import torch
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import DEAP_CHANNEL_LIST
from torcheeg.datasets.constants.emotion_recognition.deap import DEAP_CHANNEL_LOCATION_DICT

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# 按照trail分类
arr = ["CCNN", "EEG", "TSC"]
for k in range(3):
    index = list(pandas.read_csv("./Data/Random_index.csv")['Tuning'])
    if k == 0:
        dataset = DEAPDataset(io_path = f'./Data/deap_CCNN',
                root_path = './Data/data_preprocessed_python',
                io_mode = 'pickle',
                offline_transform = transforms.Compose([
                    transforms.BandDifferentialEntropy(
                        sampling_rate = 128, apply_to_baseline = True),
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
    elif k == 1:
        dataset = DEAPDataset(io_path=f'./Data/deap_EEG',
                    root_path='./Data/data_preprocessed_python',
                    online_transform=transforms.Compose([
                        transforms.To2d(),
                        transforms.ToTensor(),
                    ]),
                    io_mode='pickle',
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
    else:
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
    j = 0
    data, temp, label = None, None, []

    for i in index:
        j += 1
        label.append(float(dataset[i][1]))

        if temp == None:
            temp = torch.unsqueeze(dataset[i][0].cuda(), dim = 0).cuda()
        else:
            temp = torch.cat((temp, torch.unsqueeze(dataset[i][0].cuda(), dim = 0))).cuda()

        if j == 3000:
            if data == None:
                data = temp
            else:
                data = torch.cat((data, temp)).cuda()
            j = 0
            temp = None

    data, label = torch.cat((data, temp)).cuda(), torch.tensor(label).cuda()

    name = arr[k]

    numpy.save(f"./Data/Random_Tuning_data_{name}.npy", data.detach().cpu().numpy())
    numpy.save(f"./Data/Random_Tuning_label_{name}.npy", label.detach().cpu().numpy())
