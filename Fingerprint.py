import os
import numpy
import torch
import pandas
import random
import time
from torcheeg import transforms
from Model.Baseline_Model import *
from Model.Attacking import *
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import DEAP_CHANNEL_LIST
from torcheeg.datasets.constants.emotion_recognition.deap import DEAP_CHANNEL_LOCATION_DICT
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

def model_select(model_name, process_type, mode, student = None):
    if model_name == "CCNN":
        model = CCNN()
    elif model_name == "EEG_Net":
        model = EEG_Net()
    elif model_name == "TSCeption":
        model = TSCeption()
    else:
        model = EEG_Conformer()

    if student == None:
        model.load_state_dict(torch.load(f"./Model/Pretrain/{model_name}_fold_trail.pth"))
    
    if process_type == "Purning":
        model = Pruning(model, model_name, pruning_type = mode, amount = 0.2)

    elif process_type == "Tuning":
        model = Tuning(model, model_name, epochs = 10)

    elif process_type == "Quantization":
        model = Quantization(model, model_name, quantization_type = mode)

    return model

def data_select(model_name, index):
    if model_name == "CCNN":
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

    elif model_name == "EEG_Net" or model_name == "EEG_Conformer":
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

    data, label = None, []
    k = 0
    if STEP == 0:
        index = index[7431:7931]
        for i in index:
            if k == LEN:
                break
            if data == None:
                data = torch.unsqueeze(dataset[i][0], dim = 0)
            else:
                data = torch.cat((data, torch.unsqueeze(dataset[i][0], dim = 0)))
            k += 1
    else:
        for i in index[::STEP]:
            if k == LEN:
                break
            if data == None:
                data = torch.unsqueeze(dataset[i][0], dim = 0)
            else:
                data = torch.cat((data, torch.unsqueeze(dataset[i][0], dim = 0)))
            k += 1

    return torch.tensor(data).cuda()

def similarity(data_1, data_2, mode):
    arr_1, arr_2 = [], []
    for i in range(data_1.shape[0]):
        if data_1[i][0] > data_1[i][1]:
            arr_1.append(0.)
        else:
            arr_1.append(1.)
        if data_2[i][0] > data_2[i][1]:
            arr_2.append(0.)
        else:
            arr_2.append(1.)
    arr_1, arr_2 = torch.tensor(arr_1), torch.tensor(arr_2)
    return float(torch.dist(arr_1, arr_2)) if mode == "Euclidean" else torch.cosine_similarity(arr_1.unsqueeze(0), arr_2.unsqueeze(0))

arr_data = ["CCNN", "EEG", "TSC"]
LEN = 200
arr_attack = [("Baseline", None), ("Purning", "L1"), ("Purning", "random"), ("Quantization", "Static"), ("Quantization", "Dynamic"),\
     ("Tuning", None), ("Model_Extraction", "Labels"), ("Model_Extraction", "Probabilities")]
arr_model = ["CCNN", "EEG_Net", "TSCeption", "EEG_Conformer"]

for a in range(3):
    SIGN = arr_data[a]
    for b in range(3):
        STEP = 1
        step_type = "close"
        if b == 2:
            STEP = -1
            step_type = "far"
        elif b == 1:
            STEP = 0
            step_type = "midden"
        for e in range(1, 11):
            if a < 2 or (b < 2 or (b == 2 and e < 9)):
                continue
            with torch.no_grad():
                start = time.time()
                print(f"{SIGN}  {step_type} {e}")
                index = list(pandas.read_csv(f"./Data/SVM_{SIGN}_index.csv")['index'])
                ar1, ar2, ar3, ar4 = [], [], [], []
                for attack in arr_attack:
                    process_type, mode = attack[0], attack[1]
                    if process_type != "Model_Extraction":
                        for model_name in arr_model:
                            model = model_select(model_name, process_type, mode).cuda()
                            x = data_select(model_name, index)

                            if mode == "Dynamic":
                                model = model.cpu()
                                x = x.cpu()

                            out = model(x)
                            if model_name == "EEG_Conformer":
                                out = out[1]
                            numpy.save(f"./Data/Fingerprint/{model_name}_{process_type}_{mode}_Fingerprint.npy", out.detach().cpu().numpy())

                        for k in range(4):
                            model_1 = arr_model[k]
                            data_1 = torch.tensor(numpy.load(f"./Data/FingerPrint/{model_1}_{process_type}_{mode}_Fingerprint.npy", allow_pickle=True))
                            for j in range(4):
                                model_2 = arr_model[j]
                                data_2 = torch.tensor(numpy.load(f"./Data/FingerPrint/{model_2}_Baseline_None_Fingerprint.npy", allow_pickle=True))
                                sim = similarity(data_1, data_2, "Euclidean")
                                if j == 0:
                                    ar1.append(sim)
                                elif j == 1:
                                    ar2.append(sim)
                                elif j == 2:
                                    ar3.append(sim)
                                elif j == 3:
                                    ar4.append(sim)
                    elif process_type == "Model_Extraction":
                        for teacher_model_name in arr_model:
                            teacher_model = model_select(teacher_model_name, process_type, mode).cuda()
                            for student_model_name in arr_model:
                                if student_model_name != teacher_model_name:
                                    continue 
                                student_model = model_select(student_model_name, process_type, mode, student = True).cuda()
                                model = Model_Extraction(teacher_model, student_model, teacher_model_name, student_model_name, mode)
                                x = data_select(student_model_name, index)
                                out = model(x)
                                if student_model_name == "EEG_Conformer":
                                    out = out[1]
                                numpy.save(f"./Model/Extraction/{teacher_model_name}_{student_model_name}_{SIGN}_{step_type}_{mode}_{e}.npy", out.detach().cpu().numpy())
                                data_2 = torch.tensor(numpy.load(f"./Data/FingerPrint/{teacher_model_name}_Baseline_None_Fingerprint.npy", allow_pickle=True))
                                sim = similarity(out, data_2, "Euclidean")
                                if teacher_model_name == "CCNN":
                                    ar1.append(sim)
                                elif teacher_model_name == "EEG_Net":
                                    ar2.append(sim)
                                elif teacher_model_name == "TSCeption":
                                    ar3.append(sim)
                                else:
                                    ar4.append(sim)
                    else:   continue
                now = time.time() - start
                print(f"用时{int(now // 60)}分钟{int(now % 60)}秒")
                print(f"还需要{int(now * (90 - 30 * a - 10 * b - e) // 3600)}小时，{int(now * (90 - 30 * a - 10 * b - e) % 3600 // 60)}分钟")
                pandas.DataFrame({'CCNN':ar1, 'net':ar2, 'tsc': ar3, 'conformer': ar4}).to_csv(f"./log/{e}_{SIGN}_{step_type}.csv", index = False)