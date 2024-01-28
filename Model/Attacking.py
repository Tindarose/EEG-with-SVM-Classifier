import os
import numpy
import random
from copy import deepcopy
import torch.nn.utils.prune as prune
from .Baseline_Model import *

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class Easy_Dataset(torch.utils.data.Dataset):
    def __init__(self, sign):
        # self.data, self.label = torch.tensor(numpy.load(f"./Data/Random_Tuning_data_{sign}.npy", allow_pickle=True)),\
        #                              torch.tensor(numpy.load(f"./Data/Random_Tuning_label_{sign}.npy", allow_pickle=True))
        self.data, self.label = torch.tensor(numpy.load(f"./Data/Trail_Tuning_data_{sign}.npy", allow_pickle=True)),\
                                    torch.tensor(numpy.load(f"./Data/Trail_Tuning_label_{sign}.npy", allow_pickle=True))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class Special_Dataset(torch.utils.data.Dataset):
    def __init__(self, teacher, student):
        # self.data, self.label = torch.tensor(numpy.load(f"./Data/Random_Tuning_data_{sign}.npy", allow_pickle=True)),\
        #                              torch.tensor(numpy.load(f"./Data/Random_Tuning_label_{sign}.npy", allow_pickle=True))
        if teacher == "CCNN":
            sign1 = "CCNN"
        elif teacher == "EEG_Net" or teacher == "EEG_Conformer" :
            sign1 = "EEG"
        elif teacher == "TSCeption":
            sign1 = "TSC"

        if student == "CCNN":
            sign2 = "CCNN"
        elif student == "EEG_Net" or student == "EEG_Conformer" :
            sign2 = "EEG"
        elif student == "TSCeption":
            sign2 = "TSC"
        self.data, self.label = torch.tensor(numpy.load(f"./Data/Trail_Tuning_data_{sign1}.npy", allow_pickle=True)),\
                                    torch.tensor(numpy.load(f"./Data/Trail_Tuning_data_{sign2}.npy", allow_pickle=True))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def Pruning(model, model_name, pruning_type, amount = 0.2):
    if model_name == "CCNN":
        # parameters_to_prune = ((model.conv1[1], 'weight'), (model.conv2[1], 'weight'),
        #         (model.conv3[1], 'weight'), (model.conv4[1], 'weight'),
        #         (model.lin1[0], 'weight'), (model.lin2, 'weight'))
        parameters_to_prune = ((model.conv1[1], 'weight'), (model.conv2[1], 'weight'),
                (model.conv3[1], 'weight'), (model.conv4[1], 'weight'))

    elif model_name == "EEG_Net":
        # parameters_to_prune = ((model.block1[0], 'weight'), (model.block1[1], 'weight'), 
        #         (model.block1[2], 'weight'), (model.block1[3], 'weight'), (model.block2[0], 'weight'),
        #         (model.block2[1], 'weight'), (model.block2[2], 'weight'), (model.lin, 'weight'))
        parameters_to_prune = ((model.block1[0], 'weight'), (model.block1[1], 'weight'), 
                (model.block1[2], 'weight'), (model.block1[3], 'weight'), (model.block2[0], 'weight'),
                (model.block2[1], 'weight'), (model.block2[2], 'weight'))

    elif model_name == "TSCeption":
        # parameters_to_prune = ((model.Tception1[0], 'weight'), (model.Tception2[0], 'weight'), (model.Tception3[0], 'weight'),
        #         (model.Sception1[0], 'weight'), (model.Sception2[0], 'weight'),
        #         (model.fusion_layer[0], 'weight'),
        #         (model.BN_t, 'weight'), (model.BN_s, 'weight'), (model.BN_fusion, 'weight'),
        #         (model.fc[0], 'weight'), (model.fc[3], 'weight'))
        parameters_to_prune = ((model.Tception1[0], 'weight'), (model.Tception2[0], 'weight'), (model.Tception3[0], 'weight'),
                 (model.Sception1[0], 'weight'), (model.Sception2[0], 'weight'))

    elif model_name == "EEG_Conformer":
        # parameters_to_prune = ((model.layer[0].shallownet[0], 'weight'), (model.layer[0].shallownet[1], 'weight'), 
        # (model.layer[0].shallownet[2], 'weight'), (model.layer[0].projection[0], 'weight'), 

        # (model.layer[1][0][0].fn[0], 'weight'), (model.layer[1][0][0].fn[1].keys, 'weight'), (model.layer[1][0][0].fn[1].queries, 'weight'), 
        # (model.layer[1][0][0].fn[1].values, 'weight'), (model.layer[1][0][0].fn[1].projection, 'weight'), 
        # (model.layer[1][0][1].fn[0], 'weight'), (model.layer[1][0][1].fn[1][0], 'weight'), (model.layer[1][0][1].fn[1][3], 'weight'),

        # (model.layer[1][1][0].fn[0], 'weight'), (model.layer[1][1][0].fn[1].keys, 'weight'), (model.layer[1][1][0].fn[1].queries, 'weight'), 
        # (model.layer[1][1][0].fn[1].values, 'weight'), (model.layer[1][1][0].fn[1].projection, 'weight'), 
        # (model.layer[1][1][1].fn[0], 'weight'), (model.layer[1][1][1].fn[1][0], 'weight'), (model.layer[1][1][1].fn[1][3], 'weight'),

        # (model.layer[1][2][0].fn[0], 'weight'), (model.layer[1][2][0].fn[1].keys, 'weight'), (model.layer[1][2][0].fn[1].queries, 'weight'), 
        # (model.layer[1][2][0].fn[1].values, 'weight'), (model.layer[1][2][0].fn[1].projection, 'weight'), 
        # (model.layer[1][2][1].fn[0], 'weight'), (model.layer[1][2][1].fn[1][0], 'weight'), (model.layer[1][2][1].fn[1][3], 'weight'),

        # (model.layer[1][3][0].fn[0], 'weight'), (model.layer[1][3][0].fn[1].keys, 'weight'), (model.layer[1][3][0].fn[1].queries, 'weight'), 
        # (model.layer[1][3][0].fn[1].values, 'weight'), (model.layer[1][3][0].fn[1].projection, 'weight'), 
        # (model.layer[1][3][1].fn[0], 'weight'), (model.layer[1][3][1].fn[1][0], 'weight'), (model.layer[1][3][1].fn[1][3], 'weight'),

        # (model.layer[1][4][0].fn[0], 'weight'), (model.layer[1][4][0].fn[1].keys, 'weight'), (model.layer[1][4][0].fn[1].queries, 'weight'), 
        # (model.layer[1][4][0].fn[1].values, 'weight'), (model.layer[1][4][0].fn[1].projection, 'weight'), 
        # (model.layer[1][4][1].fn[0], 'weight'), (model.layer[1][4][1].fn[1][0], 'weight'), (model.layer[1][4][1].fn[1][3], 'weight'),

        # (model.layer[1][5][0].fn[0], 'weight'), (model.layer[1][5][0].fn[1].keys, 'weight'), (model.layer[1][5][0].fn[1].queries, 'weight'), 
        # (model.layer[1][5][0].fn[1].values, 'weight'), (model.layer[1][5][0].fn[1].projection, 'weight'), 
        # (model.layer[1][5][1].fn[0], 'weight'), (model.layer[1][5][1].fn[1][0], 'weight'), (model.layer[1][5][1].fn[1][3], 'weight'),

        # (model.layer[2].clshead[1], 'weight'), (model.layer[2].clshead[2], 'weight'), 
        # (model.layer[2].fc[0], 'weight'), (model.layer[2].fc[3], 'weight'),(model.layer[2].fc[6], 'weight'))

        parameters_to_prune = ((model.layer[0].shallownet[0], 'weight'), (model.layer[0].shallownet[1], 'weight'), 
            (model.layer[0].shallownet[2], 'weight'), (model.layer[0].projection[0], 'weight'))

    if pruning_type == "random":
        prune.global_unstructured(parameters_to_prune, pruning_method = prune.RandomUnstructured, amount = amount)
    elif pruning_type == "L1":
        prune.global_unstructured(parameters_to_prune, pruning_method = prune.L1Unstructured, amount = amount)

    return model

def Quantization(model, model_name, quantization_type = None):
    if quantization_type == "Static":
        model.eval()
        if model_name == "CCNN":
            data = Easy_Dataset("CCNN")
        elif model_name == "EEG_Net" or model_name == "EEG_Conformer" :
            data = Easy_Dataset("EEG")
        elif model_name == "TSCeption":
            data = Easy_Dataset("TSC")
        model = torch.quantization.prepare(model)
        for _ in range(10):
            train_loader = torch.utils.data.dataloader.DataLoader(data, batch_size = 512, shuffle = True, num_workers = 0)
            for x, _ in train_loader:
                model(x)
        model = torch.quantization.convert(model)

    elif quantization_type == "Dynamic":# 动态量化
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Sequential, torch.nn.Linear, torch.nn.ReLU, torch.nn.BatchNorm2d}, dtype = torch.qint8)

    elif quantization_type == "Aware":# 感知量化
        if model_name == "CCNN":
            model.conv1[0].add_module("mobilenet", torch.quantization.QuantStub())
        elif model_name == "EEG_Net":
            model.block1[0].add_module("mobilenet", torch.quantization.QuantStub())
        elif model_name == "TSCeption":
            model.Tception1[0].add_module("mobilenet", torch.quantization.QuantStub())
        else:
            model.layer[0].shallownet[0].add_module("mobilenet", torch.quantization.QuantStub())
        model.add_module("dequant", torch.quantization.DeQuantStub())
    return model

def Tuning(model, model_name, epochs = 10):
    if model_name == "CCNN":
        data = Easy_Dataset("CCNN")
    elif model_name == "EEG_Net" or model_name == "EEG_Conformer" :
        data = Easy_Dataset("EEG")
    elif model_name == "TSCeption":
        data = Easy_Dataset("TSC")

    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)

    for _ in range(1, epochs + 1):
        train_loader = torch.utils.data.dataloader.DataLoader(data, batch_size = 512, shuffle = True, num_workers = 0)

        for k in train_loader:
            x, y = k
            optimizer.zero_grad()

            out = model(x.cuda())
            if model_name == "EEG_Conformer":
                out = out[1]
            y = y.long()

            loss = criterion(out, y.cuda())
            loss.requires_grad_(True)

            loss.backward()
            optimizer.step()

    return model

def Model_Extraction(teacher_model, student_model, teacher_model_name, student_model_name, extraction_type = None, epochs = 10):
    student_model.train()
    teacher_model.eval()

    data = Special_Dataset(teacher_model_name, student_model_name)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr = 1e-1)

    for _ in range(epochs):
        data_loader = torch.utils.data.dataloader.DataLoader(data, batch_size = 512, shuffle = True, num_workers = 0)
        for x1, x2 in data_loader:
            x1, x2 = x1.type(torch.FloatTensor).cuda(), x2.type(torch.FloatTensor).cuda()
            teacher_output = teacher_model(x1)
            if teacher_model_name == "EEG_Conformer":
                teacher_output = teacher_output[1]

            if extraction_type == "Adversarial":
                pred = torch.max(teacher_output, 1)[1].data.squeeze().detach()
                x_adv, acc = PGD(student_model, x2, pred)
                output = student_model(x2)
                output_adv = student_model(x_adv)
                if student_model_name == "EEG_Conformer":
                    output, output_adv = output[1], output_adv[1]
                loss = criterion(output, pred) + criterion(output_adv, pred)

            elif extraction_type == "Labels":
                pred = torch.max(teacher_output, 1)[1].data.squeeze().detach()
                output = student_model(x2)
                if student_model_name == "EEG_Conformer":
                    output = output[1]
                loss = criterion(output, pred)

            elif extraction_type == "Probabilities":
                alpha = 0.9
                T = 20
                pred = torch.max(teacher_output, 1)[1].data.squeeze().detach()
                output = student_model(x2)
                if student_model_name == "EEG_Conformer":
                    output = output[1]
                loss = torch.nn.KLDivLoss()(torch.nn.functional.log_softmax(output / T, dim=1), torch.nn.functional.softmax(teacher_output / T, dim=1)) * (alpha * T * T) + torch.nn.functional.cross_entropy(output, pred) * (1. - alpha)
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return student_model

def denormalize(image):
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = numpy.array(image_data)

    img_copy = torch.zeros(image.shape).cuda()
    for i in range(3):
        img_copy[:,i,:,:] = image[:,i,:,:]*image_data[1,i] + image_data[0,i]

    return img_copy

def normalize(image):
    image_data = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    image_data = numpy.array(image_data)
    img_copy = torch.zeros(image.shape).cuda()
    for i in range(3):
        img_copy[:, i, :, :] = (image[:, i, :, :] - image_data[0, i])/image_data[1,i]

    return img_copy

def PGD(model, image, label):
    label = label.cuda()
    loss_func1 = torch.nn.CrossEntropyLoss()
    image_de = denormalize(deepcopy(image))
    image_attack = deepcopy(image)
    image_attack = image_attack.cuda()
    image_attack = torch.autograd.Variable(image_attack, requires_grad=True)
    alpha = 1/256
    epsilon = 4/256

    for iter in range(30):
        image_attack = torch.autograd.Variable(image_attack, requires_grad=True)
        output = model(image_attack)
        loss = -loss_func1(output,label)
        loss.requires_grad_(True)
        loss.backward()
        grad = image_attack.grad.detach().sign()
        image_attack = image_attack.detach()
        image_attack = denormalize(image_attack)
        image_attack -= alpha*grad
        eta = torch.clamp(image_attack-image_de,min=-epsilon,max=epsilon)
        image_attack = torch.clamp(image_de+eta,min=0,max=1)
        image_attack = normalize(image_attack)

    pred_prob = output.detach()
    pred = torch.argmax(pred_prob, dim=-1)
    acc_num = torch.sum(label == pred)
    num = label.shape[0]
    acc = acc_num/num
    acc = acc.data.item()

    return image_attack.detach(), acc