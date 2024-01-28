# 模型量化（Model Quantization）
# 用于制作相关模型-量化版本
import torch
import numpy

class Easy_Dataset(torch.utils.data.Dataset):
    def __init__(self, sign):
        self.data, self.label = torch.tensor(numpy.load(f"./Data/Trail_Tuning_data_{sign}.npy", allow_pickle=True)),\
                                     torch.tensor(numpy.load(f"./Data/Trail_Tuning_label_{sign}.npy", allow_pickle=True))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def quantization(model, model_name, quantization_type = None):
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