# 模型保护-微调
import os
import numpy
import torch
import random

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
        self.data, self.label = torch.tensor(numpy.load(f"./Data/Trail_Tuning_data_{sign}.npy", allow_pickle=True)),\
                                     torch.tensor(numpy.load(f"./Data/Trail_Tuning_label_{sign}.npy", allow_pickle=True))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def tuning(model, model_name, epochs = 15):
    if model_name == "CCNN":
        data = Easy_Dataset("CCNN")
    elif model_name == "EEG_Net" or model_name == "EEG_Conformer" :
        data = Easy_Dataset("EEG")
    elif model_name == "TSCeption":
        data = Easy_Dataset("TSC")

    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

    for i in range(1, epochs + 1):
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