# 模型微调（Model Tuning）
# 用于制作相关模型-微调版本
from .Baseline_Model import *

def FATL(model, name):
    for n, param in model.named_parameters():
        if n in name or len(param.shape) < 2:
            continue
        torch.nn.init.xavier_uniform_(param)
        param.requires_grad = False
    return model

def tuning(model, model_name, tuning_type):
    if model_name == "CCNN":
        if tuning_type == "FTLL":
            # Fine-Tune Last Layer 更新最后一个分类层的参数
            torch.nn.init.xavier_uniform_(model.lin2.weight)
        elif tuning_type == "FTAL":
            # Fine-Tune All Layers 更新模型的所有图层
            # model = FATL(model, ["lin2.weight"])
            torch.nn.init.xavier_uniform_(model.conv1[1].weight)
            torch.nn.init.xavier_uniform_(model.conv2[1].weight)
            torch.nn.init.xavier_uniform_(model.conv3[1].weight)
            torch.nn.init.xavier_uniform_(model.conv4[1].weight)
            torch.nn.init.xavier_uniform_(model.lin1[0].weight)
            torch.nn.init.xavier_uniform_(model.lin2.weight)

    elif model_name == "EEG_Net":
        if tuning_type == "FTLL":
            # Fine-Tune Last Layer 更新最后一个分类层的参数
            torch.nn.init.xavier_uniform_(model.lin.weight)
        elif tuning_type == "FTAL":
            # model = FATL(model, ["lin.weight"])
            torch.nn.init.xavier_uniform_(model.block1[0].weight)
            torch.nn.init.xavier_uniform_(model.block1[2].weight)
            torch.nn.init.xavier_uniform_(model.block2[0].weight)
            torch.nn.init.xavier_uniform_(model.block2[1].weight)
            torch.nn.init.xavier_uniform_(model.lin.weight)

    elif model_name == "TSCeption":
        if tuning_type == "FTLL":
            # Fine-Tune Last Layer 更新最后一个分类层的参数
            torch.nn.init.xavier_uniform_(model.fc[3].weight)
        elif tuning_type == "FTAL":
            # model = FATL(model, ["model.fc.3.weight"])
            torch.nn.init.xavier_uniform_(model.Tception1[0].weight)
            torch.nn.init.xavier_uniform_(model.Tception2[0].weight)
            torch.nn.init.xavier_uniform_(model.Tception3[0].weight)
            torch.nn.init.xavier_uniform_(model.Sception1[0].weight)
            torch.nn.init.xavier_uniform_(model.Sception2[0].weight)
            torch.nn.init.xavier_uniform_(model.fusion_layer[0].weight)
            torch.nn.init.xavier_uniform_(model.fc[0].weight)
            torch.nn.init.xavier_uniform_(model.fc[3].weight)

    elif model_name == "EEG_Conformer":
        if tuning_type == "FTLL":
            # Fine-Tune Last Layer 更新最后一个分类层的参数
            torch.nn.init.xavier_uniform_(model.layer[2].fc[6].weight)
        elif tuning_type == "FTAL":
            # model = FATL(model, ["model.layer.2.fc.6.weight"])
            torch.nn.init.xavier_uniform_(model.layer[0].shallownet[0].weight)
            torch.nn.init.xavier_uniform_(model.layer[0].shallownet[1].weight)
            torch.nn.init.xavier_uniform_(model.layer[0].projection[0].weight)
            torch.nn.init.xavier_uniform_(model.layer[2].fc[0].weight)
            torch.nn.init.xavier_uniform_(model.layer[2].fc[3].weight)
            torch.nn.init.xavier_uniform_(model.layer[2].fc[6].weight)

    # if tuning_type == "FTLL":
    #     # Fine-Tune Last Layer 更新最后一个分类层的参数
    #     torch.nn.init.xavier_uniform_(model.fc.weight)
    # elif tuning_type == "FTAL":
    #     # Fine-Tune All Layers 更新模型的所有图层
    #     torch.nn.init.xavier_uniform_(model.fc.weight)
    # elif tuning_type == "RTLL":
    #     # Re-Train Last Layer 初始化输出层的参数，只更新它们
    #     raise ValueError("模型不训练，不使用该微调方式")
    # else:
    #     # Re-Train All Layers 初始化输出层的参数，并更新网络所有层的参数
    #     raise ValueError("模型不训练，不使用该微调方式")

    return model