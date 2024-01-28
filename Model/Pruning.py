# 模型剪枝（Model Pruning）
# 用于制作相关模型-剪枝版本
import torch.nn.utils.prune as prune
from .Baseline_Model import *

import os
import numpy
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

def pruning(model, model_name, pruning_type, amount = 0.2):
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