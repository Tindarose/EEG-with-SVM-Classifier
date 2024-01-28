import math
import numpy
import pandas
import sklearn.metrics

def norm(list_a, list_b, mode):
    if mode % 3 == 1:
        # min-max规范化
        min_num1, max_num1 = min(list_a), max(list_a)
        min_num2, max_num2 = min(list_b), max(list_b)
        for i in range(len(list_a)):
            list_a[i] = (list_a[i] - min(min_num1, min_num2)) / (max(max_num1, max_num2) - min(min_num1, min_num2))
        for i in range(len(list_b)):
            list_b[i] = (list_b[i] - min(min_num1, min_num2)) / (max(max_num1, max_num2) - min(min_num1, min_num2))

    elif mode % 3 == 2:
        # Z-score归一化
        mean, var = numpy.mean(list_a + list_b), numpy.var(list_a + list_b)
        for i in range(len(list_a)):
            list_a[i] = (list_a[i] - mean) / var
        for i in range(len(list_b)):
            list_b[i] = (list_b[i] - mean) / var

    else:
        # Sigmoid
        for i in range(len(list_a)):
            list_a[i] = 1 / (1 + math.exp(-list_a[i]))
        for i in range(len(list_b)):
            list_b[i] = 1 / (1 + math.exp(-list_b[i]))

    return list_a, list_b

def AUC(mode, list_a, list_b):
    y_true, y_score = [], []
    list_a, list_b = norm(list_a, list_b, mode)
    for _ in range(len(list_a)):
        y_true.append(0)
    for _ in range(len(list_b)):
        y_true.append(1)
    y_score.extend(list_a)
    y_score.extend(list_b)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
    return sklearn.metrics.auc(fpr, tpr)

lenth = 250
mode = int(input("mode:"))

arr_sign, arr_step = ["CCNN", "EEG", "TSC"], ["close", "midden", "far"]
for a in arr_sign:
    for b in arr_step:
        ar1, ar2, ar3, ar4 = [], [], [], []
        no_related_CCNN, no_related_Net, no_related_TSC, no_related_Conformer = [], [], [], []
        related_CCNN, related_Net, related_TSC, related_Conformer = [], [], [], []
        purning_L1_CCNN, purning_L1_Net, purning_L1_TSC, purning_L1_Conformer = [], [], [], []
        purning_Random_CCNN, purning_Random_Net, purning_Random_TSC, purning_Random_Conformer = [], [], [], []
        quantization_Static_CCNN, quantization_Static_Net, quantization_Static_TSC, quantization_Static_Conformer = [], [], [], []
        quantization_Dynamic_CCNN, quantization_Dynamic_Net, quantization_Dynamic_TSC, quantization_Dynamic_Conformer = [], [], [], []
        tuning_CCNN, tuning_Net, tuning_TSC, tuning_Conformer = [], [], [], []
        Labels_CCNN, Labels_Net, Labels_TSC, Labels_Conformer = [], [], [], []
        Probabilities_CCNN, Probabilities_Net, Probabilities_TSC, Probabilities_Conformer = [], [], [], []

        for e in range(1, 11):
            data = pandas.read_csv(f"./log/多项式核_{lenth}_trail/{e}_{a}_{b}.csv")
            related_CCNN.append(data["CCNN"][0])
            if mode < 4:
                ori = 4
            else:
                ori = 24
            for m in range(ori):
                if m % 4 != 0:
                    no_related_CCNN.append(data["CCNN"][m])
                if m % 4 != 1:
                    no_related_Net.append(data["net"][m])
                if m % 4 != 2:
                    no_related_TSC.append(data["tsc"][m])
                if m % 4 != 3:
                    no_related_Conformer.append(data["conformer"][m])
            related_Net.append(data["net"][1])
            related_TSC.append(data["tsc"][2])
            related_Conformer.append(data["conformer"][3])

            purning_L1_CCNN.append(data["CCNN"][4])
            purning_L1_Net.append(data["net"][5])
            purning_L1_TSC.append(data["tsc"][6])
            purning_L1_Conformer.append(data["conformer"][7])

            purning_Random_CCNN.append(data["CCNN"][8])
            purning_Random_Net.append(data["net"][9])
            purning_Random_TSC.append(data["tsc"][10])
            purning_Random_Conformer.append(data["conformer"][11])

            quantization_Static_CCNN.append(data["CCNN"][12])
            quantization_Static_Net.append(data["net"][13])
            quantization_Static_TSC.append(data["tsc"][14])
            quantization_Static_Conformer.append(data["conformer"][15])

            quantization_Dynamic_CCNN.append(data["CCNN"][16])
            quantization_Dynamic_Net.append(data["net"][17])
            quantization_Dynamic_TSC.append(data["tsc"][18])
            quantization_Dynamic_Conformer.append(data["conformer"][19])

            tuning_CCNN.append(data["CCNN"][20])
            tuning_Net.append(data["net"][21])
            tuning_TSC.append(data["tsc"][22])
            tuning_Conformer.append(data["conformer"][23])

            Labels_CCNN.append(data["CCNN"][24])
            Labels_Net.append(data["net"][24])
            Labels_TSC.append(data["tsc"][24])
            Labels_Conformer.append(data["conformer"][24])

            Probabilities_CCNN.append(data["CCNN"][25])
            Probabilities_Net.append(data["net"][25])
            Probabilities_TSC.append(data["tsc"][25])
            Probabilities_Conformer.append(data["conformer"][25])

        ar1.append(AUC(mode, no_related_CCNN, related_CCNN))
        ar2.append(AUC(mode, no_related_Net, related_Net))
        ar3.append(AUC(mode, no_related_TSC, related_TSC))
        ar4.append(AUC(mode, no_related_Conformer, related_Conformer))

        ar1.append(AUC(mode, no_related_CCNN, purning_L1_CCNN))
        ar2.append(AUC(mode, no_related_Net, purning_L1_Net))
        ar3.append(AUC(mode, no_related_TSC, purning_L1_TSC))
        ar4.append(AUC(mode, no_related_Conformer, purning_L1_Conformer))

        ar1.append(AUC(mode, no_related_CCNN, purning_Random_CCNN))
        ar2.append(AUC(mode, no_related_Net, purning_Random_Net))
        ar3.append(AUC(mode, no_related_TSC, purning_Random_TSC))
        ar4.append(AUC(mode, no_related_Conformer, purning_Random_Conformer))

        ar1.append(AUC(mode, no_related_CCNN, quantization_Static_CCNN))
        ar2.append(AUC(mode, no_related_Net, quantization_Static_Net))
        ar3.append(AUC(mode, no_related_TSC, quantization_Static_TSC))
        ar4.append(AUC(mode, no_related_Conformer, quantization_Static_Conformer))

        ar1.append(AUC(mode, no_related_CCNN, quantization_Dynamic_CCNN))
        ar2.append(AUC(mode, no_related_Net, quantization_Dynamic_Net))
        ar3.append(AUC(mode, no_related_TSC, quantization_Dynamic_TSC))
        ar4.append(AUC(mode, no_related_Conformer, quantization_Dynamic_Conformer))

        ar1.append(AUC(mode, no_related_CCNN, tuning_CCNN))
        ar2.append(AUC(mode, no_related_Net, tuning_Net))
        ar3.append(AUC(mode, no_related_TSC, tuning_TSC))
        ar4.append(AUC(mode, no_related_Conformer, tuning_Conformer))

        ar1.append(AUC(mode, no_related_CCNN, Labels_CCNN))
        ar2.append(AUC(mode, no_related_Net, Labels_Net))
        ar3.append(AUC(mode, no_related_TSC, Labels_TSC))
        ar4.append(AUC(mode, no_related_Conformer, Labels_Conformer))

        ar1.append(AUC(mode, no_related_CCNN, Probabilities_CCNN))
        ar2.append(AUC(mode, no_related_Net, Probabilities_Net))
        ar3.append(AUC(mode, no_related_TSC, Probabilities_TSC))
        ar4.append(AUC(mode, no_related_Conformer, Probabilities_Conformer))

        pandas.DataFrame({'CCNN':ar1, 'net':ar2, 'tsc': ar3, 'conformer': ar4}).to_csv(f"./log/trail_{a}_{b}.csv", index = False)