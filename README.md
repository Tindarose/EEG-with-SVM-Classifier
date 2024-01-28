# EEG-with-SVM-Classifier

## 2024年1月提交论文《EEG Model Protection via Fingerprint Close To Optimal Classification Hyperplane》代码留档

运行步骤如下

- 将DEAP数据集下载到 \Data\data_preprocessed_python 中，格式为DAT
- 运行 Pretrain.py 和 test_data.py ，生成预处理过的数据与模型
- 运行 \Data\SVM\SVM_Kernel.ipynb ，生成测试数据集
- 运行 Fingerprint.py ,得到指纹
- AUC.py，输出excel记录的数据