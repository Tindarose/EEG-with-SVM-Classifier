# EEG-with-SVM-Classifier
## 代码留档

运行步骤如下

- 将DEAP数据集下载到 \Data\data_preprocessed_python 中，格式为DAT
- 运行 Pretrain.py 和 test_data.py ，生成预处理过的数据与模型
- 运行 \Data\SVM\SVM_Kernel.ipynb ，生成测试数据集
- 运行 Fingerprint.py ,得到指纹
- 运行AUC.py，输出excel记录的数据

- The operating steps are as follows
- Download the DEAP dataset to \Data\data_preprocessed_Python in DAT format
- Run Pretrain. py and test_data. py to generate preprocessed data and models
- Run \Data\SVM\SVM_Kernel. ipynb to generate a test dataset
- Run Fingerprint.py to obtain fingerprints
- Run AUC.py, output data recorded in Excel
