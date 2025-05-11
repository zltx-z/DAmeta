# DAmeta
we propose DAmeta, a novel drug combination synergy prediction method that uniquely integrates domain adaptive networks with meta-learning to enhance personalized predictions across varied cancer types. To effectively address the challenges posed by diverse cancer types, DAmeta leverages the heterogeneity of driver genes to learn specific parameters for each cell line task through a meta-learning network. Furthermore, it aligns the distributional discrepancies between cell line and patient data using a domain adaptive network, thereby enhancing the accuracy of predictions across different biological conditions.

# Requirements
Python == 3.10

torch == 1.12.1+cu113

torch-geometric == 1.6.1

torch-scatter == 1.6.0

torch-sparse ==  0.6.15

torch-cluster == 1.6.0                             

rdkit ==  2023.9.6 

pytorchtools == 0.0.2

scikit-learn  == 1.4.2 

# Usage (Step by step runing)
# 0. Screening for driver genes
If you want to preprocess gene expression data to predict specific cancer driver genes, Please run the driver gene prediction folder
# 1. Pre-training
If you want to train a generalized model, please run [pre_training.py](pre_training.py)
# 2. training 
If you want to train DAmeta, please run [train_meta_learning.py](train_meta_learning.py)
# 3. test
If you would like to test the power of DAmata on Patient data , please run [test.py](test.py)


