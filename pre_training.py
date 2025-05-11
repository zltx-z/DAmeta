from meta_learning1.model.model import SynergyNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, cohen_kappa_score
import pickle
import random
from torch.optim import lr_scheduler
import torch.nn as nn
from meta_learning1.model import datasets
from meta_learning1.model.datasets import *

from const import synergy_path,fpath

# training function at each epoch
def train( model,device,train_loader,train_loader1,optimizer,lr_scheduler):

    model.train()
    LOSS=0
    for batch_idx, data in enumerate(train_loader):
         for batch_idx1, data1 in enumerate(train_loader1):
            if batch_idx1==batch_idx:
                data=data.to(device)
                data1 = data1.to(device)
                optimizer.zero_grad()
                output = model(data, data1)
                loss = loss_func(output, data['y'].view(-1, 1).float().to(device))
                loss.backward()
                optimizer.step()
                if lr_scheduler.get_last_lr()[0]>0.00001:
                    lr_scheduler.step()
                LOSS=loss.item()
         #print("data loader")
    return LOSS

def predicting(model,device,loader,loader0,total_labels_p,total_preds_p):
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            for batch_idx1, data1 in enumerate(loader0):
                if batch_idx1 == batch_idx:
                    data = data.to(device)
                    data1 = data1.to(device)
                    output =model(data, data1)
                    total_preds_p = torch.cat((total_preds_p, output.cpu()), 0)
                    total_labels_p = torch.cat((total_labels_p, data['y'].float().view(-1, 1).cpu()), 0)

    return total_labels_p, total_preds_p

if __name__ == '__main__':
    cuda_name = "cuda:0"
    print('cuda_name:', cuda_name)
    TRAIN_BATCH_SIZE= 500
    VAL_BATCH_SIZE =1024
    lr =0.0001
    LOG_INTERVAL = 10
    NUM_EPOCHS = 200

    codes = pickle.load(open(fpath + 'codes.p', 'rb'))
    drug_features = pickle.load(open(fpath + 'drug_feature.p', 'rb'))
    cell_features = pickle.load(open(fpath + 'cell_exp.p', 'rb'))

    device=torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    SN=SynergyNet(n_output=1,num_features_xd=78,  output_dimd=128, output_dimc=256,dropoutc=0.2,dropoutf=0.3).to(device)


    loss_func = nn.BCELoss()


    optimizer = torch.optim.Adam(SN.parameters(), lr=lr)
    scheduler=lr_scheduler.StepLR(optimizer,step_size=50000,gamma=1.0)

    # few_shot setting or zero_shot setting
    model_file_name = 'pretrain_synergy' + '.model'

    #save_representation_model file
    model_path = 'pretrain_model/'

    seed = 1500
    tranditional_train = datasets.tranditional_model_data(train_rate=0.8, val_rate=0.2,data_path= synergy_path,seed=seed)
    df_train, df_val, _ = tranditional_train.get_train_batch(augment=False)
    train_sample=df_train
    print("train_sample_shape:", train_sample.shape)
    print("val_sample_shape:", df_val.shape)

    train_len = int(len(train_sample) / TRAIN_BATCH_SIZE)
    val_len = int(len(df_val) / VAL_BATCH_SIZE)
    best_auc = 0.0

    print('Training on {} samples...'.format(len(train_sample)))
    for epoch in range(1, NUM_EPOCHS + 1):
        index = [i for i in range(len(train_sample))]
        random.shuffle(index)
        train_sample = train_sample[index]
        aver_loss = 0.0
        for iteratation in range(0, train_len):
            train_data0 = train_sample[iteratation * TRAIN_BATCH_SIZE:(iteratation + 1) * TRAIN_BATCH_SIZE]
            train_data = TestbedDataset(train_data0, drug_features, cell_features, codes)
            train_data1 = TestbedDataset1(train_data0, drug_features, cell_features, codes)
            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
            train_loader1 = DataLoader(train_data1, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
            loss_all = train(SN, device, train_loader, train_loader1, optimizer,scheduler)
            aver_loss += loss_all
            log_interval = 20
            if iteratation % log_interval == 0 and iteratation > 0:
                cur_loss = aver_loss / log_interval
                print('| epoch {:3d} | {:5d}/{:5d} batches |  loss {:8.5f}'.format(epoch, iteratation,
                             int(len(train_loader) / TRAIN_BATCH_SIZE), cur_loss))
                aver_loss = 0
        #val
        if epoch % 50 == 0:
            print('val on {} samples...'.format(len(df_val)))
            total_preds = torch.Tensor()
            total_labels = torch.Tensor()
            for iteratation_val in range(0, val_len):
                val_sample = df_val[iteratation_val * VAL_BATCH_SIZE:(iteratation_val + 1) * VAL_BATCH_SIZE]
                val_data = TestbedDataset(val_sample, drug_features, cell_features, codes)
                val_data1 = TestbedDataset1(val_sample, drug_features, cell_features, codes)
                val_loader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False)
                val_loader1 = DataLoader(val_data1, batch_size=VAL_BATCH_SIZE, shuffle=False)
                total_labels, total_preds = predicting(SN, device, val_loader, val_loader1, total_labels, total_preds)
            G, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()

            # 计算评估指标
            accuracy = accuracy_score(G, P.round())
            precision = precision_score(G, P.round())
            recall = recall_score(G, P.round())
            f1 = f1_score(G, P.round())
            auc = roc_auc_score(G, P)
            aupr = average_precision_score(G, P)
            kappa = cohen_kappa_score(G, P.round())

            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("AUC:", auc)
            print("AUPR:", aupr)
            print("Cohen's Kappa:", kappa)

            # 检查当前 AUC 是否是最高的
            if auc > best_auc:
                best_auc = auc
                print(f"New best AUC: {best_auc:.4f}. Saving model...")
                best_model_name = f"best_model_auc_{best_auc:.4f}_epoch_{epoch}.pth"
                torch.save(SN.state_dict(), os.path.join(model_path, best_model_name))

            # 打印评估指标
            print("Accuracy:", accuracy)

