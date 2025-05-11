import argparse
from datetime import datetime
from itertools import cycle

from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, cohen_kappa_score
import pickle

from meta_learning1.model import datasets, loss,network
from meta_learning1.model.datasets import *
from torch.nn.parameter import Parameter
from const import fpath,fpaths
from meta_learning1.model.model import DAmeta,AdversarialNetwork

torch.set_num_threads(int(2))
warnings.filterwarnings('ignore')
time_str = str(datetime.now().strftime('%y%m%d%H%M'))

class LearnedWeight(nn.Module):
    def __init__(self):
        super(LearnedWeight, self).__init__()
        # 初始化可学习的权重，所有权重初始化为1
        self.s1_weight = Parameter(torch.ones(1), requires_grad=True)
        self.s2_weight = Parameter(torch.ones(1), requires_grad=True)
        self.sy_weight = Parameter(torch.ones(1), requires_grad=True)

    def forward(self, s1_loss, s2_loss, sy_loss):
        # 应用 exp(-weight) 公式以确保权重为正且可调节
        weighted_s1_loss = torch.exp(-self.s1_weight) * s1_loss
        weighted_s2_loss = torch.exp(-self.s2_weight) * s2_loss
        weighted_sy_loss = torch.exp(-self.sy_weight) * sy_loss

        # 计算最终的加权损失
        final_loss = weighted_s1_loss + weighted_s2_loss + weighted_sy_loss
        return final_loss

def train(train_batches, data_train, model, optimizer, optimizer_lr, lr_scheduler, device):
    print('start train')

    saved_model = 'pretrain_model/'
    _load_model = saved_model + 'beat_model_auc_' + str(0.8562) + 'epoch' + str(500) + '.pth'
    state_dict = torch.load(_load_model)['state_dict']
    model.load_state_dict(state_dict)

    autoweight = LearnedWeight().cuda()

    best_loss = float('inf')
    best_auc = -float('inf')

    total_preds_train = torch.Tensor().to(device)
    total_labels_train = torch.Tensor().to(device)
    total_loss = []

    for step in range(1, train_batches + 1):

        model.train()
        optimizer.zero_grad()
        optimizer_lr.zero_grad()
        # do training
        x_support_set, x_target = data_train.get_train_batch(augment=False)
        pa_support_set, pa_target = data_train.get_pa_batch(augment=False)

        b, spc, t = np.shape(x_support_set)
        support_set_samples_ = x_support_set.reshape(b, spc, t)
        bt, spct, tt = np.shape(x_target)
        target_samples_ = x_target.reshape(bt, spct, tt)
        support_target_samples = np.concatenate([support_set_samples_, target_samples_], axis=1)

        b, s, t = np.shape(support_target_samples)
        support_target_samples = support_target_samples.reshape(b * s, t)
        Num_samples = len(support_target_samples)
        print("data over", Num_samples)



        b1, spc1, t1 = np.shape(pa_support_set)
        support_set_samples1_ = pa_support_set.reshape(b1, spc1, t1)
        bt1, spct1, tt1 = np.shape(pa_target)
        target_samples1_ = pa_target.reshape(bt1, spct1, tt1)
        support_target_samples1 = np.concatenate([support_set_samples1_, target_samples1_], axis=1)

        b1, s1, t1 = np.shape(support_target_samples1)
        support_target_samples1 = support_target_samples1.reshape(b1 * s1, t1)
        Num_samples1 = len(support_target_samples1)
        print("data over", Num_samples1)






        train_data = TestbedDataset(support_target_samples, drug_features, cell_features, codes)
        train_data1 = TestbedDataset1(support_target_samples, drug_features, cell_features, codes)

        train_loader = DataLoader(train_data, batch_size=Num_samples, shuffle=False)
        train_loader1 = DataLoader(train_data1, batch_size=Num_samples, shuffle=False)



        pa_data = TestbedDataset(support_target_samples1, drug_features1, cell_features1, codes,Ispa = True)
        pa_data1 = TestbedDataset1(support_target_samples1, drug_features1, cell_features1, codes)
        pa_loader = DataLoader(pa_data, batch_size=Num_samples1, shuffle=False)
        pa_loader1 = DataLoader(pa_data1, batch_size=Num_samples1, shuffle=False)

        pa_loader_cycle = cycle(pa_loader)
        pa_loader1_cycle = cycle(pa_loader1)




        for batch_idx, data in enumerate(train_loader):
            for batch_idx1, data1 in enumerate(train_loader1):
                if batch_idx1 == batch_idx:
                    data = data.to(device)
                    data1 = data1.to(device)

                    data2 = next(pa_loader_cycle)
                    data22 = next(pa_loader1_cycle)
                    data2 = data2.to(device)
                    data22 = data22.to(device)


                    query_loss, query_output, query_target,target_embedings,target_label,target_label2,pa_embeddings = \
                        model.run_batch(data, data1,data2,data22, batch_size)
                    print('(DAmeta-TRAIN) [Step: %d/%d]  query_loss: %4.4f' % (
                        step, train_batches, query_loss))

                    features = torch.cat((target_embedings, pa_embeddings), dim=0)
                    outputs = torch.cat((target_label, target_label2), dim=0)
                    softmax_out = nn.Softmax(dim=1)(outputs)
                    entropy = loss.Entropy(softmax_out)
                    transfer_loss = loss.CDAN([features, softmax_out], AdversarialNetwork, entropy, network.calc_coeff(step),
                                              None)

                    losses = autoweight(query_loss, transfer_loss) / 10

                    losses.backward()
                    total_loss.append(losses.item())
                    total_preds_train = torch.cat((total_preds_train, query_output), 0)
                    total_labels_train = torch.cat((total_labels_train, query_target), 0)
                    optimizer.step()
                    optimizer_lr.step()
                    if lr_scheduler.get_last_lr()[0] > 0.00001:
                        lr_scheduler.step()

        # if step % 10 == 0:
        #
        #     train_total_preds = total_preds_train.cpu().detach().numpy().flatten()
        #     train_total_labels = total_labels_train.cpu().detach().numpy().flatten()
        #
        #     # 将输出转换为二元标签
        #     train_total_preds_binary = (train_total_preds > 0.5)
        #
        #     # 计算评估指标
        #     accuracy = accuracy_score(train_total_labels, train_total_preds_binary)
        #     precision = precision_score(train_total_labels, train_total_preds_binary)
        #     recall = recall_score(train_total_labels, train_total_preds_binary)
        #     f1 = f1_score(train_total_labels, train_total_preds_binary)
        #     auc = roc_auc_score(train_total_labels, train_total_preds)
        #     aupr = average_precision_score(train_total_labels, train_total_preds)
        #     kappa = cohen_kappa_score(train_total_labels, train_total_preds_binary)
        #
        #     # 打印评估指标
        #     print("Accuracy:", accuracy)
        #     print("Precision:", precision)
        #     print("Recall:", recall)
        #     print("F1 Score:", f1)
        #     print("AUC:", auc)
        #     print("AUPR:", aupr)
        #     print("Cohen's Kappa:", kappa)
        #
        #     print()
        #     print('=' * 50)
        #     print("train Epoch: {} --- accuracy: {:4.4f}".format(step, accuracy), precision,
        #           recall, f1, auc, aupr, kappa)
        #     print('=' * 50)
        #
        #     total_preds_train = torch.Tensor().to(device)
        #     total_labels_train = torch.Tensor().to(device)
        #     train_loss = []

        if step % 20 == 0:
            # val model
            # model.eval()
            val_losses = []
            total_preds = torch.Tensor().to(device)
            total_labels = torch.Tensor().to(device)

            for val_step in range(total_val_batches):
                optimizer.zero_grad()
                optimizer_lr.zero_grad()
                x_support_set, x_target = data_train.get_test_batch(augment=False)

                b, spc, t = np.shape(x_support_set)
                support_set_samples_ = x_support_set.reshape(b, spc, t)
                bt, spct, tt = np.shape(x_target)
                target_sample_ = x_target.reshape(bt, spct, tt)

                support_target_samples = np.concatenate([support_set_samples_, target_sample_], axis=1)

                b, s, t = np.shape(support_target_samples)
                support_target_samples = support_target_samples.reshape(b * s, t)
                Num_samples = len(support_target_samples)
                val_data = TestbedDataset(support_target_samples, drug_features, cell_features, codes)
                val_data1 = TestbedDataset1(support_target_samples, drug_features, cell_features, codes)

                val_loader = DataLoader(val_data, batch_size=Num_samples, shuffle=False)
                val_loader1 = DataLoader(val_data1, batch_size=Num_samples, shuffle=False)

                # val result
                for batch_idx, data in enumerate(val_loader):
                    for batch_idx1, data1 in enumerate(val_loader1):
                        if batch_idx1 == batch_idx:
                            data = data.to(device)
                            data1 = data1.to(device)

                            query_loss, val_output, val_target = model.run_batch(
                                data, data1, b, False)
                            print(
                                '(Meta-Valid) [Step: %d/%d]  query_loss: %4.4f' % (
                                    step, train_batches, query_loss))

                            val_losses.append(query_loss.item())  #
                            total_preds = torch.cat((total_preds, val_output), 0)
                            total_labels = torch.cat((total_labels, val_target), 0)
            print(total_preds.size())
            LOSS = sum(val_losses) / len(val_losses)

            total_preds = total_preds.cpu().detach().numpy().flatten()
            total_labels = total_labels.cpu().detach().numpy().flatten()

            # 将输出转换为二元标签
            total_preds_binary = (total_preds > 0.5)
            total_labels_numpy = total_labels

            # 计算评估指标
            accuracy = accuracy_score(total_labels_numpy, total_preds_binary)
            precision = precision_score(total_labels_numpy, total_preds_binary)
            recall = recall_score(total_labels_numpy, total_preds_binary)
            f1 = f1_score(total_labels_numpy, total_preds_binary)
            auc = roc_auc_score(total_labels_numpy, total_preds)
            aupr = average_precision_score(total_labels_numpy, total_preds)
            kappa = cohen_kappa_score(total_labels_numpy, total_preds_binary)

            # 打印评估指标
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("AUC:", auc)
            print("AUPR:", aupr)
            print("Cohen's Kappa:", kappa)

            # 根据新的指标决定是否保存模型
            if LOSS < best_loss or auc > best_auc:  # 示例：同时考虑 LOSS 和 accuracy
                best_loss = LOSS
                best_auc = auc
                model_name = '%dk_%4.4f_%4.4facc_model' % (step, sum(val_losses) / len(val_losses), auc)
                # defined model name
                state = {'step': step, 'state_dict': model.state_dict()}
                if not os.path.exists('saved_model/saved_model_few_shot_setting/'):
                    os.makedirs('saved_model/saved_model_few_shot_setting/', exist_ok=False)
                save_path = "saved_model/saved_model_few_shot_setting/{}_{}.pth".format(model_name, step)
                torch.save(state, save_path)

            model.train()

            print()
            print('=' * 50)  #
            print("Validation Epoch: {} --- Meta val Loss: {:4.4f}".format((step, accuracy), precision,
                                                                           recall, f1, auc, aupr, kappa))
            print('=' * 50)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help="n epoch")
    parser.add_argument('--batch', type=int, default=128, help="batch size")
    parser.add_argument('--gpu', type=int, default=0, help="cuda device")
    parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
    parser.add_argument('--suffix', type=str, default=time_str, help="model dir suffix")
    # parser.add_argument('--hidden', type=int, nargs='+', default=[2048, 4096, 8192], help="hidden size")
    # parser.add_argument('--lr', type=float, nargs='+', default=[1e-3, 1e-4, 1e-5], help="learning rate")
    parser.add_argument('--hidden', type=int, default=1024, help="hidden size")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--support', type=int, default=5, help="support set")
    parser.add_argument('--query', type=int, default=4, help="query set")
    args = parser.parse_args()

    codes = pickle.load(open(fpath + 'codes.p', 'rb'))
    drug_features = pickle.load(open(fpath + 'drug_feature.p', 'rb'))
    cell_features = pickle.load(open(fpath + 'cell_exp.p', 'rb'))

    codes1 = pickle.load(open(fpaths + 'codes.p', 'rb'))
    drug_features1 = pickle.load(open(fpaths + 'drug_feature.p', 'rb'))
    cell_features1 = pickle.load(open(fpaths + 'patient_exp.p', 'rb'))



    cuda_name = 'cuda:0'

    lr = args.lr
    total_epochs = args.epoch
    batch_size = args.batch
    samples_support = args.support
    samples_query = args.query

    total_val_batches = 3

    device = cuda_name if torch.cuda.is_available() else "cpu"
    print(f'{samples_support}way{samples_query}shot, with {batch_size} tasks, test_batch is {total_val_batches}')

    save_model_name = 'DAmeta_' + str(samples_query) + 'qs_' + str(samples_support) + 'ss' + str(batch_size)

    print(device)

    mini = datasets.MiniCellDataSet(batch_size=batch_size, samples_support=samples_support, samples_query=samples_query)



    DAmeta_model = DAmeta(num_support = samples_support, num_query=samples_query).to(device)
    Adversarial_Network = AdversarialNetwork(1024,1024).to(device)

    autoweight = LearnedWeight().cuda()

    model_path = './pretrain_model/best_model.pth'

    pretrained_dict = torch.load(model_path, map_location=device)

    model_dict = DAmeta_model.extractor.state_dict()
    state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)

    # load part of model params
    DAmeta_model.extractor.load_state_dict(model_dict)
    # freeze
    for p in DAmeta_model.extractor.parameters():
        p.requires_grad = False


    lr_list = ['finetune_lr']
    # inner learning rate
    params = [x[1] for x in list(filter(lambda kv: kv[0] not in lr_list, DAmeta_model.named_parameters() ))]
    # 获取 AdversarialNetwork 和 autoweight 的参数
    adversarial_params = list(Adversarial_Network.parameters())
    autoweight_params = list(autoweight.parameters())

    # 将 AdversarialNetwork 和 autoweight 的参数添加到 params 列表中
    params.extend(adversarial_params)
    params.extend(autoweight_params)

    lr_params = [x[1] for x in list(filter(lambda kv: kv[0] in lr_list, DAmeta_model.named_parameters()))]

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=lr, weight_decay=1.0e-5)  # 1.0e-6
    # optimize inner learning rate
    optimizer_lr = torch.optim.Adam(lr_params, lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 2500, 5000, 10000, 30000], gamma=0.5)

    # Train
    print("-------------------begin train ----------------------")
    train(total_epochs, mini, DAmeta_model, optimizer, optimizer_lr, scheduler, device)
