import argparse
from datetime import datetime

from torch.utils.data import DataLoader
import warnings
from utils_syn import *
import pickle

torch.set_num_threads(int(2))
warnings.filterwarnings('ignore')
time_str = str(datetime.now().strftime('%y%m%d%H%M'))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, cohen_kappa_score


def test(total_test_batches, data_test, model, device,best_step,best_auc,saved_model,sup):
    # load state dict

    _load_model =saved_model +str(best_step)+'k_1.5062_'+str(best_auc)+'acc_model_'+str(best_step)+'.pth'
    state_dict = torch.load(_load_model)['state_dict']
    model.load_state_dict(state_dict)
    # model.eval()
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    auc = 0
    aupr = 0
    kappa = 0
    for P in range(10):
        print("P=:", P)
        test_losses = []
        total_preds = torch.DoubleTensor().to(device)
        total_labels = torch.DoubleTensor().to(device)

        for test_step in range(total_test_batches):
            print("P:",P,"test_step:",test_step)
            x_support_set, x_target = data_test.get_test_batch(augment=False)

            b, spc, t = np.shape(x_support_set)
            support_set_images_ = x_support_set.reshape(b, spc, t)
            bt, spct, tt = np.shape(x_target)
            target_image_ = x_target.reshape(bt, spct, tt)

            support_target_images = np.concatenate([support_set_images_, target_image_], axis=1)
            b, s, t = np.shape(support_target_images)
            print(np.shape(support_target_images))
            support_target_images = support_target_images.reshape(b * s, t)
            Num_samples = len(support_target_images)
            test_data = TestbedDataset(support_target_images, drug_features, cell_features, codes)
            test_data1 = TestbedDataset1(support_target_images, drug_features, cell_features, codes)

            test_loader = DataLoader(test_data, batch_size=Num_samples, shuffle=False)
            test_loader1 = DataLoader(test_data1, batch_size=Num_samples, shuffle=False)

        # test_result
            for batch_idx, data in enumerate(test_loader):

                for batch_idx1, data1 in enumerate(test_loader1):
                    if batch_idx1 == batch_idx:

                        data = data.to(device)
                        data1 = data1.to(device)

                        test_loss, val_output, val_target = model.run_batch(data, data1, b,  False)
                        test_losses.append(test_loss.item())

                        total_preds = torch.cat((total_preds, val_output), 0)
                        total_labels = torch.cat((total_labels, val_target), 0)

        total_preds = total_preds.cpu().detach().numpy().flatten()
        total_labels = total_labels.cpu().detach().numpy().flatten()
        print(np.shape(total_preds))

        total_preds_binary = (total_preds > 0.5)
        total_labels_numpy = total_labels

        # 计算评估指标
        accuracy += accuracy_score(total_labels_numpy, total_preds_binary)
        precision += precision_score(total_labels_numpy, total_preds_binary)
        recall += recall_score(total_labels_numpy, total_preds_binary)
        f1 += f1_score(total_labels_numpy, total_preds_binary)
        auc += roc_auc_score(total_labels_numpy, total_preds)
        aupr += average_precision_score(total_labels_numpy, total_preds)
        kappa += cohen_kappa_score(total_labels_numpy, total_preds_binary)



        # np.savetxt(result_path + "label_test_" + str(sup) + "_" + str(best_MSE) + "_" + str(P) + ".txt",
        #  total_labels, delimiter=",")
        # np.savetxt(result_path + "pred_test_" + str(sup) + "_" + str(best_MSE) + "_" + str(P) + ".txt",
        # total_preds, delimiter=",")

    accuracy = accuracy / 10
    precision = precision / 10
    recall = recall / 10
    f1 = f1 / 10
    auc = auc / 10
    aupr = aupr / 10
    kappa = kappa / 10

    # 打印评估指标
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC:", auc)
    print("AUPR:", aupr)
    print("Cohen's Kappa:", kappa)

    print()
    print('=' * 50)

    print("test Loss: {:0.05f}".format(accuracy), precision,
            recall, f1, auc, aupr, kappa)
    print('=' * 50)

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


    cuda_name = 'cuda:0'
    device = cuda_name if torch.cuda.is_available() else "cpu"
    lr = 0.0001
    batch_size = 5
    samples_support = 5 #10,30
    samples_query = 5
    total_test_batches = 100
    saved_model_path='saved_model/saved_model_few_shot_setting/'

    result_path = 'results/result_few_shot/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    fpath = './dr_data/all_sample_features/'
    codes = pickle.load(open(fpath + 'codes.p', 'rb'))
    drug_features = pickle.load(open(fpath + 'drug_feature.p', 'rb'))
    cell_features = pickle.load(open(fpath + 'cell_exp.p', 'rb'))

    embed = 128
    best_auc, best_step= format(0.5948, '.4f'), 20  #or your trained model
    method = 'test_few_shot_cell lines, 50-5,dim=128, inner learning rate=0.1'

    experiment_nameTEs = f'cell_few_shot_test_{embed}embed_{lr}LR_{1}shot'
    logss = "{}way{}shot , with {} tasks, test_batch is{},mse is{},method is {},step is{} ".format(samples_support,
                                              samples_query,batch_size,total_test_batches,
                                               best_auc, method,best_step)


    save_model_name = 'cell_few_shot_' + str(samples_query) + 'qs_' + str(samples_support) + 'ss' + str(
        batch_size)

    print(device)

    mini = data_syn.MiniCellDataSet(batch_size=batch_size, samples_support=samples_support, samples_query=samples_query)
    DAmeta_model = DAmeta(num_support = samples_support, num_query=samples_query).to(device)

    model_path = './pretrain_model/best_model_auc_0.8553_epoch_100.pth'

    pretrained_dict = torch.load(model_path, map_location=device)
    # read MMN's params
    model_dict = DAmeta_model.extractor.state_dict()
    # read same params in two model
    state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    # updata ,
    model_dict.update(state_dict)
    # load part of model params
    DAmeta_model.extractor.load_state_dict(model_dict)
    # freeze
    for p in DAmeta_model.extractor.parameters():
        # print(p)
        p.requires_grad = False

    print("-------------------begin test ----------------------")

    test(total_test_batches, mini, DAmeta_model, device,best_step,best_auc,saved_model_path,samples_support)