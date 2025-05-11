import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import GCNConv, global_max_pool as gmp
import torch

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class Weight_generstive_netwrok(nn.Module):
    def __init__(self):
        super(Weight_generstive_netwrok, self).__init__()
        self.metanet_g1 = nn.Linear(512, ((512 + 1) * 1024) * 1)
        self.metanet_g2 = nn.Linear(512, ((1024 + 1) * 256) * 1)
        self.metanet_g3 = nn.Linear(512, ((256 + 1) * 1) * 1)

    def forward(self, x):

        x_pool = torch.mean(x, dim=1)
        final1 =self.metanet_g1(x_pool)
        final2 =self.metanet_g2(x_pool)
        final3 = self.metanet_g3(x_pool)
        meta_wts_1 = final1[:, :512 * 1024]
        meta_bias_1 = final1[:, 512 * 1024:]
        meta_wts_2 = final2[:, :1024 * 256]
        meta_bias_2 = final2[:, 1024 * 256:]
        meta_wts_3 = final3[:, :256 * 1]
        meta_bias_3 = final3[:, 256 * 1:]
        meta_wts1 = F.normalize(meta_wts_1, p=2, dim=1)
        meta_wts2 = F.normalize(meta_wts_2, p=2, dim=1)
        meta_wts3 = F.normalize(meta_wts_3, p=2, dim=1)

        return [meta_wts1, meta_bias_1, meta_wts2, meta_bias_2, meta_wts3, meta_bias_3]


class Feature_embedding_network(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78,  output_dimd=128, output_dimc=256,dropout=0.2):
        super(Feature_embedding_network, self).__init__()

        #Drugs
        self.conv1= GCNConv(num_features_xd, num_features_xd*2)
        self.conv2= GCNConv(num_features_xd*2, num_features_xd*4)
        self.conv3 = GCNConv(num_features_xd*4, num_features_xd * 2)

        # Cell lines
        self.gconv1 = nn.Conv2d(1, 32, 7, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.15)
        self.gconv2 = nn.Conv2d(32, 64, 5, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gconv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.gconv4 = nn.Conv2d(128, 64, 3, 1, 1)

        # Aggregation
        self.fc_g1= torch.nn.Linear(num_features_xd*2, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dimd)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm=nn.BatchNorm1d(num_features_xd*2)

        self.fcc1 = nn.Linear(64*1*7, output_dimc)
        self.normf=nn.BatchNorm1d(2*output_dimd+output_dimc)

    def forward(self, data_train,data0_train,task_batch):
        #Drug A
        x1_train, edge_index1_train, batch1_train= data_train['x'], data_train['edge_index'] ,data_train['batch']
        x1 = self.conv1(x1_train, edge_index1_train)
        x11 = self.relu(x1)
        x11 = self.conv2(x11, edge_index1_train)
        x11 = self.relu(x11)
        x11 = self.conv3(x11, edge_index1_train)
        x1=x1+x11
        x1=self.norm(x1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1_train)

        #Drug B
        x2_train, edge_index2_train,batch2_train=data0_train['x'], data0_train['edge_index'],data0_train['batch']
        x2 = self.conv1(x2_train, edge_index2_train)
        x22 = self.relu(x2)
        x22 = self.conv2(x22, edge_index2_train)
        x22 = self.relu(x22)
        x22 = self.conv3(x22, edge_index2_train)
        x2=x2+x22
        x2=self.norm(x2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2_train)

        #Cell lines
        cell = data_train.c
        spc, h = cell.size()
        # cell = cell.view(spc, int(h ** 0.5), int(h ** 0.5))
        # cell = cell.view(spc, 16, 31)
        cell = cell.view(spc, 15, 39)
        cell = cell.unsqueeze(1)
        xt = self.pool1(F.relu(self.norm1(self.gconv1(cell))))
        xt = self.pool2(F.relu(self.norm2(self.gconv2(xt))))
        xt = F.relu(self.gconv3(xt))
        xt = F.relu(self.gconv4(xt))
        # bs,kn, ce,ce1 = xt.size() #batch size *(K+Q),channel, width, height
        # xt = xt.view(-1,kn*ce*ce1)
        # xt = xt.view(-1, 64 * 2 * 5)
        xt = xt.view(-1, 64*1*7)

        # Aggregation
        x1 = self.relu(self.fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.fc_g2(x1)

        x2 = self.relu(self.fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.fc_g2(x2)

        xt = self.relu(self.fcc1(xt))

        xc = torch.cat((x1, x2, xt), 1)
        xc = self.normf(xc)
        bskn, ce = xc.size()

        embeddings = xc.view(task_batch,-1,ce)

        return embeddings


class DAmeta(nn.Module):
    def __init__(self, num_support,num_query):

        super(DAmeta, self).__init__()
        self.extractor = Feature_embedding_network()
        self.generative = Weight_generstive_netwrok()
        self.finetune_lr = nn.Parameter(torch.FloatTensor([0.0001]), requires_grad=True)
        self.generative_optimizer = optim.SGD(self.generative.parameters(), lr=0.0001)
        self.num_support = num_support  # K
        self.num_query = num_query  # Q
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(256)


    def run_batch(self, data_train, data0_train,data2,data22, batch_size, Is_train = True):

        Nq=self.num_query
        NS=self.num_support
        NB= batch_size

        Y = data_train['y'].view(NB, -1)
        label  = data_train['label'].view(NB, -1)
        label2  = data2['label'].view(NB, -1)

        # 使用 PyTorch 函数来检查无穷大和 NaN 值
        mask_finite = torch.isfinite(Y)  # 创建一个有限值的掩码
        valid_values = Y[mask_finite]  # 只保留有限（非 inf 和非 NaN）的值

        # 计算平均值，确保避开 inf 和 NaN 值
        mean_value = valid_values.mean()

        # 替换无穷大和 NaN 值
        Y = torch.where(torch.isinf(Y), mean_value, Y)  # 替换无穷大值
        Y = torch.where(torch.isnan(Y), mean_value, Y)  # 替换 NaN 值

        support_labels= Y[:, :NS]
        target_label=Y[:,NS:]

        target_Islabel = label[:,NS:]
        target_Islabel2 = label2[:,NS:]

        support_target_embeddings = self.extractor(data_train,data0_train,NB)
        pa_embeddings = self.extractor(data2,data22,NB)





        support_embedings = support_target_embeddings[:,:NS]
        target_embedings = support_target_embeddings[:, NS:]
        specific_weights, loss = self.train_all_samples(support_embedings, support_labels, 15,
                                                                   Is_train, drop=0.5)

        # self.generative_optimizer.zero_grad()
        # s_weights = self.generative(target_embedings)

        if not Is_train:
            with torch.no_grad():
                target_loss, _, target_output = self.cal_target_loss(target_embedings,  specific_weights , target_label,
                                                                 drop=0.0)
                print(f"验证阶段——询问集样本损失：{target_loss}")

        else:
            target_loss, _, target_output = self.cal_target_loss(target_embedings, specific_weights, target_label,
                                                                 drop=0.5)
            print(f"询问集样本损失：{target_loss}")



        return target_loss, target_output, target_label, target_embedings,target_Islabel,target_Islabel2,pa_embeddings

    def weight_update(self, specific_weights):
        for K in range(15):
            for i in range(len(specific_weights)):
                specific_weights[i] = specific_weights[i] - self.finetune_lr * specific_weights[i].grad
                specific_weights[i].retain_grad()

        update_specific_weights = specific_weights

        return update_specific_weights



    def train_all_samples(self, inputs, target,H, Is_train,  drop=0.0):

        global train_loss, losses
        specific_weights = self.generative(inputs)

        for i in range(len(specific_weights)):
            specific_weights[i].retain_grad()


        for i in range(15):
            # self.generative_optimizer.zero_grad()
            # 计算整个批次的总损失
            train_loss, losses, _ = self.cal_target_loss(inputs, specific_weights, target, drop)
            #print(f"当前循环中批次的训练损失为{train_loss}")
            # train_loss.backward()
            train_loss.backward(retain_graph=True)
            for i in range(len(specific_weights)):
                specific_weights[i] = specific_weights[i] - self.finetune_lr * specific_weights[i].grad
                specific_weights[i].retain_grad()

            # grads = torch.autograd.grad(train_loss, self.generative.parameters(), create_graph=True)
            # updated_params = list(map(lambda p: p[1] - p[0], zip(grads, self.generative.parameters())))
            # updated_model = update_model_params(updated_model, updated_params)

            # self.generative_optimizer.step()
            # specific_weights = self.generative(inputs)

        print(f"当前循环中批次的训练损失为{train_loss}")


        losses1 = losses
        hard_loss = 0



        # 困难样本采样  # 找到损失最高的 H 个样本的索引
        hard_batches_indices = sorted(range(len(losses1)), key=lambda i: losses1[i], reverse=True)[:H]

        # 收集苦难样本和对应的目标
        hard_inputs = inputs[hard_batches_indices]
        hard_targets = target[hard_batches_indices]

        # self.generative_optimizer.zero_grad()
        # 计算苦难样本的平均损失
        hard_loss, _, _ = self.cal_target_loss(hard_inputs, specific_weights, hard_targets, drop)
        print(f"当前循环中批次的困难样本损失为{hard_loss}")

        hard_loss.backward(retain_graph=True)



        return specific_weights, train_loss



    def cal_target_loss(self, inputs, specific_weights, target, drop):
        outputs = self.Synergy_Prediction_Network(inputs, specific_weights, drop)
        criterion = nn.BCELoss()
        loss = criterion(outputs,target)
        losses = [criterion(output, target[i]).item() for i, output in enumerate(outputs)]
        return loss, losses, outputs



    def Synergy_Prediction_Network(self, inputs, weight, drop):
        b_size, K, embed_size = inputs.size()

        outputs = torch.Tensor().cuda()
        for i in range(b_size):
            input = inputs[i].view(K, embed_size)
            weights1 = weight[0][i].view(1024, 512)
            weight_b1 = weight[1][i].view(1024)
            outputs1 = F.linear(input, weights1, weight_b1)
            outputs1 = self.batch_norm1(outputs1)
            outputs1 = F.relu(outputs1)
            outputs1 = F.dropout(outputs1, drop)

            weights1 = weight[2][i].view(256, 1024)
            weight_b1 = weight[3][i].view(256)
            outputs1 = F.linear(outputs1, weights1, weight_b1)
            outputs1 = self.batch_norm2(outputs1)
            outputs1 = F.relu(outputs1)
            outputs1 = F.dropout(outputs1, drop)

            weights2 = weight[4][i].view(1, 256)
            weight_b2 = weight[5][i].view(1)
            output = F.linear(outputs1, weights2, weight_b2)
            output = torch.sigmoid(output)
            outputs = torch.cat([outputs, output.squeeze()], dim=0)

        return outputs.view(b_size, -1)


class SynergyNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78,  output_dimd=128, output_dimc=256,dropoutc=0.2,dropoutf=0.2):

        super(SynergyNet, self).__init__()
        self.n_output = n_output

        #drugs
        self.conv1 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv2= GCNConv(num_features_xd*2, num_features_xd*4)
        self.conv3 = GCNConv(num_features_xd*4, num_features_xd*2 )
        self.fc_g1= torch.nn.Linear(num_features_xd*2, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dimd)
        self.norm=nn.BatchNorm1d(num_features_xd*2)
        self.relu = nn.ReLU()
        self.dropoutc = nn.Dropout(dropoutc)
        self.dropoutf = nn.Dropout(dropoutf)

        # cell lines (2d conv)
        self.gconv1 = nn.Conv2d(1, 32, 7, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.15)
        self.gconv2 = nn.Conv2d(32, 64, 5, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gconv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.gconv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.fcc1 = nn.Linear(64*2*5, output_dimc) #144

        # combined layers
        self.normf=nn.BatchNorm1d(2*output_dimd+output_dimc)
        self.fc1 = nn.Linear(2*output_dimd+output_dimc, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.out = nn.Linear(1024, self.n_output)

    def forward(self, data, data0, return_embeddings=False, task_batch=None):

        x1, edge_index1, batch1= data['x'], data['edge_index'] ,data['batch']
        x2, edge_index2,batch2=data0['x'], data0['edge_index'],data0['batch']
        cell = (data['c'])

        x1 = self.conv1(x1, edge_index1)
        x11 = self.relu(x1)
        x11 = self.conv2(x11, edge_index1)
        x11 = self.relu(x11)
        x11 = self.conv3(x11, edge_index1)

        x1=x11+x1
        x1=self.norm(x1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)  # global max pooling

        # flatten
        x1 = self.relu(self.fc_g1(x1))
        x1 = self.dropoutc(x1)
        x1 = self.fc_g2(x1)

        x2 = self.conv1(x2, edge_index2)
        x21 = self.relu(x2)
        x21 = self.conv2(x21, edge_index2)
        x21 = self.relu(x21)

        x21 = self.conv3(x21, edge_index2)
        x2=x21+x2
        x2 = self.norm(x2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2) # global max pooling

        # flatten
        x2 = self.relu(self.fc_g1(x2))
        x2 = self.dropoutc(x2)
        x2 = self.fc_g2(x2)

        #cell
        spc, h = cell.size()
        #cell = cell.view(spc, 127, 128)
        # cell = cell.view(spc, 16, 31)
        # cell = cell.view(spc, 15, 39)
        cell = cell.view(spc, 16, 31)
        cell = cell.unsqueeze(1)

        xt = self.pool1(F.relu(self.norm1(self.gconv1(cell))))
        xt = self.pool2(F.relu(self.norm2(self.gconv2(xt))))
        xt = F.relu(self.gconv3(xt))
        xt = F.relu(self.gconv4(xt))

        xt = xt.view(-1,64*2*5)
        # xt = xt.view(-1, 64*1*7)
        xt=self.fcc1(xt)

        # concat
        xc = torch.cat((x1,x2, xt), 1)
        xc=self.normf(xc)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropoutf(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropoutf(xc)
        # out = self.out(xc)
        out = torch.sigmoid(self.out(xc))
        return out


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
