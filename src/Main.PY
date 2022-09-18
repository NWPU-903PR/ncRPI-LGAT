import sys
import os.path as osp
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import pandas as pd
import random
from torch_geometric.data.data import Data
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from tqdm import tqdm
import time
import gc
import argparse
from src.util_functions import *
from src.model import Net, Net2
from data_process.classes import Net_1, LncRNA_Protein_Interaction_dataset, LncRNA_Protein_Interaction_inMemoryDataset

sys.path.append(os.path.realpath('.'))
from data_process.methods import dataset_analysis, average_list, Accuracy_Precision_Sensitivity_Specificity_MCC
from data_process.classes import LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory

from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
import torch
import torch.nn.functional as F
from torch.optim import *


def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument('--trainingName', default='1223_1', help='the name of this training')
    parser.add_argument('--trainingDatasetName', default='1223_1_inMemory2_train_0', help='the name of this object')
    parser.add_argument('--testingDatasetName', default='1223_1_inMemory2_test_0', help='the name of this object')
    parser.add_argument('--inMemory', default=1, type=int, help='in memory dataset or not')
    parser.add_argument('--interactionDatasetName', default='NPInter2', help='raw interactions dataset')
    parser.add_argument('--fold', default=0, type=int, help='this is part of cross validation, the ith fold')  #
    parser.add_argument('--epochNumber', default=100, type=int, help='number of training epoch')
    parser.add_argument('--hopNumber', default=1, type=int, help='hop number of subgraph')
    parser.add_argument('--node2vecWindowSize', default=5, type=int, help='node2vec window size')
    # parser.add_argument('--crossValidation', default=1, type=int, help='do cross validation')
    # parser.add_argument('--foldNumber', default=5, type=int, help='fold number of cross validation')
    parser.add_argument('--initialLearningRate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--l2WeightDecay', default=0.001, type=float, help='L2 weight')
    parser.add_argument('--batchSize', default=32, type=int, help='batch size')
    return parser.parse_args()


def train():
    classifier.train()
    loss_all = 0
    for data in train_loader:
        if torch.cuda.device_count() > 1:
            data = data
        else:
            data = data.cuda(device)
        optimizer.zero_grad()
        output = classifier(data)
        if torch.cuda.device_count() > 1:
            target = torch.cat([i.y.long() for i in data]).to(output.device)
        else:
            target = data.y.long()
        loss = F.nll_loss(output, target)
        loss.backward()
        if torch.cuda.device_count() > 1:
            loss_all += len(data) * loss.item()
        else:
            loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


if __name__ == "__main__":
    # 参数
    args = parse_args()

    training_dataset_path = f'../../data/dataset/{args.trainingDatasetName}'
    testing_dataset_path = f'../../data/dataset/{args.testingDatasetName}'
    # 读取数据集
    if args.inMemory == 0:
        raise Exception("not ready yet")
        train_dataset = Dataset(root=training_dataset_path)
        test_dataset = Dataset(root=testing_dataset_path)

    elif args.inMemory == 1:
        train_dataset = LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory(root=training_dataset_path)
        test_dataset = LncRNA_Protein_Interaction_dataset_1hop_1220_InMemory(root=testing_dataset_path)
    else:
        print(f'--inMemory : {args.inMemory}')
        raise Exception("--inMemory has to be 0 or 1")
    # 打乱数据集
    print('shuffle dataset\n')
    train_dataset = train_dataset.shuffle()
    test_dataset = test_dataset.shuffle()

    print('number of samples in testing dataset：', len(test_dataset), 'number of samples in training dataset：',
          len(train_dataset))
    print('training dataset')
    dataset_analysis(train_dataset)
    print('testing dataset')
    dataset_analysis(test_dataset)
    ###################################
    #train_dataset = train_dataset[:20]
    #test_dataset = test_dataset[:20]
    ###################################

    train_lines = to_linegraphs1(train_dataset)
    test_lines = to_linegraphs1(test_dataset)

    # 准备日志
    saving_path = f'../../src/result/{args.trainingName}'
    if not osp.exists(saving_path):
        print(f'创建文件夹：{saving_path}')
        os.makedirs(saving_path)

    # 迭代次数
    num_of_epoch = args.epochNumber

    # 学习率
    LR = args.initialLearningRate

    # L2正则化系数
    L2_weight_decay = args.l2WeightDecay

    # 日志基本信息写入
    log_path = saving_path + f'/log_{args.fold}.txt'
    result_file = open(file=log_path, mode='w')
    result_file.write(f'training dataset : {args.trainingDatasetName}')
    result_file.write(f'testing dataset : {args.testingDatasetName}')
    result_file.write(f'database：{args.interactionDatasetName}\n')
    result_file.write(f'node2vec_windowSize = {args.node2vecWindowSize}\n')
    result_file.write(f'number of eopch ：{num_of_epoch}\n')
    result_file.write(f'learn rate：initial = {LR}，whenever loss increases, multiply by 0.95\n')
    result_file.write(f'L2 weight decay = {L2_weight_decay}\n')

    # 创建保存模型的文件夹
    if osp.exists(saving_path + f'/model_{args.fold}_fold'):
        raise Exception('Same fold has been done')
    else:
        print(f'创建文件夹：{saving_path}' + f'/model_{args.fold}_fold')
        os.makedirs(saving_path + f'/model_{args.fold}_fold')
    # 创建模型

    num_of_classes = 2
    latent_dim = [16]
    latent_dim_a = [32]
    hidden = 128
    hidden_a = 64
    with_dropout = True
    batch_size = args.batchSize
    printAUC = True
    feat_dim = train_lines[0].num_node_features
    heads = 3
    dropout = 0.6
    Net_type = 'GAT'

    if torch.cuda.device_count() > 1:
        train_loader = DataListLoader(train_lines, batch_size=batch_size, shuffle=True)
        test_loader = DataListLoader(test_lines, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_lines, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_lines, batch_size=batch_size, shuffle=False)

    if Net_type == 'GAT':
        print('GAT')
        classifier = Net2(feat_dim, hidden_a, latent_dim_a, heads, dropout, with_dropout)
    else:
        print('GCN')
        classifier = Net(feat_dim, hidden, latent_dim, with_dropout)

    # CPU / GPU

    if torch.cuda.device_count() > 1:
        classifier = DataParallel(classifier)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, weight_decay=L2_weight_decay)
    # scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[int(num_of_epoch * 0.2),int(num_of_epoch * 0.8)],gamma = 0.8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)  # 0.95

    MCC_max = -1
    epoch_MCC_max = 0
    ACC_MCC_max = 0
    Pre_MCC_max = 0
    Sen_MCC_max = 0
    Spe_MCC_max = 0

    # 训练开始
    loss_last = float('inf')
    for epoch in range(num_of_epoch):
        loss = train()

        # loss增大时,降低学习率
        if loss > loss_last:
            scheduler.step()
        loss_last = loss

        # 训练中评价模型，监视训练过程中的模型变化, 并且写入文件
        if (epoch + 1) % 5 == 0 and epoch != num_of_epoch - 1:
            # 用Accuracy, Precision, Sensitivity, MCC评价模型
            # Accuracy, Precision, Sensitivity ,MCC = Accuracy_Precision_Sensitivity_MCC(model, train_loader, device)
            Accuracy, Precision, Sensitivity, Specificity, MCC, _ = Accuracy_Precision_Sensitivity_Specificity_MCC(
                classifier, train_loader, device)
            output = 'Epoch: {:03d}, training dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(
                epoch + 1, Accuracy, Precision, Sensitivity, Specificity, MCC)
            print(output)
            result_file.write(output + '\n')
            # Accuracy, Precision, Sensitivity, MCC = Accuracy_Precision_Sensitivity_MCC(model, test_loader, device)
            Accuracy, Precision, Sensitivity, Specificity, MCC, interaction_list = Accuracy_Precision_Sensitivity_Specificity_MCC(
                classifier, test_loader, device)
            output = 'Epoch: {:03d}, testing dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(
                epoch + 1, Accuracy, Precision, Sensitivity, Specificity, MCC)
            print(output)
            result_file.write(output + '\n')
            # 保存模型
            if MCC > MCC_max:
                MCC_max = MCC
                epoch_MCC_max = epoch + 1
                ACC_MCC_max = Accuracy
                Pre_MCC_max = Precision
                Sen_MCC_max = Sensitivity
                Spe_MCC_max = Specificity
                interaction_list_max = interaction_list
            network_model_path = saving_path + f'/model_{args.fold}_fold/{epoch + 1}'
            torch.save(classifier.state_dict(), network_model_path)

    # 训练结束，评价模型，并且把结果写入文件
    Accuracy, Precision, Sensitivity, Specificity, MCC, _ = Accuracy_Precision_Sensitivity_Specificity_MCC(classifier,
                                                                                                        train_loader,
                                                                                                        device)
    output = 'result, training dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(
        Accuracy, Precision, Sensitivity, Specificity, MCC)
    print(output)
    result_file.write(output + '\n')
    Accuracy, Precision, Sensitivity, Specificity, MCC, interaction_list = Accuracy_Precision_Sensitivity_Specificity_MCC(classifier,
                                                                                                        test_loader,
                                                                                                        device)
    output = 'result, testing dataset, Accuracy: {:.5f}, Precision: {:.5f}, Sensitivity: {:.5f}, Specificity: {:.5f}, MCC: {:.5f}'.format(
        Accuracy, Precision, Sensitivity, Specificity, MCC)
    print(output)

    result_file.write(output + '\n')
    if MCC > MCC_max:
        MCC_max = MCC
        epoch_MCC_max = args.epochNumber
        ACC_MCC_max = Accuracy
        Pre_MCC_max = Precision
        Sen_MCC_max = Sensitivity
        Spe_MCC_max = Specificity
        interaction_list_max = interaction_list
    # 把模型存起来
    network_model_path = saving_path + f'/model_{args.fold}_fold/{num_of_epoch}'
    torch.save(classifier.state_dict(), network_model_path)

    result_file.write('\n')
    output = f'MCC最大的时候的性能：'
    print(output)
    result_file.write(output + '\n')
    output = f'epoch: {epoch_MCC_max}, MCC: {MCC_max}, ACC: {ACC_MCC_max}, Pre: {Pre_MCC_max}, Sen: {Sen_MCC_max}, Spe: {Spe_MCC_max}'
    print(output)
    print(interaction_list_max)                  #add   FP_list
    result_file.write(output + '\n')

    # 完毕

    result_file.close()

    df = pd.DataFrame(interaction_list_max)

    df.to_excel(saving_path + f'/log_{args.fold}.xlsx', index=False)


    print('\nexit\n')

