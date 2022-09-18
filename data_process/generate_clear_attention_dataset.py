
import os.path as osp
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import random
import pickle
import sys
import os.path as osp
import os
import argparse
import torch
from torch_geometric.data import Data
import numpy as np
sys.path.append(os.path.realpath('.'))
from data_process.classes import LncRNA_Protein_Interaction
from data_process.classes import LncRNA_Protein_Interaction_dataset_1hop_1222_InMemory

from data_process.generate_edgelist import read_interaction_dataset
from data_process.methods import nodeSerialNumber_listIndex_dict_generation, nodeName_listIndex_dict_generation
from data_process.methods import reset_basic_data
from data_process.generate_dataset import read_set_interactionKey
from data_process.generate_dataset import build_dict_serialNumber_node
from data_process.generate_dataset import exam_set_allInteractionKey_train_test
from data_process.generate_dataset import return_node_list_and_edge_list
from data_process.generate_dataset import read_random_node_embedding
from data_process.generate_dataset import load_node_k_mer
from data_process.generate_dataset import load_exam
from data_process.generate_dataset import exam_list_all_interaction
from data_process.methods import nodeSerialNumber_listIndex_dict_generation

def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument('--projectName', default='1230_1', help='project name')
    parser.add_argument('--fold', default=3, help='which fold is this')
    # parser.add_argument('--datasetType', help='training or testing or testing_selected')
    parser.add_argument('--interactionDatasetName', default='NPInter2', help='raw interactions dataset')
    parser.add_argument('--createBalanceDataset', default=1, type=int,
                        help='have you create a balance dataset when you run generate_edgelist.py, 0 means no, 1 means yes')
    parser.add_argument('--inMemory', default=1, type=int, help='1 or 0: in memory dataset or not')
    # parser.add_argument('--hopNumber', default=1, type=int, help='hop number of subgraph')
    parser.add_argument('--shuffle', default=1, type=int, help='shuffle interactions before generate dataset')
    parser.add_argument('--noKmer', default=0, type=int, help='Not using k-mer')
    parser.add_argument('--randomNodeEmbedding', default=0, type=int,
                        help='1: use rangdom vector as node Embedding, 0: use node2vec')
    parser.add_argument('--output', default=1, type=int, help='output dataset or not')

    return parser.parse_args()




def rebuild_all_negativeInteraction(set_negativeInteractionKey):
    global lncRNA_list, protein_list, negative_interaction_list
    dict_serialNumber_lncRNA = build_dict_serialNumber_node(lncRNA_list)
    dict_serialNumber_protein = build_dict_serialNumber_node(protein_list)
    # 根据set_negativeInteractionKey把负样本集构造出来
    for negativeInteractionKey in set_negativeInteractionKey:
        lncRNA_temp = dict_serialNumber_lncRNA[negativeInteractionKey[0]]
        protein_temp = dict_serialNumber_protein[negativeInteractionKey[1]]
        # 构造负样本
        temp_negativeInteraction = LncRNA_Protein_Interaction(lncRNA_temp, protein_temp, 0, negativeInteractionKey)
        negative_interaction_list.append(temp_negativeInteraction)
        #lncRNA_temp.interaction_list.append(temp_negativeInteraction)
        #protein_temp.interaction_list.append(temp_negativeInteraction)


def read_node2vec_result(path):
    print('read node2vec result')
    node_list, edge_list = return_node_list_and_edge_list()
    serialNumber_listIndex_dict = nodeSerialNumber_listIndex_dict_generation(node_list)

    node2vec_result_file = open(path, mode='r')
    lines = node2vec_result_file.readlines()
    lines.pop(0)  # 第一行包含：节点个数 节点嵌入后的维度
    for line in lines:
        arr = line.strip().split(' ')
        serial_number = int(arr[0])
        arr.pop(0)
        node_list[serialNumber_listIndex_dict[serial_number]].embedded_vector = arr

    count_node_without_node2vecResult = 0
    for node in node_list:
        if len(node.embedded_vector) != 64:
            count_node_without_node2vecResult += 1
            node.embedded_vector = [0] * 64
    print(f'没有node2vec结果的节点数：{count_node_without_node2vecResult}')
    node2vec_result_file.close()

def return_node_list_and_edge_list():
    global interaction_list, negative_interaction_list, lncRNA_list, protein_list

    node_list = lncRNA_list[:]
    node_list.extend(protein_list)
    edge_list = interaction_list[:]
    edge_list.extend(negative_interaction_list)
    return node_list, edge_list





def local_subgraph_generation(interaction, h):
    # 防止图中的回路导致无限循环，所以添加过的interaction，要存起来
    added_interaction_list = []

    x = []
    edge_index = [[], []]
    sum_node = 0

    # 子图中每个节点都得有自己独特的serial number和structural label
    # 这是为了构建edge_index
    subgraph_node_serial_number = 0

    nodeSerialNumber_subgraphNodeSerialNumber_dict = {}
    subgraphNodeSerialNumber_node_dict = {}

    # 要加入局部子图的边
    set_interactionSerialNumberPair_wait_to_add = set()
    set_interactionSerialNumberPair_wait_to_add.add(
        (interaction.lncRNA.serial_number, interaction.protein.serial_number))

    # 给要加入局部子图中的点都分配好序号
    subgraph_serialNumber = 0
    nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction.lncRNA.serial_number] = subgraph_serialNumber
    subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction.lncRNA
    subgraph_serialNumber += 1
    nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction.protein.serial_number] = subgraph_serialNumber
    subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction.protein
    subgraph_serialNumber += 1
    RNA_proterinlist = []

    for interaction_temp in interaction.lncRNA.interaction_list:
        interaction_key = (interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number)
        if interaction_key not in set_interactionKey_cannotUse:###
            set_interactionSerialNumberPair_wait_to_add.add(
                (interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
            if interaction_temp.protein.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.protein.serial_number] = subgraph_serialNumber
                subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.protein
                subgraph_serialNumber += 1
                RNA_proterinlist.append(interaction_temp.protein.serial_number)  ##########################

    for interaction_temp in interaction.protein.interaction_list:
        interaction_key = (interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number)
        if interaction_key not in set_interactionKey_cannotUse:
            set_interactionSerialNumberPair_wait_to_add.add(
                (interaction_temp.lncRNA.serial_number, interaction_temp.protein.serial_number))
            if interaction_temp.lncRNA.serial_number not in nodeSerialNumber_subgraphNodeSerialNumber_dict.keys():
                nodeSerialNumber_subgraphNodeSerialNumber_dict[interaction_temp.lncRNA.serial_number] = subgraph_serialNumber
                subgraphNodeSerialNumber_node_dict[subgraph_serialNumber] = interaction_temp.lncRNA
                subgraph_serialNumber += 1
                for i in RNA_proterinlist:
                    if (interaction_temp.lncRNA.serial_number, i) in set_interactionKey_forline:
                        set_interactionSerialNumberPair_wait_to_add.add((interaction_temp.lncRNA.serial_number, i))

    set_interactionSerialNumberPair_wait_to_add.remove((interaction.lncRNA.serial_number, interaction.protein.serial_number))
    set_interactionSerialNumberPair_wait_to_add = [(interaction.lncRNA.serial_number,interaction.protein.serial_number)] + list(set_interactionSerialNumberPair_wait_to_add)

    # 构造edge_list
    for interaction_serialNumber_pair in set_interactionSerialNumberPair_wait_to_add:
        node1_subgraphSerialNumber = nodeSerialNumber_subgraphNodeSerialNumber_dict[
            interaction_serialNumber_pair[0]]
        node2_subgraphSerialNumber = nodeSerialNumber_subgraphNodeSerialNumber_dict[
            interaction_serialNumber_pair[1]]
        edge_index[0].append(node1_subgraphSerialNumber)
        edge_index[1].append(node2_subgraphSerialNumber)
        edge_index[0].append(node2_subgraphSerialNumber)
        edge_index[1].append(node1_subgraphSerialNumber)

    # 构造x
    for i in range(len(subgraphNodeSerialNumber_node_dict.keys())):
        node_temp = subgraphNodeSerialNumber_node_dict[i]
        vector = []
        if i == 0 or i == 1:
            vector.append(0)
        else:
            vector.append(1)
        for f in node_temp.embedded_vector:
            vector.append(float(f))
        vector.extend(node_temp.attributes_vector)
        x.append(vector)

    # y记录这个interaction的真假
    if interaction.y == 1:
        y = [1]
    else:
        y = [0]

    sum_node += len(x)
    # 用x,y,edge_index创建出data，加入存放data的列表local_subgraph_list
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index, )

    return data, subgraphNodeSerialNumber_node_dict, sum_node



if __name__ == "__main__":
    args = parse_args()

    #重现相互作用数据集
    interaction_dataset_path = '../../data/source_database_data/' + args.interactionDatasetName + '.xlsx'
    interaction_list, negative_interaction_list, lncRNA_list, protein_list, lncRNA_name_index_dict, protein_name_index_dict, set_interactionKey, \
    set_negativeInteractionKey = read_interaction_dataset(dataset_path=interaction_dataset_path,
                                                          dataset_name=args.interactionDatasetName)

    path_set_allInteractionKey = f'../../data/set_allInteractionKey/{args.projectName}'
    path_set_negativeInteractionKey_all = path_set_allInteractionKey + '/set_negativeInteractionKey_all'
    if args.createBalanceDataset == 1:
        set_negativeInteractionKey = read_set_interactionKey(path_set_negativeInteractionKey_all)

    # 重建负样本
    if args.createBalanceDataset == 1:
        rebuild_all_negativeInteraction(set_negativeInteractionKey)

    # 把训练集和测试集包含的边读取出来
    path_set_interactionKey_train = path_set_allInteractionKey + f'/set_interactionKey_train_{args.fold}'
    path_set_negativeInteractionKey_train = path_set_allInteractionKey + f'/set_negativeInteractionKey_train_{args.fold}'
    path_set_interactionKey_test = path_set_allInteractionKey + f'/set_interactionKey_test_{args.fold}'
    path_set_negativeInteractionKey_test = path_set_allInteractionKey + f'/set_negativeInteractionKey_test_{args.fold}'

    set_interactionKey_train = read_set_interactionKey(path_set_interactionKey_train)
    set_negativeInteractionKey_train = read_set_interactionKey(path_set_negativeInteractionKey_train)
    set_interactionKey_test = read_set_interactionKey(path_set_interactionKey_test)
    set_negativeInteractionKey_test = read_set_interactionKey(path_set_negativeInteractionKey_test)

    # 检查一下训练集和测试集有没有重合
    exam_set_allInteractionKey_train_test(set_interactionKey_train, set_negativeInteractionKey_train,
                                          set_interactionKey_test, set_negativeInteractionKey_test)

    # load node2vec result
    node2vec_result_path = f'../../data/node2vec_result/{args.projectName}/training_{args.fold}/result.emb'
    if args.randomNodeEmbedding == 0:
        read_node2vec_result(path=node2vec_result_path)
    else:
        read_random_node_embedding()

    # load k-mer
    if args.noKmer == 0:
        lncRNA_3_mer_path = f'../../data/lncRNA_3_mer/{args.interactionDatasetName}/lncRNA_3_mer.txt'
        protein_2_mer_path = f'../../data/protein_2_mer/{args.interactionDatasetName}/protein_2_mer.txt'
        load_node_k_mer(lncRNA_list, 'lncRNA', lncRNA_3_mer_path)
        load_node_k_mer(protein_list, 'protein', protein_2_mer_path)

    # 执行检查
    load_exam(args.noKmer, lncRNA_list, protein_list)  #

    # 数据集生成
    exam_list_all_interaction(interaction_list)
    exam_list_all_interaction(negative_interaction_list)
    all_interaction_list = interaction_list.copy()
    all_interaction_list.extend(negative_interaction_list)
    exam_list_all_interaction(all_interaction_list)

    # 生成局部子图，不能有测试集的边
    set_interactionKey_cannotUse = set()
    set_interactionKey_cannotUse.update(set_interactionKey_test)
    set_interactionKey_cannotUse.update(set_negativeInteractionKey_test)


    # 生成测试集
    if args.noKmer == 0:
        dataset_test_path = f'../../data/dataset/{args.projectName}_inMemory2_attention_test_{args.fold}'
    else:
        dataset_test_path = f'../../data/dataset/{args.projectName}_inMemory2_noKmer_attention_test_{args.fold}'
    if not osp.exists(dataset_test_path):
        print(f'创建了文件夹：{dataset_test_path}')
        os.makedirs(dataset_test_path)
    set_interactionKey_forGenerate = set()
    set_interactionKey_forline = set()
    set_interactionKey_forline.update(set_interactionKey_train)
    set_interactionKey_forGenerate.update(set_interactionKey_test)
    set_interactionKey_forGenerate.update(set_negativeInteractionKey_test)
   # My_testingDataset, subgraphNodeSerialNumber_node_list = LncRNA_Protein_Interaction_dataset_1hop_1222_InMemory(
    #    dataset_test_path,all_interaction_list, 1,set_interactionKey_forGenerate,set_interactionKey_cannotUse)

    num_data = len(set_interactionKey_forGenerate)
    print(f'the number of samples:{num_data}')
    data_list = []
    subgraphNodeSerialNumber_node_list = []
    count = 0
    sum_nodes = 0
    for interaction in all_interaction_list:
        interaction_key = (interaction.lncRNA.serial_number, interaction.protein.serial_number)
        if interaction_key in set_interactionKey_forGenerate:
            data, subgraphNodeSerialNumber_node_dict, sum_node = local_subgraph_generation(interaction, 1)
            data_list.append(data)
            subgraphNodeSerialNumber_node_list.append(subgraphNodeSerialNumber_node_dict)
            count = count + 1
            sum_nodes += sum_node
            if count % 100 == 0:
                print(f'{count}/{num_data}')
                print(f'average node number = {sum_nodes / count}')





    # test example
    test_list = []
    number_to_node_list = []
    for i in range(len(data_list)):
        if data_list[i].y == torch.tensor([1], dtype=torch.float):
            test_list.append(data_list[i])
            number_to_node_list.append(subgraphNodeSerialNumber_node_list[i])

    list_number_node={}
    list_node_number={}
    for i in lncRNA_list:
        list_number_node[i.serial_number]=i.name
        list_node_number[i.name]=i.serial_number
    for i in protein_list:
        list_number_node[i.serial_number]=i.name
        list_node_number[i.name] = i.serial_number



    """
    test = test_list[0]
    number_to_node = number_to_node_list[0]
    for i in range(len(test_list)):
        if test_list[i].num_nodes < test.num_nodes:
            test = test_list[i]
            number_to_node = number_to_node_list[i]
    """

    print('the subgraph for the test of attention:')
    print('test')
    print('number_to_node')

    torch.save(test_list, dataset_test_path+'/test')
    torch.save(number_to_node_list, dataset_test_path+'/number_to_node')
    torch.save(list(set_interactionKey_train), dataset_test_path+'/all_node_forgenerate_positiveline')
    torch.save(list(set_negativeInteractionKey_train), dataset_test_path + '/all_node_forgenerate_negativeline')
    torch.save(list_node_number,dataset_test_path+'/list_node_number')
    torch.save(list_number_node, dataset_test_path + '/list_number_node')
