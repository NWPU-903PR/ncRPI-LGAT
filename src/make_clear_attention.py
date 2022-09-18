
import os.path as osp
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import random
import pickle
import torch_geometric.utils as utils
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os.path as osp
import os
import argparse
import torch
from torch_geometric.data import DataLoader, DataListLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DataParallel
import numpy as np
sys.path.append(os.path.realpath('.'))
from src.util_functions import to_linegraphs1
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument('--projectName', default='1230_1', help='project name')
    parser.add_argument('--fold', default=3, help='which fold is this')
    parser.add_argument('--num_of_epoch', default=70, help='which num_of_epoch is best')
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
    parser.add_argument('--batchSize', default=32, type=int, help='batch size')

    return parser.parse_args()


from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros


class GATv2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention Networks?"
    <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the static
    attention problem of the standard :class:`~torch_geometric.conv.GATConv`
    layer: since the linear layers in the standard GAT are applied right after
    each other, the ranking of attended nodes is unconditioned on the query
    node. In contrast, in GATv2, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: int,
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True,
                 share_weights: bool = False,
                 **kwargs):
        super(GATv2Conv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights

        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights: bool = True):
        # type: (Union[Tensor, PairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class Net2(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim=[32, 32], heads=3, dropout=0.6, with_dropout=False):
        super(Net2, self).__init__()
        conv = GATv2Conv  # GATConv  SAGEConv
        self.latent_dim = latent_dim
        self.conv_params = nn.ModuleList()
        self.conv_params.append(conv(input_dim, latent_dim[0], heads, dropout=dropout))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(conv(latent_dim[i - 1]*heads, latent_dim[i], heads=heads, dropout=dropout))

        latent_dim = sum(latent_dim) * heads

        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

        self.with_dropout = with_dropout

    def forward(self, data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        cur_message_layer = x
        cat_message_layers = []
        lv = 0
        while lv < len(self.latent_dim):
            attr = self.conv_params[lv](cur_message_layer, edge_index)
            cur_message_layer = attr[0]
            cur_message_layer = torch.tanh(cur_message_layer)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)

        batch_idx = torch.unique(batch)
        idx = []
        for i in batch_idx:
            idx.append((batch == i).nonzero(as_tuple=False)[0].cpu().numpy()[
                           0])  # batch=0,wei di ji zhang tu ,yaodiyige,weisuo qiu jie dian dui

        cur_message_layer = cur_message_layer[idx, :]

        hidden = self.linear1(cur_message_layer)
        self.feature = hidden
        hidden = F.relu(hidden)

        if self.with_dropout:
            hidden = F.dropout(hidden, training=self.training)

        logits = self.linear2(hidden)
        logits = F.log_softmax(logits, dim=1)

        return attr, logits

if __name__ == "__main__":
    args = parse_args()

    # 生成测试集
    if args.noKmer == 0:
        dataset_test_path = f'../../data/dataset/{args.projectName}_inMemory2_attention_test_{args.fold}'
    else:
        dataset_test_path = f'../../data/dataset/{args.projectName}_inMemory2_noKmer_attention_test_{args.fold}'


    test_list = torch.load(dataset_test_path+'/test')
    number_to_node_list = torch.load(dataset_test_path+'/number_to_node')
    all_node_for_generate_positiveline = torch.load(dataset_test_path+'/all_node_forgenerate_positiveline')
    all_node_for_generate_negativeline = torch.load(dataset_test_path + '/all_node_forgenerate_negativeline')
    list_number_node = torch.load(dataset_test_path + '/list_number_node')
    list_node_number = torch.load(dataset_test_path + '/list_node_number')

    all_node_for_generate_positive=[]
    for i in all_node_for_generate_positiveline:
        all_node_for_generate_positive.append((list_number_node[i[0]],list_number_node[i[1]]))

    all_node_for_generate_negative= []
    for i in all_node_for_generate_negativeline:
        all_node_for_generate_negative.append((list_number_node[i[0]], list_number_node[i[1]]))


    #look the 1th graph
    test = test_list[1]# the same to down   56
    number_to_node = number_to_node_list[1]

    # linegraph_test
    test_line = to_linegraphs1([test])



    nodeNumber_subgraphNumbers = []
    sr = np.array(test.edge_index[0])
    for i in range(int(len(sr)/2)):
        node_number = [sr[i*2], sr[i*2+1]]
        nodeNumber_subgraphNumbers.append(node_number)
    for i in nodeNumber_subgraphNumbers:
        if i[0] > i[1]:
            i[0], i[1] = i[1], i[0]

    nodeNumber_subgraphNumbers.sort()
    #make the vertex of the Line_graph according the RNA and protein
    vertex_p_r = []
    for i in nodeNumber_subgraphNumbers:
        node1 = number_to_node[i[0]].name
        node2 = number_to_node[i[1]].name
        pair_node = [node1, node2]
        vertex_p_r.append(pair_node)



    #模型存
    saving_path = f'../../src/result/{args.projectName}'
    network_model_path = saving_path + f'/model_{args.fold}_fold/{args.num_of_epoch}'

    batch_size = args.batchSize
    num_of_classes = 2
    latent_dim = [16]
    latent_dim_a = [8]
    hidden = 128
    hidden_a = 64
    feat_dim = test_line[0].num_node_features
    heads = 3
    dropout = 0.6
    with_dropout = True

    if torch.cuda.device_count() > 1:
        test_loader = DataListLoader(test_line, batch_size=batch_size, shuffle=False)
    else:
        test_loader = DataLoader(test_line, batch_size=batch_size, shuffle=False)

    classifier = Net2(feat_dim, hidden_a, latent_dim_a, heads, dropout, with_dropout)

    # CPU / GPU
    if torch.cuda.device_count() > 1:
        classifier = DataParallel(classifier)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(network_model_path))
    classifier.eval()


    for data in test_loader:
        if torch.cuda.device_count() > 1:
            data = data
        else:
            data = data.to(device)
    attention, data = classifier(data)
    attention = attention[1]
    pred = data.max(dim=1)[1]
    print(int(pred))    #chose a graph of the value= 1
    #make the attention of the vertex
    vertex_atten = []
    for i, j in enumerate(attention[0][1]):
        if j == 0:
            a = attention[0][0][i].cpu().numpy().tolist()
            b = attention[1][i].cpu().detach().numpy().tolist()
            vertex_atten.append([a, b])
    """
    #draw map
    map_sample = utils.to_networkx(test)
    color_map =[]
    for node in map_sample:
        if node < 2:
            color_map.append("red")
        else:
            color_map.append("green")
    nx.draw(map_sample, node_color=color_map, with_labels=True, font_size=18, node_size=700)
    plt.show()
    """





    df = pd.DataFrame(vertex_p_r)
    df.to_excel(dataset_test_path + '/vertex_p_r.xlsx', index=False)

    df = pd.DataFrame(vertex_atten)
    df.to_excel(dataset_test_path + '/vertex_atten.xlsx', index=False)

    df = pd.DataFrame(all_node_for_generate_positive)
    df.to_excel(dataset_test_path + '/all_node_for_generate_positive.xlsx', index=False)

