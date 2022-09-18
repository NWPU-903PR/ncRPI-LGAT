#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:31:24 2019

@author: lei.cai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from src.gatv2_conv import GATv2Conv
from torch.autograd import Variable


class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim=[32, 32, 32], with_dropout=False):
        super(Net, self).__init__()
        conv = GCNConv  # GATConv  SAGEConv
        self.latent_dim = latent_dim
        self.conv_params = nn.ModuleList()
        self.conv_params.append(conv(input_dim, latent_dim[0], cached=False))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(conv(latent_dim[i - 1], latent_dim[i], cached=False))

        latent_dim = sum(latent_dim)

        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

        self.with_dropout = with_dropout

    def forward(self, data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        cur_message_layer = x
        cat_message_layers = []
        lv = 0
        while lv < len(self.latent_dim):
            cur_message_layer = self.conv_params[lv](cur_message_layer, edge_index)
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

        return logits


class Net2(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim=[32, 32], heads=3, dropout=0.6, with_dropout=False):
        super(Net2, self).__init__()
        conv = GATv2Conv  # GATConv  SAGEConv
        self.latent_dim = latent_dim
        self.conv_params = nn.ModuleList()
        self.conv_params.append(conv(input_dim, latent_dim[0], heads, dropout=dropout))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(conv(latent_dim[i - 1] * heads, latent_dim[i], heads=heads, dropout=dropout))

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
            cur_message_layer = self.conv_params[lv](cur_message_layer, edge_index)
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

        return logits