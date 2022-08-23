#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Port_Scheduling
@File    ：state_model.py
@Author  ：JacQ
@Date    ：2022/3/30 9:41
"""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 0,
                 hidden_sizes: Sequence[int] = (256, 256),
                 activation: nn.Module = nn.LeakyReLU()
                 ) -> None:
        """
        Multilayer Perceptron

        Args:
            input_dim: dimension of input is [batch_size, input_dim]
            output_dim: dimension of output is [batch_size, output_dim]
            hidden_sizes: a sequence consisting of number of neurons per hidden layer
            activation: activation function
        """
        super().__init__()
        net = nn.Sequential()
        dim_last_layer = input_dim
        for i, num_neurons in enumerate(hidden_sizes):
            net.add_module(f'fc{i}', nn.Linear(dim_last_layer, num_neurons))
            net.add_module(f'act{i}', activation)
            dim_last_layer = num_neurons
        net.add_module('output layer', nn.Linear(dim_last_layer, output_dim))
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FlattenMlp(MLP):
    """
    Flatten inputs along dimension 1 ([u_action, u_state]).
    """

    def forward(self, *inputs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs)


class QNet(nn.Module):
    def __init__(self, device: torch.device, hidden=256, max_num=10):
        super(QNet, self).__init__()
        self.device = device
        fea_dim = max_num * 6
        self.conv1 = GCNConv(fea_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.linear = FlattenMlp(22 * (hidden + fea_dim), 4, (hidden, hidden, hidden))
        # self.resnet = models.resnet50(pretrained=False, num_classes=4)
        # self.linear = nn.Linear(machine_num * (hidden + fea_dim), 4)

    def forward(self, state) -> torch.Tensor:
        xx, edge_index, edge_weight = state.x, state.edge_index, state.edge_weight
        x = self.conv1(xx, edge_index, edge_weight)  # 传入卷积层
        x = F.leaky_relu_(x)  # 激活函数
        # x = F.dropout(x, p=0.5, training=self.training)  # dropout层,防止过拟合
        x = self.conv2(x, edge_index, edge_weight)  # 第二层卷积层
        x = F.leaky_relu_(x)  # 激活函数
        x = torch.cat((x, xx), dim=1)
        x = self.linear(x.reshape(int(len(x) / 22), -1))  # fixme
        return x  # F.log_softmax(x, dim=1)  # 将经过两层卷积得到的特征输入log_softmax函数得到概率分布


class PQNet(nn.Module):
    def __init__(self, device: torch.device, hidden=256, max_num=10, machine_num=22):
        super(PQNet, self).__init__()
        self.device = device
        self.fea_dim = max_num * 6
        self.machine_num = machine_num
        # self.conv1 = nn.Conv1d(1, 10, kernel_size=5)  # 定义第一类卷积层，输入通道为1，输出通道为10，卷积核大小为5
        # self.conv2 = nn.Conv1d(10, 20, kernel_size=3)  # 定义第二类卷积层，输入通道为10，输出通道为20，卷积核大小为3
        # self.conv3 = nn.Conv1d(20, 50, kernel_size=2)  # 定义第三类卷积层，输入通道为20，输出通道为50，卷积核大小为3
        # self.conv2_drop = nn.Dropout2d()  # 二维dropout
        # self.conv3_drop = nn.Dropout2d()
        self.fc1 = FlattenMlp(self.fea_dim * machine_num, 4, (hidden, hidden, hidden))

    def forward(self, state) -> torch.Tensor:
        xx, edge_index, edge_weight = state.x, state.edge_index, state.edge_weight
        x = xx.view(-1, self.fea_dim * self.machine_num)
        # # 第一个卷积，池化，激活   #输入batch 1 28 28，卷积输出batch 10 24 24 池化输出batch 10 12 12
        # x = F.relu(F.max_pool1d(self.conv1(x), 2))
        # # 第二个卷积，池化，激活#输入batch 10 12 12，卷积输出 输出batch 20 10 10，池化输出batch 20 5 5
        # x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        # # drop不改变维度随机删去一些数据
        # # 第三个卷积，池化，激活 #输入 batch 20 5 5 卷积输出 batch 50 4 4  池化 batch 50 2 2
        # x = F.relu(F.max_pool1d(self.conv3_drop(self.conv3(x)), 2))
        # x = x.view(-1, 50 * 31)  # view将数据变为200维的数据   batch 50 2 2变为batch 200
        # x = F.relu(self.fc1(x))  # 第一层全连接层 输入维度 batch 200 输出batch 50
        # x = F.dropout(x, training=self.training)  # dropout层，防止过拟合
        x = self.fc1(x)  # 第二层全连接层 输入维度 batch 50 输出batch 10
        return x  # 输出


# class Dueling_DQN(nn.Module):
#     def __init__(self, m_max_num: int, dim_mission_fea: int, dim_mach_fea: int, dim_yard_fea: int, hidden_size: int,
#                  n_layers: int, device: torch.device):
#         super(Dueling_DQN, self).__init__()
#         self.device = device
#         self.m_max_num = m_max_num
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.output_dim = 8
#         model = nn.ModuleDict()
#         model['rnn_station'] = nn.GRU(dim_mach_fea, hidden_size, n_layers, batch_first=True)
#         model['emb_station'] = FlattenMlp(input_dim=hidden_size, hidden_sizes=(64,), output_dim=self.output_dim)
#         model['rnn_cross'] = nn.GRU(dim_mach_fea, hidden_size, n_layers, batch_first=True)
#         model['emb_cross'] = FlattenMlp(input_dim=hidden_size, hidden_sizes=(64,), output_dim=self.output_dim)
#         model['rnn_yard'] = nn.GRU(dim_yard_fea, hidden_size, n_layers, batch_first=True)
#         model['emb_yard'] = FlattenMlp(input_dim=hidden_size, hidden_sizes=(64,), output_dim=self.output_dim)
#         model['adv_mlp'] = FlattenMlp(input_dim=dim_mission_fea + 3 * 8, hidden_sizes=(64,), output_dim=1)
#         model['val_mlp'] = FlattenMlp(input_dim=dim_mission_fea + 2 * 8, hidden_sizes=(64,), output_dim=1)
#
#         self.model = model.to(self.device)
#
#     def forward(self, s_mission, s_station, s_cross, s_yard) -> torch.Tensor:
#         cross_emb = self.forward_pack_sequence(s_cross, 'cross')
#         yard_emb = self.forward_pack_sequence(s_yard, 'yard')
#         ls_adv: List[torch.Tensor] = []
#         for station in s_station:
#             station_emb = self.forward_pack_sequence(station, 'station')
#             station_out = self.model['adv_mlp'](
#                 torch.cat((s_mission / 1000.0, station_emb, cross_emb, yard_emb), dim=1))
#             ls_adv.append(station_out)
#         adv = torch.cat(ls_adv, 1)
#         val = self.model['val_mlp'](torch.cat((s_mission / 1000.0, cross_emb, yard_emb), dim=1)).repeat(1,
#                                                                                                         adv.shape[
#                                                                                                             1])
#         x = adv + val - adv.mean(1, keepdim=True)
#         return x
#
#     def forward_pack_sequence(self, m_p: PackedSequence, m_n: str) -> torch.tensor:
#         m_p, _ = self.model['rnn_' + m_n](m_p.to(self.device))
#         m_s, m_l = pad_packed_sequence(m_p, batch_first=True, total_length=self.m_max_num + 1)
#         m_s = self.model['emb_' + m_n](m_s).squeeze()
#         if m_s.dim() != 3:
#             m_s = m_s.unsqueeze(0)
#         index = m_l.to(self.device) - torch.ones(len(m_l), dtype=torch.int64, device=self.device)
#         index = index.view(-1, 1, 1).repeat(1, 1, self.output_dim)
#         m_emb = torch.gather(m_s, dim=1, index=index)
#         return m_emb.reshape(-1, self.output_dim)


class Actor(nn.Module):
    def __init__(self,
                 crane_num: int,
                 quay_buffer_size: int,
                 dim_attri: int,
                 device: torch.device):
        super(Actor, self).__init__()
        self.device = device
        self.crane_num = crane_num
        self.quay_buffer_size = quay_buffer_size
        model = nn.ModuleDict()
        model['actor'] = FlattenMlp(input_dim=crane_num * quay_buffer_size * dim_attri,
                                    output_dim=crane_num * quay_buffer_size,
                                    hidden_sizes=(256, 256, 256))
        self.model = model.to(self.device)

    def forward(self, x, adjust):
        """

        :param x: buffer_size, quay_num * quay_buffer_size * dim_attri
        :param adjust: batch_size*Cf.QUAY_BUFFER_SIZE 修正最后几个任务不足size补0不选的情况
        :return: buffer_size*1 ,
        """
        x = self.model['actor'](x.reshape(len(x), -1))
        pi = F.softmax(x, dim=1).reshape(-1, self.crane_num, self.quay_buffer_size)
        pi = pi * adjust.repeat(1, len(pi[0, :, :]), 1)
        dist = Categorical(pi)
        action = dist.sample()
        log_pi = dist.log_prob(action)
        return action, log_pi


class Critic(nn.Module):
    def __init__(self,
                 crane_num: int,
                 quay_buffer_size: int,
                 dim_attri: int,
                 device: torch.device):
        super(Critic, self).__init__()
        self.device = device
        self.crane_num = crane_num
        self.quay_buffer_size = quay_buffer_size
        model = nn.ModuleDict()
        model['critic'] = FlattenMlp(input_dim=crane_num * quay_buffer_size * dim_attri, output_dim=1,
                                     hidden_sizes=(256, 256, 256))
        self.model = model.to(self.device)

    def forward(self, x):
        """

        :param x: buffer_size, quay_num * quay_buffer_size * dim_attri
        :return: buffer_size*1
        """
        x = self.model['critic'](x.reshape(len(x), -1))
        return x


class ActorNew(nn.Module):
    def __init__(self, device: torch.device, hidden=256, max_num=10, machine_num=22):
        super(ActorNew, self).__init__()
        self.device = device
        model = nn.ModuleDict()
        fea_dim = max_num * 6
        model['conv1'] = GCNConv(fea_dim, hidden)
        model['conv2'] = GCNConv(hidden, hidden)
        model['linear'] = FlattenMlp(machine_num * (hidden + fea_dim), 1, (hidden, hidden, hidden))
        self.model = model.to(self.device)

    def forward(self, state) -> torch.Tensor:
        xx, edge_index, edge_weight = state.x, state.edge_index, state.edge_weight
        x = self.model['conv1'](xx, edge_index, edge_weight)  # 传入卷积层
        x = F.leaky_relu_(x)  # 激活函数
        x = F.dropout(x, p=0.5, training=self.training)  # dropout层,防止过拟合
        x = self.model['conv2'](x, edge_index, edge_weight)  # 第二层卷积层
        x = F.leaky_relu_(x)  # 激活函数
        x = torch.cat((x, xx), dim=1)
        x = self.model['linear'](x.reshape(int(len(x) / 22), -1))
        y = torch.tanh(x) * 40.0 + torch.tensor(50.0)
        return y


class CriticNew(nn.Module):
    def __init__(self, device: torch.device, hidden=256, max_num=10, machine_num=22):
        super(CriticNew, self).__init__()
        self.device = device
        model = nn.ModuleDict()
        fea_dim = max_num * 6
        model['conv1'] = GCNConv(fea_dim, hidden)
        model['conv2'] = GCNConv(hidden, hidden)
        model['linear1'] = FlattenMlp(machine_num * (hidden + fea_dim) + 2, 1, (hidden, hidden, hidden))
        # model['linear2'] = FlattenMlp(1 + 2, 1, (hidden, hidden))
        self.model = model.to(self.device)

    def forward(self, state, u_act, l_act) -> torch.Tensor:
        xx, edge_index, edge_weight = state.x, state.edge_index, state.edge_weight
        x = self.model['conv1'](xx, edge_index, edge_weight)  # 传入卷积层
        x = F.leaky_relu_(x)  # 激活函数
        x = F.dropout(x, p=0.5, training=self.training)  # dropout层,防止过拟合
        x = self.model['conv2'](x, edge_index, edge_weight)  # 第二层卷积层
        x = F.leaky_relu_(x)  # 激活函数
        x = torch.cat((x.reshape(int(len(x) / 22), -1), xx.reshape(int(len(x) / 22), -1), u_act, l_act), dim=1)
        # TODO l_a gaicheng onehot
        x = self.model['linear1'](x)
        # x = F.leaky_relu_(x)  # 激活函数
        # x = self.model['linear2'](torch.cat((x, u_act, l_act), dim=1))
        return x


if __name__ == "__main__":
    pass
