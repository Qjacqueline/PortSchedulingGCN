# -*- coding: utf-8 -*-
# @Time    : 2022/7/10 11:13 AM
# @Author  : JacQ
# @File    : tst.py
# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
#
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# for batch in loader:
#     a = 1

from torch_geometric.data import Dataset, Data
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class Buffer(Dataset):
    def __init__(self):
        super(Buffer, self).__init__()
        self.vector_ls = []
        self.data_ls = []

    def __getitem__(self, item):
        return self.data_ls[item], self.vector_ls[item]

    def __len__(self):
        return len(self.data_ls)

    def add(self, v, g):
        self.vector_ls.append(v)
        self.data_ls.append(g)


v1 = torch.rand(2, 1)
v2 = torch.rand(2, 1)
g1 = Data(x=torch.rand(4, 10), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
g2 = Data(x=torch.rand(4, 10), edge_index=torch.tensor([[0, 0, 0], [1, 2, 3]]))

buffer = Buffer()
buffer.add(v1, g1)
buffer.add(v2, g2)
buffer.add(v1, g1)
buffer.add(v2, g2)
loader = DataLoader(buffer, batch_size=2)
with tqdm(loader, desc=f'epoch{1}', ncols=100) as pbar:
    for batch in pbar:
        a = 1
