import torch

class Hete_DropFeatures:
    r"""Drops node features with probability p.以一定的概率p丢弃节点特征"""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p
        self.eps = 1e-7

    def __call__(self, data):
        if self.p == 0.0:
            return data

        for node_type in data.metadata()[0]:#代码遍历数据中的每一种节点类型。这里的data是一个包含多种类型节点的异构图，metadata()[0]返回的是图中所有节点类型的列表。
            #对于每一种节点类型，代码生成一个与该类型节点特征维度相同的随机张量drop_mask。这个张量的每个元素都是一个在0到1之间的随机数，然后与丢弃概率self.p进行比较，
            # 如果随机数小于self.p，则对应位置的值为True，否则为False。这样，drop_mask就成了一个布尔掩码，用于标记哪些特征需要被丢弃。
            drop_mask = torch.empty((data[node_type].x.size(1),), dtype=torch.float32, device=data[node_type].x.device).uniform_(0, 1) < self.p
            #对于每一种节点类型，代码将特征矩阵data[node_type].x中对应drop_mask值为True的位置的特征值设置为0，从而实现了特征的丢弃。
            data[node_type].x[:, drop_mask] = 0
        
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)

