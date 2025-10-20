import os
import torch
import numpy as np
from argument import printConfig
from torch_geometric.loader import HGTLoader
from sklearn.metrics.cluster import adjusted_rand_score
from data import read_data, normalize, construct_graph

class embedder:
    def __init__(self, args):
        self.args = args
        printConfig(args)

        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)
        
        self.model_path = os.path.join('./weights', f'{str(self.args.recon)}_{str(self.args.name)}.pt')
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
    def _init_dataset(self, seed):
        #这段代码的主要目的是初始化数据集。
        # 首先，它读取数据并进行标准化处理。
        # 然后，它构建了一个图结构。
        # 接下来，它创建了两个随机数生成器和对应的随机采样器。
        # 最后，它创建了两个训练数据加载器和一个评估数据加载器。
        adata = read_data(self.args.name)
        # print(adata.obs.columns)
        # 输出可能：['n_genes', 'percent_mito', 'cell_type', 'batch']
        # 对数据进行标准化处理
        self.adata = normalize(adata, HVG=self.args.HVG, size_factors=self.args.sf, logtrans_input=self.args.log, normalize_input=self.args.normal)
        # 构建图结构
        self.c_g_graph = construct_graph(self.adata.raw.X, self.adata.X, self.adata.n_obs, self.adata.n_vars)
        # 创建一个包含数据观察值数量的索引数组
        input_idx = torch.arange(self.adata.n_obs)
        # 创建第一个随机数生成器，并设置种子
        generator1 = torch.Generator()
        generator1.manual_seed(seed)
        # 创建第一个随机采样器
        sampler1 = torch.utils.data.RandomSampler(input_idx, generator=generator1)
        # 创建第二个随机数生成器，并设置种子
        generator2 = torch.Generator()
        generator2.manual_seed(seed)
        # 创建第二个随机采样器
        sampler2 = torch.utils.data.RandomSampler(input_idx, generator=generator2)
        # 解析参数中的样本数量
        NS = eval(self.args.ns)
        # 创建一个包含'cell'字符串和布尔张量的元组
        cell_nodes = ('cell', torch.ones(self.adata.n_obs).bool())
        # 创建第一个训练数据加载器
        self.train_loader1 = HGTLoader(self.c_g_graph, num_samples=NS,  sampler=sampler1, input_nodes=cell_nodes, batch_size=self.args.batch_size)
        # 创建第二个训练数据加载器
        self.train_loader2 = HGTLoader(self.c_g_graph, num_samples=NS, sampler=sampler2, input_nodes=cell_nodes, batch_size=self.args.batch_size)
        # 创建评估数据加载器
        self.eval_loader = HGTLoader(self.c_g_graph, num_samples=NS, shuffle=False, input_nodes=cell_nodes, batch_size=self.args.batch_size)
    def Pretrain_Evaluate_Convergence(self, epoch):
        #在预训练阶段评估模型的收敛情况
        flag=0
        self.model.eval()
        cell_rep = self.model.predict_full_cell_rep(self.eval_loader, self.gene_embedding)
        y_pred = self.model.predict_celltype(cell_rep)
        
        if epoch == self.args.warmup+1:
            self.old_celltype_result = y_pred
        else:
            ari = adjusted_rand_score(self.old_celltype_result, y_pred)
            self.old_celltype_result = y_pred
            if ari > self.args.r:
                flag=1
                print("Reach tolerance threshold. Stopping pre-training.")

        return flag

    def Fine_Evaluate_Convergence(self):
        #微调阶段评估模型的收敛情况
        flag = 0
        num = self.adata.X.shape[0]

        delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
        self.y_pred_last = self.y_pred
        if delta_label < self.args.tol:
            print('delta_label ', delta_label, '< tol ', self.args.tol)
            print("Reach tolerance threshold. Stopping fine-tuning.")
            flag = 1

        return flag