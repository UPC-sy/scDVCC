import os
import torch
import numpy as np
import scanpy as sc
from src.argument import printConfig
from torch_geometric.loader import HGTLoader
from sklearn.metrics.cluster import adjusted_rand_score
from src.data import read_data, normalize, construct_graph_knn, construct_graph_dm

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
        adata = read_data(self.args.name)
        self.adata = normalize(adata)
        adata_knn = self.adata.copy()
        adata_dm = self.adata.copy()
        sc.tl.pca(adata_knn, svd_solver='arpack', n_comps=50)
        print("Building KNN graph...")
        self.c_g_graph_knn = construct_graph_knn(adata_knn, num_cell=adata_knn.n_obs, num_gene=adata_knn.n_vars)
        print("Building Diffusion Map graph...")
        self.c_g_graph_dm = construct_graph_dm(adata_dm, num_cell=adata_dm.n_obs, num_gene=adata_dm.n_vars)

        input_idx = torch.arange(self.adata.n_obs)

        def create_generator(seed):
            generator = torch.Generator()
            generator.manual_seed(seed)
            return generator

        sampler1 = torch.utils.data.RandomSampler(input_idx, generator=create_generator(seed))
        sampler2 = torch.utils.data.RandomSampler(input_idx, generator=create_generator(seed + 1))  # 不同种子保证采样差异

        cell_nodes = ('cell', torch.ones(self.adata.n_obs).bool())
        NS = eval(self.args.ns)

        self.train_loader1 = HGTLoader(
            self.c_g_graph_knn,
            num_samples=NS,
            sampler=sampler1,
            input_nodes=cell_nodes,
            batch_size=self.args.batch_size
        )

        self.train_loader2 = HGTLoader(
            self.c_g_graph_dm,
            num_samples=NS,
            sampler=sampler2,
            input_nodes=cell_nodes,
            batch_size=self.args.batch_size
        )


        self.eval_loader = HGTLoader(
            self.c_g_graph_knn,
            num_samples=NS,
            shuffle=False,
            input_nodes=cell_nodes,
            batch_size=self.args.batch_size
        )

    def Pretrain_Evaluate_Convergence(self, epoch):
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
        flag = 0
        num = self.adata.X.shape[0]

        delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
        self.y_pred_last = self.y_pred
        if delta_label < self.args.tol:
            print('delta_label ', delta_label, '< tol ', self.args.tol)
            print("Reach tolerance threshold. Stopping fine-tuning.")
            flag = 1

        return flag