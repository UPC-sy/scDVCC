
import scanpy as sc
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import torch
from torch_geometric.data import HeteroData
import numpy as np

def read_data(name):#读取数据

    path = f'./data/{name}.h5'
    adata = sc.read(path)

    return adata


# def normalize(adata,min_genes=200,min_cells=3,target_sum=1e4,n_top_genes=2000,max_value=10):
#     """
#     优化后的单细胞预处理流程 (兼容Scanpy生态系统)
#
#     参数说明：
#     - adata: 原始AnnData对象
#     - min_genes: 细胞保留的最小基因数
#     - min_cells: 基因保留的最小细胞数
#     - target_sum: 表达量归一化的目标总和
#     - n_top_genes: 选择的高变基因数量
#     - max_value: 缩放后的最大值限制
#     """
#
#     # --------------------------
#     # 1. 质量控制增强版
#     # --------------------------
#     # 标记线粒体基因 (关键改进)
#     adata.var['mt'] = adata.var_names.str.startswith('MT-')  # 人类
#     # 对于小鼠数据使用：adata.var['mt'] = adata.var_names.str.startswith('mt-')
#
#     # 计算QC指标 (新增pct_counts_mt)
#     sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, inplace=True)
#
#     # 细胞过滤 (增强鲁棒性)
#     sc.pp.filter_cells(adata, min_genes=min_genes)
#     sc.pp.filter_genes(adata, min_cells=min_cells)
#
#     # --------------------------
#     # 2. 数据标准化与预处理
#     # --------------------------
#     # 原始数据备份 (兼容性保障)
#     adata.raw = adata
#
#     # 表达量归一化 (优化参数)
#     sc.pp.normalize_total(adata, target_sum=target_sum)
#     sc.pp.log1p(adata)  # 保持对数转换
#
#     # 特征缩放 (关键改进：增加max_value限制)
#     sc.pp.scale(adata, max_value=max_value, zero_center=True)
#
#     # --------------------------
#     # 3. 高变基因选择标准化
#     # --------------------------
#     # 使用Scanpy标准方法 (替换原始手动选择)
#     sc.pp.highly_variable_genes(
#         adata,
#         n_top_genes=n_top_genes,
#         flavor='seurat',
#         subset=True
#     )
#
#     # --------------------------
#     # 4. 降维处理 (可选但推荐)
#     # --------------------------
#     # PCA降维 (优化随机种子设置)
#     sc.tl.pca(adata, svd_solver='arpack', random_state=42)
#
#     # 邻居图构建 (兼容性增强)
#     sc.pp.neighbors(adata, n_pcs=30, n_neighbors=20)
#
#     # --------------------------
#     # 5. 元数据增强 (新增字段)
#     # --------------------------
#     # 添加预处理版本标记 (可追溯性改进)
#     adata.uns['preprocessing_version'] = {
#         'pipeline_version': '2.1',
#         'parameters': {
#             'min_genes': min_genes,
#             'n_top_genes': n_top_genes,
#             'scaling_max': max_value
#         }
#     }
#
#     return adata



def normalize(adata, HVG=0.2, filter_min_counts=True, size_factors=True, logtrans_input=True, normalize_input=True):
    # 进行预处理和标准化
    # 如果filter_min_counts为True，那么将会过滤掉基因和细胞的最小计数
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    # 计算高变异基因的数量
    n = int(adata.X.shape[1] * HVG)
    # 获取高变异基因的索引
    hvg_gene_idx = np.argsort(adata.X.var(axis=0))[-n:]
    # 只保留高变异基因
    adata = adata[:,hvg_gene_idx]
    # 创建原始数据的副本
    adata.raw = adata.copy()
    # 如果size_factors为True，那么将会对每个细胞进行标准化
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        # 计算大小因子
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    # 如果logtrans_input为True，那么将会对数据进行对数转换
    if logtrans_input:
        sc.pp.log1p(adata)

    # 如果normalize_input为True，那么将会对数据进行标准化
    if normalize_input:
        sc.pp.scale(adata)

    # 返回预处理和标准化后的数据
    return adata


def construct_graph_dm(adata, num_cell, num_gene, n_components=50, sigma=10, alpha=0.5):
    # 确保 adata 是 AnnData 对象，执行 Diffusion Map 操作
    assert hasattr(adata, "obsm"), "Input must be an AnnData object, not HeteroData!"

    # Step 1: 计算高斯相似性核 (rbf kernel)
    affinity_matrix = rbf_kernel(adata.X, gamma=1 / (2 * sigma ** 2))  # 高斯核

    # Step 2: 标准化得到马尔可夫转移矩阵
    if alpha > 0:
        D = np.diag(affinity_matrix.sum(axis=1) ** -alpha)
        affinity_matrix = D @ affinity_matrix @ D

    # Step 3: 计算扩散图的拉普拉斯矩阵
    P = csgraph.laplacian(affinity_matrix, normed=True)  # 扩散操作

    # Step 4: 计算特征值和特征向量
    eigenvalues, eigenvectors = eigsh(P, k=n_components, which='LM')

    # Step 5: 将特征向量存储到 adata.obsm 中（Diffusion Map的嵌入空间）
    adata.obsm['X_diffmap'] = eigenvectors[:, ::-1]  # 按特征值降序排列

    # 获取Diffusion Map特征
    featureMatrix = adata.obsm['X_diffmap']
    Unnormalized_featureMatrix = adata.raw.X if adata.raw is not None else adata.X

    if not isinstance(Unnormalized_featureMatrix, np.ndarray):
        raw_matrix = Unnormalized_featureMatrix.toarray()
    else:
        raw_matrix = Unnormalized_featureMatrix

    X = torch.tensor(raw_matrix)
    cells, genes = torch.nonzero(X, as_tuple=True)

    data = HeteroData()
    data['cell'].x = torch.tensor(featureMatrix.copy(), dtype=torch.float32)
    data['gene'].x = torch.eye(num_gene, dtype=torch.float32)

    data['cell', 'to', 'gene'].edge_index = torch.stack((cells, genes))
    data['gene', 'to', 'cell'].edge_index = torch.stack((genes, cells))

    data['cell'].num_nodes = num_cell
    data['gene'].num_nodes = num_gene
    data['cell']['n_id'] = torch.arange(num_cell)
    data['gene']['n_id'] = torch.arange(num_gene)

    return data

# def construct_graph_dm(Unnormalized_featureMatrix, featureMatrix, num_cell, num_gene, k=5):
#     # 数据完整性检查
#     if num_cell <= 0 or num_gene <= 0 or featureMatrix.shape[0] != num_cell or featureMatrix.shape[1] != num_gene:
#         raise ValueError("Invalid cell or gene count. Please check the input data.")
#
#     # 使用Diffusion Maps (DM) 算法计算相似度矩阵
#     distance_matrix = pairwise_distances(featureMatrix)
#
#     # 计算相似度矩阵的扩散核
#     epsilon = np.median(distance_matrix)
#     affinity_matrix = np.exp(-distance_matrix ** 2 / (2 * epsilon ** 2))
#
#     # 对于每个细胞节点，找到最近的k个基因节点
#     cell_indices = []
#     gene_indices = []
#
#     for i in range(num_cell):
#         gene_distances = affinity_matrix[i, :]
#         nearest_genes = np.argsort(gene_distances)[-k:]
#         cell_indices.extend([i] * k)
#         gene_indices.extend(nearest_genes)
#
#     # 转换为张量
#     cell_indices = torch.tensor(np.array(cell_indices), dtype=torch.long)
#     gene_indices = torch.tensor(np.array(gene_indices), dtype=torch.long)
#
#     # 创建一个异构图数据对象
#     data = HeteroData()
#     data['cell'].x = torch.tensor(np.array(featureMatrix), dtype=torch.float32)
#     data['gene'].x = torch.tensor(np.array(featureMatrix.T), dtype=torch.float32)
#
#     # 设置细胞到基因的边和基因到细胞的边
#     data['cell', 'to', 'gene'].edge_index = torch.stack((cell_indices, gene_indices))
#     data['gene', 'to', 'cell'].edge_index = torch.stack((gene_indices, cell_indices))
#
#     # 设置节点的数量和ID
#     data['cell'].num_nodes = num_cell
#     data['gene'].num_nodes = num_gene
#     data['cell']['n_id'] = torch.arange(num_cell)
#     data['gene']['n_id'] = torch.arange(num_gene)
#
#     return data

def construct_graph_knn(adata, num_cell, num_gene, n_neighbors=15, use_pca=True):
    # 确保有X_pca（否则 KNN 构图会失败）
    if use_pca and 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

    featureMatrix = adata.obsm['X_pca'] if use_pca else adata.X
    Unnormalized_featureMatrix = adata.raw.X if adata.raw is not None else adata.X

    if not isinstance(Unnormalized_featureMatrix, np.ndarray):
        raw_matrix = Unnormalized_featureMatrix.toarray()
    else:
        raw_matrix = Unnormalized_featureMatrix

    X = torch.tensor(raw_matrix)
    cells, genes = torch.nonzero(X, as_tuple=True)

    data = HeteroData()
    # 强制创建副本，避免负步长问题
    data['cell'].x = torch.tensor(featureMatrix.copy(), dtype=torch.float32)
    data['gene'].x = torch.eye(num_gene, dtype=torch.float32)

    data['cell', 'to', 'gene'].edge_index = torch.stack((cells, genes))
    data['gene', 'to', 'cell'].edge_index = torch.stack((genes, cells))

    data['cell'].num_nodes = num_cell
    data['gene'].num_nodes = num_gene
    data['cell']['n_id'] = torch.arange(num_cell)
    data['gene']['n_id'] = torch.arange(num_gene)

    return data




def construct_graph(Unnormalized_featureMatrix, featureMatrix, num_cell, num_gene):
    """保持原接口兼容性"""
    # 原始实现保持不变
    data = HeteroData()
    X = torch.tensor(Unnormalized_featureMatrix)
    cells, genes = torch.nonzero(X, as_tuple=True)

    data['cell'].x = torch.tensor(featureMatrix)
    data['gene'].x = torch.eye(num_gene)  # 基因特征使用单位矩阵

    data['cell', 'to', 'gene'].edge_index = torch.stack((cells, genes))
    data['gene', 'to', 'cell'].edge_index = torch.stack((genes, cells))

    data['cell'].num_nodes = num_cell
    data['gene'].num_nodes = num_gene
    data['cell']['n_id'] = torch.arange(num_cell)
    data['gene']['n_id'] = torch.arange(num_gene)

    return data
# def build_knn_graph(adata, n_neighbors=15, metric='euclidean', use_pca=True):
#     """
#     构建KNN邻接图
#     参数:
#         adata: 预处理后的AnnData对象
#         n_neighbors: 每个细胞的邻居数
#         metric: 距离度量方式 ('euclidean', 'cosine', 'correlation'等)
#         use_pca: 是否使用PCA降维加速计算
#     """
#     # 降维加速计算（可选）
#     if use_pca and 'X_pca' not in adata.obsm:
#         sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
#
#     # 使用PCA特征或原始表达矩阵
#     X = adata.obsm['X_pca'] if use_pca else adata.X
#
#     # 计算KNN
#     nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
#     nn_model.fit(X)
#     distances, indices = nn_model.kneighbors()
#
#     # 构建稀疏邻接矩阵
#     n_cells = X.shape[0]
#     rows = np.repeat(np.arange(n_cells), n_neighbors)
#     cols = indices.flatten()
#     data = np.ones_like(rows)  # 未加权邻接
#
#     knn_adj = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
#
#     # 对称化邻接矩阵（可选）
#     knn_adj = knn_adj.maximum(knn_adj.T)  # 保证对称性
#
#     adata.obsp['knn_adj'] = knn_adj
#     return adata
#
# def build_diffusion_map_graph(adata, n_components=10, sigma=10, alpha=0.5, use_pca=True):
#     """
#     构建扩散映射邻接图
#     参数:
#         sigma: 高斯核带宽
#         alpha: 扩散过程的标准化参数 (0-1)
#     """
#     # 降维加速计算（可选）
#     if use_pca and 'X_pca' not in adata.obsm:
#         sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
#
#     X = adata.obsm['X_pca'] if use_pca else adata.X
#
#     # Step 1: 构建高斯相似性核
#     affinity_matrix = rbf_kernel(X, gamma=1 / (2 * sigma ** 2))  # 高斯核
#
#     # Step 2: 标准化得到马尔可夫转移矩阵
#     if alpha > 0:
#         D = np.diag(affinity_matrix.sum(axis=1) ** -alpha)
#         affinity_matrix = D @ affinity_matrix @ D
#
#     # Step 3: 行归一化
#     P = csgraph.laplacian(affinity_matrix, normed=True)  # 扩散操作
#
#     # Step 4: 保留主要扩散成分
#     from scipy.sparse.linalg import eigsh
#     eigenvalues, eigenvectors = eigsh(P, k=n_components, which='LM')
#
#     # 存储结果
#     adata.obsm['X_diffmap'] = eigenvectors[:, ::-1]  # 按特征值降序排列
#     adata.uns['diffusion_components'] = eigenvalues[::-1]
#
#     return adata
#
# def construct_graph(Unnormalized_featureMatrix, featureMatrix, num_cell, num_gene):#根据输入的特征矩阵和节点数量，构造一个异构图的
#     # 将未标准化的特征矩阵转换为张量
#     X = torch.tensor(Unnormalized_featureMatrix)
#     # 获取非零元素的索引，这些索引对应于细胞和基因
#     cells, genes = torch.nonzero(X, as_tuple=True)
#     # 创建一个异构图数据对象
#     data = HeteroData()
#     # 将标准化的特征矩阵转换为张量，并设置为细胞节点的特征
#     data['cell'].x = torch.tensor(featureMatrix)
#     # 设置细胞节点到基因节点的边，边的索引由细胞和基因的索引构成
#     data['cell', 'to', 'gene'].edge_index = torch.stack((cells,genes))
#     # 设置基因节点到细胞节点的边，边的索引由基因和细胞的索引构成
#     data['gene', 'to', 'cell'].edge_index = torch.stack((genes,cells))
#     # 设置细胞节点和基因节点的数量
#     data['cell'].num_nodes = num_cell
#     data['gene'].num_nodes = num_gene
#     # 设置细胞节点和基因节点的ID，ID的范围是从0到节点数量-1
#     data['cell']['n_id'] = torch.arange(num_cell)
#     data['gene']['n_id'] = torch.arange(num_gene)
#     # 返回构造的异构图
#     return data
