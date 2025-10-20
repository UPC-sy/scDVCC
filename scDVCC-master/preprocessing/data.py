import torch
import scanpy as sc
import numpy as np
from torch_geometric.data import HeteroData
import scanpy as sc
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csgraph

def read_data(name):

    path = f'./data/{name}.h5'
    adata = sc.read(path)

    return adata

def normalize(adata, HVG=0.2, filter_min_counts=True, size_factors=True, logtrans_input=True, normalize_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    n = int(adata.X.shape[1] * HVG)
    hvg_gene_idx = np.argsort(adata.X.var(axis=0))[-n:]
    adata = adata[:,hvg_gene_idx]
    adata.raw = adata.copy()
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if normalize_input:
        sc.pp.scale(adata)

    return adata




def build_diffusion_map_graph(adata, n_components=10, sigma=10, alpha=0.5, use_pca=True):

    if use_pca and 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
    X = adata.obsm['X_pca'] if use_pca else adata.X
    affinity_matrix = rbf_kernel(X, gamma=1 / (2 * sigma ** 2))
    if alpha > 0:
        D = np.diag(affinity_matrix.sum(axis=1) ** -alpha)
        affinity_matrix = D @ affinity_matrix @ D
    P = csgraph.laplacian(affinity_matrix, normed=True)
    from scipy.sparse.linalg import eigsh
    eigenvalues, eigenvectors = eigsh(P, k=n_components, which='LM')
    adata.obsm['X_diffmap'] = eigenvectors[:, ::-1]
    adata.uns['diffusion_components'] = eigenvalues[::-1]
    return adata

def build_knn_graph(adata, n_neighbors=15, metric='euclidean', use_pca=True):
    if use_pca and 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

    X = adata.obsm['X_pca'] if use_pca else adata.X
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn_model.fit(X)
    distances, indices = nn_model.kneighbors()

    n_cells = X.shape[0]
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.flatten()
    data = np.ones_like(rows)

    knn_adj = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    knn_adj = knn_adj.maximum(knn_adj.T)
    adata.obsp['knn_adj'] = knn_adj
    return adata

def construct_graph(Unnormalized_featureMatrix, featureMatrix, num_cell, num_gene):
    X = torch.tensor(Unnormalized_featureMatrix)
    cells, genes = torch.nonzero(X, as_tuple=True)
    data = HeteroData()
    data['cell'].x = torch.tensor(featureMatrix)
    data['cell', 'to', 'gene'].edge_index = torch.stack((cells,genes))
    data['gene', 'to', 'cell'].edge_index = torch.stack((genes,cells))
    data['cell'].num_nodes = num_cell
    data['gene'].num_nodes = num_gene
    data['cell']['n_id'] = torch.arange(num_cell)
    data['gene']['n_id'] = torch.arange(num_gene)
    return data

