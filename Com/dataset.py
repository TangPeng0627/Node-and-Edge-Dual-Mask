''' Code adapted from https://github.com/kavehhassani/mvgrl '''
import numpy as np
import torch as th
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv
from torch_geometric.datasets import Amazon, WikiCS
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, AmazonCoBuyComputerDataset
import networkx as nx

from sklearn.preprocessing import MinMaxScaler

from dgl.nn import APPNPConv


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


def process_dataset(name, epsilon=0.01):
    if name == 'cora':
        dataset = CoraGraphDataset(raw_dir='./')
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset(raw_dir='./')
    # elif name == 'computer':
    #     dataset = Amazon(root='./', name='Computers')
    #     data = dataset[0]
    #     graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
    #     graph.ndata['feat'] = data.x
    #     graph.ndata['label'] = data.y
    elif name == 'computer':
        dataset = AmazonCoBuyComputerDataset(raw_dir='./')
    elif name == 'pubmed':
        dataset = PubmedGraphDataset(raw_dir='./')
    elif name == 'wikics':
        dataset = WikiCS(root='./Wiki')
        data = dataset[0]
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    # print(data)
    graph = dataset[0]
    print(graph)
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    # train_mask = graph.ndata.pop('train_mask')
    # val_mask = graph.ndata.pop('val_mask')
    # test_mask = graph.ndata.pop('test_mask')

    # train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    # val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    # test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()


    graph = graph.add_self_loop()

    return graph, feat, label


def process_dataset_appnp(epsilon):
    k = 20
    alpha = 0.2
    dataset = PubmedGraphDataset()
    graph = dataset[0]
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    appnp = APPNPConv(k, alpha)
    id = th.eye(graph.number_of_nodes()).float()
    diff_adj = appnp(graph.add_self_loop(), id).numpy()

    diff_adj[diff_adj < epsilon] = 0
    scaler = MinMaxScaler()
    scaler.fit(diff_adj)
    diff_adj = scaler.transform(diff_adj)
    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    diff_graph = dgl.graph(diff_edges)

    return graph, diff_graph, feat, label, train_mask, val_mask, test_mask, diff_weight


# process_dataset('wikics')
