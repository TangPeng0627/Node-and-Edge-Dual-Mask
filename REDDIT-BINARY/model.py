import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm
from dgl import DropEdge, AddEdge


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def mask_node(g, mask_rate=0.5, noise=0.05):
    num_nodes = g.num_nodes()
    perm = torch.randperm(num_nodes, device=g.device)
    num_mask_nodes = int(mask_rate * num_nodes)

    # random masking
    # num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]
    keep_nodes = perm[num_mask_nodes:]

    num_noise_nodes = int(noise * num_mask_nodes)
    # 长度为打乱1354
    perm_mask = torch.randperm(num_mask_nodes, device=g.device)
    # 在1354的基础上随机选择1258
    token_nodes = mask_nodes[perm_mask[: int((1 - noise) * num_mask_nodes)]]
    # 1354减去1258剩下的67个
    noise_nodes = mask_nodes[perm_mask[-int(noise * num_mask_nodes):]]
    # 2708中随机选择67个替换上面的67个
    noise_to_be_chosen = torch.randperm(num_nodes, device=g.device)[:num_noise_nodes]

    # out_x = data.x.clone()
    # out_x[token_nodes] = 0.0
    # out_x[noise_nodes] = data.x[noise_to_be_chosen]

    return token_nodes, noise_nodes, noise_to_be_chosen, mask_nodes

def mask_edge(g, mask_rate=0.5, noise=0.05):
    new_g = g.clone()
    new_g = new_g.remove_self_loop()
    drop_edge = DropEdge(mask_rate)
    new_g = drop_edge(new_g)
    add_edge = AddEdge(noise)
    new_g = add_edge(new_g)
    new_g = new_g.add_self_loop()

    return new_g


class Encoder(nn.Module):
    def __init__(self, in_hidden, out_hidden, n, p):
        super(Encoder, self).__init__()
        self.n = n
        self.conv1 = GraphConv(in_hidden, out_hidden, norm='both',
                               bias=True, activation=nn.PReLU())
        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.actions = nn.ModuleList()

        self.layers.append(self.conv1)
        self.bn.append(BatchNorm(out_hidden))
        self.actions.append(nn.PReLU())

        for _ in range(1, self.n):
            self.layers.append(GraphConv(out_hidden, out_hidden, norm='both',
                               bias=True, activation=nn.PReLU()))
            self.bn.append(BatchNorm(out_hidden))
            self.actions.append(nn.PReLU())
        self.dp = nn.Dropout(p)
        self.pooling = SumPooling()

    def forward(self, graph, heat):
        x = self.dp(heat)
        h = self.conv1(graph, x)
        h = self.bn[0](h)
        h = self.actions[0](h)
        gh = self.pooling(graph, h)

        for i in range(1, self.n):
            h = self.layers[i](graph, h)
            h = self.bn[i](h)
            h = self.actions[i](h)
            #gh = torch.cat((gh, self.pooling(graph, h)), -1)
            gh += self.pooling(graph, h)

        return h, gh


class Decoder1(nn.Module):
    def __init__(self, in_hidden, out_hidden, n, p):
        super(Decoder1, self).__init__()
        self.n = n
        self.conv1 = GraphConv(in_hidden, out_hidden, norm='none',
                               bias=True, activation=nn.PReLU())
        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.actions = nn.ModuleList()

        self.bn.append(BatchNorm(out_hidden))
        self.layers.append(self.conv1)
        self.actions.append(nn.PReLU())

        for _ in range(1, self.n):
            self.layers.append(GraphConv(out_hidden, out_hidden, norm='none',
                                         bias=True, activation=nn.PReLU()))
            self.bn.append(BatchNorm(out_hidden))
            self.actions.append(nn.PReLU())
        self.dp = nn.Dropout(p)

    def forward(self, graph, heat):
        x = self.dp(heat)
        h = self.conv1(graph, x)
        h = self.bn[0](h)
        h = self.actions[0](h)

        for i in range(1, self.n):
            h = self.layers[i](graph, h)
            h = self.bn[i](h)
            h = self.actions[i](h)

        return h

class Decoder2(nn.Module):
    def __init__(self, in_hidden, out_hidden, n, p):
        super(Decoder2, self).__init__()
        self.n = n
        self.conv1 = GraphConv(in_hidden, out_hidden, norm='none',
                               bias=True, activation=nn.PReLU())
        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.actions = nn.ModuleList()

        self.bn.append(BatchNorm(out_hidden))
        self.layers.append(self.conv1)
        self.actions.append(nn.PReLU())

        for _ in range(1, self.n):
            self.layers.append(GraphConv(out_hidden, out_hidden, norm='none',
                                         bias=True, activation=nn.PReLU()))
            self.bn.append(BatchNorm(out_hidden))
            self.actions.append(nn.PReLU())
        self.dp = nn.Dropout(p)

    def forward(self, graph, heat):
        x = self.dp(heat)
        h = self.conv1(graph, x)
        h = self.bn[0](h)
        h = self.actions[0](h)

        for i in range(1, self.n):
            h = self.layers[i](graph, h)
            h = self.bn[i](h)
            h = self.actions[i](h)

        return F.normalize(h)


class MG(nn.Module):
    def __init__(self, in_hidden, out_hidden, beta, beta1, rate_node,
                 rate_edge, alpha, n1, n2, n3, p1, p2, p3, noise_node, noise_edge):
        super(MG, self).__init__()
        self.enc = Encoder(in_hidden, out_hidden, n1, p1)
        self.dec1 = Decoder1(out_hidden, in_hidden, n2, p2)
        self.dec2 = Decoder2(out_hidden, in_hidden, n3, p3)
        self.rate_node = rate_node
        self.rate_edge = rate_edge
        self.alpha = alpha
        self.noise_node = noise_node
        self.noise_edge = noise_edge
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_hidden))
        self.enc_mask_token1 = nn.Parameter(torch.zeros(1, in_hidden))
        # self.encoder_to_decoder = nn.Linear(out_hidden, out_hidden, bias=False)
        self.criterion = self.setup_loss_fn("sce", beta)
        self.criterion1 = self.setup_loss_fn("mse", beta1)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, graph, matrix):

        token_nodes, noise_nodes, \
        noise_to_be_chosen1, mask_nodes1 = mask_node(graph, mask_rate=self.rate_node,
                                                noise=self.noise_node)
        new_graph = mask_edge(graph, mask_rate=self.rate_edge, noise=self.noise_edge)
        x = graph.ndata["attr"].clone()
        x[token_nodes] = 0.0
        if self.noise_node > 0:
            x[noise_nodes] = graph.ndata["attr"][noise_to_be_chosen1]
        else:
            x[noise_nodes] = 0.0
        x[token_nodes] += self.enc_mask_token1

        h1, gh1 = self.enc(graph, x)
        h2, gh2 = self.enc(new_graph, graph.ndata["attr"])

        h1 = self.dec1(graph, h1)
        h2 = self.dec2(new_graph, h2)
        new_matrix = torch.matmul(h2, h2.T)

        loss1 = self.criterion(h1, graph.ndata["attr"])
        loss2 = self.criterion1(new_matrix, matrix)

        return self.alpha * loss1 + loss2 * (1 - self.alpha)

    def get_embed(self, graph):
        h1, gh1 = self.enc(graph, graph.ndata['attr'])

        return gh1
