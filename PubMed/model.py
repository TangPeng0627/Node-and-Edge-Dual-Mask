import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv
from dgl import DropEdge, AddEdge
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def mask_node(g, x, mask_rate=0.5, noise=0.05):
    num_nodes = g.num_nodes()
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)

    # random masking
    # num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]
    keep_nodes = perm[num_mask_nodes:]

    num_noise_nodes = int(noise * num_mask_nodes)
    # 长度为打乱1354
    perm_mask = torch.randperm(num_mask_nodes, device=x.device)
    # 在1354的基础上随机选择1258
    token_nodes = mask_nodes[perm_mask[: int((1 - noise) * num_mask_nodes)]]
    # 1354减去1258剩下的67个
    noise_nodes = mask_nodes[perm_mask[-int(noise * num_mask_nodes):]]
    # 2708中随机选择67个替换上面的67个
    noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

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
    def __init__(self, in_hidden, out_hidden, p1):
        super(Encoder, self).__init__()
        self.conv1 = GraphConv(in_hidden, out_hidden, norm='both',
                              bias=True, activation=nn.PReLU())
        self.dp1 = nn.Dropout(p1)
        #self.conv2 = GraphConv(out_hidden, out_hidden, norm='both',
        #                      bias=True, activation=nn.PReLU())
        # self.conv1 = GATConv(in_hidden, out_hidden, num_heads=4, attn_drop=0.1,
        #                      feat_drop=0.0, bias=True, activation=nn.PReLU())
        self.act = nn.PReLU()
        # self.dp2 = nn.Dropout(p2)
        self.bn = BatchNorm(out_hidden)
        self.ln = LayerNorm(out_hidden)

    def forward(self, graph, feat):
        x = self.dp1(feat)

        h = self.conv1(graph, x)
        #print(h.size().squeeze(1))
        h = self.bn(h)
        h = self.act(h)

        #h = self.conv2(graph, h)
        #h = self.bn(h)
        #h = self.act(h)

        return h


class Decoder1(nn.Module):
    def __init__(self, in_hidden, out_hidden, p1):
        super(Decoder1, self).__init__()
        self.conv1 = GraphConv(in_hidden, out_hidden, norm='none',
                              bias=True, activation=nn.PReLU())
        self.dp1 = nn.Dropout(p1)
        self.act = nn.PReLU()
        self.bn = BatchNorm(out_hidden)
        self.ln = LayerNorm(out_hidden)

    def forward(self, graph, feat):
        x = self.dp1(feat)
        h = self.conv1(graph, x)
        h = self.bn(h)
        h = self.act(h)

        return h

class Decoder2(nn.Module):
    def __init__(self, in_hidden, out_hidden, p1):
        super(Decoder2, self).__init__()
        self.conv1 = GraphConv(in_hidden, out_hidden, norm='none',
                              bias=True, activation=nn.PReLU())
        self.dp1 = nn.Dropout(p1)
        self.act = nn.PReLU()
        self.bn = BatchNorm(out_hidden)
        self.ln = LayerNorm(out_hidden)

    def forward(self, graph, feat):
        x = self.dp1(feat)
        h = self.conv1(graph, x)
        h = self.bn(h)
        h = self.act(h)


        return F.normalize(h)


class MG(nn.Module):
    def __init__(self, in_hidden, out_hidden, p1, p2, p3, beta, beta1, rate_node,
                 rate_edge, alpha, noise_node, noise_edge):
        super(MG, self).__init__()

        self.enc = Encoder(in_hidden, out_hidden, p1)
        self.dec1 = Decoder1(out_hidden, in_hidden, p2)
        self.dec2 = Decoder2(out_hidden, in_hidden, p3)
        self.rate_node = rate_node
        self.rate_edge = rate_edge
        self.alpha = alpha
        self.noise_node = noise_node
        self.noise_edge = noise_edge
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

    def forward(self, graph, feat, matrix):

        token_nodes, noise_nodes, \
        noise_to_be_chosen, mask_nodes = mask_node(graph, feat, mask_rate=self.rate_node, noise=self.noise_node)
        new_graph = mask_edge(graph, mask_rate=self.rate_edge, noise=self.noise_edge)

        x = feat.clone()
        x[token_nodes] = 0.0
        if self.noise_node > 0:
            x[noise_nodes] = feat[noise_to_be_chosen]
        else:
            x[noise_nodes] = 0.0
        x[token_nodes] += self.enc_mask_token

        h1 = self.enc(graph, x)
        h2 = self.enc(new_graph, feat)

        h1 = self.dec1(graph, h1)
        h2 = self.dec2(new_graph, h2)
        new_matrix = torch.matmul(h2, h2.T)

        loss1 = self.criterion(h1, feat)
        loss2 = self.criterion1(new_matrix, matrix)
        # return loss2
        return self.alpha * loss1 + loss2 * (1 - self.alpha)

    def get_embed(self, graph, feat):
        h1 = self.enc(graph, feat)

        return h1


# class MG(nn.Module):
#     def __init__(self, in_hidden, out_hidden, p1, p2, beta, beta1, rate,
#                  rate1, alpha, n, n1):
#         super(MG, self).__init__()
#         self.enc = Encoder1(in_hidden, out_hidden, p1)
#         self.dec = Encoder2(in_hidden, out_hidden, p2)
#         self.rate = rate
#         self.rate1 = rate1
#         self.alpha = alpha
#         self.noise = n
#         self.noise1 = n1
#         self.enc_mask_token = nn.Parameter(torch.zeros(1, in_hidden))
#         self.enc_mask_token1 = nn.Parameter(torch.zeros(1, in_hidden))
#         # self.encoder_to_decoder = nn.Linear(out_hidden, out_hidden, bias=False)
#         self.criterion = self.setup_loss_fn("sce", beta)

#     def setup_loss_fn(self, loss_fn, alpha_l):
#         if loss_fn == "mse":
#             criterion = nn.MSELoss()
#         elif loss_fn == "sce":
#             criterion = partial(sce_loss, alpha=alpha_l)
#         else:
#             raise NotImplementedError
#         return criterion

#     def forward(self, graph, diff_graph, feat, edge_weight):

#         token_nodes, noise_nodes, \
#         noise_to_be_chosen, mask_nodes = mask(graph, feat, mask_rate=self.rate,
#                                               noise=self.noise)
#         x = feat.clone()
#         x[token_nodes] = 0.0
#         x[noise_nodes] = feat[noise_to_be_chosen]
#         x[token_nodes] += self.enc_mask_token

#         h1 = self.enc(graph, diff_graph, x, edge_weight)

#         token_nodes1, noise_nodes1, \
#         noise_to_be_chosen1, mask_nodes1 = mask(graph, feat, mask_rate=self.rate1,
#                                                 noise=self.noise1)
#         x = feat.clone()
#         x[token_nodes1] = 0.0
#         x[noise_nodes1] = feat[noise_to_be_chosen1]
#         x[token_nodes1] += self.enc_mask_token1

#         h2 = self.dec(graph, diff_graph, x, edge_weight)
#         loss1 = self.criterion(h1[mask_nodes], h2[mask_nodes])

#         return loss1

#     def get_embed(self, graph, diff_graph, feat, edge_weight):
#         h1 = self.enc(graph, diff_graph, feat, edge_weight)
#         h2 = self.dec(graph, diff_graph, feat, edge_weight)

#         return h1, h2