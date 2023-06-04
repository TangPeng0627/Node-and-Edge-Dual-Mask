import torch
import dgl
from model import MG
from torch_geometric import seed_everything
import numpy as np
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from eval import label_classification
from dataset import process_dataset
import networkx as nx

warnings.filterwarnings('ignore')


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = torch.optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = torch.optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return torch.optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


dataname = "wikics"
label_type = 0
graph, feat, label, train_mask, val_mask, test_mask = process_dataset(dataname, 0.01)
nx_g = dgl.to_networkx(graph)
matrix = torch.tensor(nx.convert_matrix.to_numpy_array(nx_g), dtype=torch.float32)
n_feat = feat.shape[1]
n_classes = np.unique(label).shape[0]
seed_everything(35536)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

graph = graph.to(device)
label = label.to(device)
feat = feat.to(device)
matrix = matrix.to(device)



n_node = graph.number_of_nodes()


def TT(space):
    r = space
    model = MG(feat.size(1), int(space['outdim']), space['p1'], space['p2'], space['p3'], space['beta'],
               space['beta1'], space['rate'], space['rate1'], space['alpha'], space['n'], space['n1']).to(device)
    optimizer = create_optimizer("adam", model, space['lr'], space['w'])
    a = []
    for epoch in range(1, 10 + 1):
        model.train()
        loss = model(graph, feat, matrix)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        model.eval()
        z1 = model.get_embed(graph, feat)
        acc = label_classification(z1, train_mask, val_mask, test_mask,
                                   label, label_type)
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Acc: {acc}")
        a.append(acc)
    r['acc'] = np.array(a).max()
    # r['a'] = np.array(a)
    print(r)

    return {'loss': -round(r['acc'], 4), 'status': STATUS_OK}


# trials = Trials()
# space = {
#     "outdim": hp.choice('outdim', [1024, 2048]),
#     "lr": hp.choice('lr', [1e-5, 5e-5, 8e-5, 1e-4, 5e-4, 8e-4, 1e-3,
#                            5e-3, 8e-3, 1e-2, 5e-2, 8e-2]),
#     "w": hp.choice('w', [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
#     "p1": hp.quniform('p1', 0.0, 0.9, 0.1),
#     "p2": hp.quniform('p2', 0.0, 0.9, 0.1),
#     "p3": hp.quniform('p3', 0.0, 0.9, 0.1),
#     "beta": hp.quniform('beta', 1, 5, 1),
#     "rate": hp.quniform('rate', 0.1, 0.9, 0.1),
#     "rate1": hp.quniform('rate1', 0.1, 0.9, 0.1),
#     "alpha": hp.quniform('alpha', 0.1, 0.9, 0.1),
#     "beta1": hp.quniform('beta1', 1, 2, 1),
#     "n": hp.quniform('n', 0.0, 0.9, 0.01),
#     "n1": hp.quniform('n1', 0.0, 0.9, 0.01),
# }
# best = fmin(TT, space=space, algo=tpe.suggest, max_evals=300, trials=trials)
# print(best)


# TT({'alpha': 0.8, 'beta': 1.0, 'beta1': 3.0, 'lr': 5e-05, 'n': 0.24, 'n1': 0.49,
#     'outdim': 2048, 'p1': 0.9, 'p2': 0.7, 'p3': 0.0, 'rate': 0.9,
#     'rate1': 0.9, 'w': 0.001, 'acc': 0.7961347699675047})

TT({'alpha': 0.8, 'beta': 1.0, 'beta1': 3.0, 'lr': 5e-05, 'n': 0.24, 'n1': 0.49,
    'outdim': 2048, 'p1': 0.9, 'p2': 0.7, 'p3': 0.0, 'rate': 0.1,
    'rate1': 0.9, 'w': 0.001, 'acc': 0.7961347699675047})

# arr = [0.1,0.3,0.5,0.7,0.9]
# arr1 = [0.1,0.3,0.5,0.7,0.9]
# for rate in arr:
#     for rate1 in arr1:
#         TT({'alpha': 0.8, 'beta': 1.0, 'beta1': 3.0, 'lr': 5e-05, 'n': 0.24, 'n1': 0.49,
#             'outdim': 2048, 'p1': 0.9, 'p2': 0.7, 'p3': 0.0, 'rate': rate,
#             'rate1': rate1, 'w': 0.001, 'acc': 0.7961347699675047})
