import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from model import MG
from torch_geometric import seed_everything
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from dataset import load_graph_classification_dataset
import networkx as nx

warnings.filterwarnings('ignore')


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


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


def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels

seed_everything(35536)
dataname = "REDDIT-BINARY"

graphs, (n_feat, num_classes) = load_graph_classification_dataset(dataname)
train_idx = torch.arange(len(graphs))
batch_size = 50
train_sampler = SubsetRandomSampler(train_idx)

train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn,
                               batch_size=batch_size, pin_memory=True)
eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=batch_size,
                              shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def TT(space):
    print(space)
    r = space
    model = MG(n_feat, space['dim'], space['beta'], space['beta1'], space['rate'],
               space['rate1'], space['alpha'], space['n1'], space['n2'], space['n3'],
               space['p1'], space['p2'], space['p3'], space['noise'], space['noise1']).to(device)
    optimizer = create_optimizer("adam", model, space['lr'], space['w'])
    a = []
    for epoch in range(1, 30+1):
        model.train()
        for g, label in train_loader:
            nx_g = dgl.to_networkx(g)
            matrix = torch.tensor(nx.convert_matrix.to_numpy_array(nx_g), dtype=torch.float32)
            loss = model(g.to(device), matrix.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        x_list = []
        y_list = []
        model.eval()
        for g, label in eval_loader:
            z1 = model.get_embed(g.to(device))
            y_list.append(label.numpy())
            x_list.append((z1).detach().cpu().numpy())
        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
        # print(f"Epoch: {epoch}, Loss: {loss.item()}, Acc: {test_f1}, Std: {test_std}")
        a.append(test_f1)
    r['acc'] = np.array(a).max()
    # r['a'] = np.array(a)
    print(r)

#     return {'loss': -round(r['acc'], 4), 'status': STATUS_OK}


# trials = Trials()
# space = {
#     "lr": hp.choice('lr', [1e-5, 5e-5, 8e-5, 1e-4, 5e-4, 8e-4, 1e-3,
#                            5e-3, 8e-3, 1e-2, 5e-2, 8e-2]),
#     "w": hp.choice('w', [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
#     "n1": hp.choice('n1', [1, 2, 3, 4, 5]),
#     "n2": hp.choice('n2', [1, 2, 3, 4, 5]),
#     "dim": hp.choice('dim', [32, 64, 128, 256, 512, 1024]),
#     "beta": hp.quniform('beta', 1, 5, 1),
#     "rate": hp.quniform('rate', 0.1, 0.9, 0.1),
#     "rate1": hp.quniform('rate1', 0.1, 0.9, 0.1),
#     "alpha": hp.quniform('alpha', 0.1, 0.9, 0.1),
#     "beta1": hp.quniform('beta1', 1, 5, 1),
#     "p1": hp.quniform('p1', 0.0, 0.9, 0.1),
#     "p2": hp.quniform('p2', 0.0, 0.9, 0.1),
#     "noise": hp.choice('noise', [0.0, 0.05, 0.1])
# }
# best = fmin(TT, space=space, algo=tpe.suggest, max_evals=200, trials=trials)
# print(best)
#TT({'alpha': 0.4, 'beta': 4.0, 'beta1': 1.0, 'dim': 256, 'lr': 8e-05,
#    'n1': 2, 'n2': 1, 'noise': 0.05, 'p1': 0.6, 'p2': 0.8, 'rate': 0.7,
#    'rate1': 0.8, 'w': 0.0005})
# TT({'alpha': 0.2, 'beta': 5.0, 'beta1': 3.0, 'dim': 64, 'lr': 0.0005, 'n1': 5, 'n2': 2,
#     'n3': 2, 'noise': 0.1, 'noise1': 0.1, 'p1': 0.8, 'p2': 0.9, 'p3': 0.9, 'rate': 0.6,
#     'rate1': 0.6, 'w': 1e-05, 'acc': 0.77})


alpha = [0.1,0.3,0.5,0.7,0.9]
for i in alpha:
    TT({'alpha': i, 'beta': 5.0, 'beta1': 3.0, 'dim': 64, 'lr': 0.0005, 'n1': 5, 'n2': 2,
        'n3': 2, 'noise': 0.1, 'noise1': 0.1, 'p1': 0.8, 'p2': 0.9, 'p3': 0.9, 'rate': 0.6,
        'rate1': 0.6, 'w': 1e-05})