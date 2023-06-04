import torch
import dgl
from model import MG
from torch_geometric import seed_everything
import numpy as np
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from eval import label_classification
from dataset import process_dataset

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


dataname = "Physics"
label_type = 0
graph, diff_graph, feat, label, edge_weight = process_dataset(dataname, 0.01)
n_feat = feat.shape[1]
n_classes = np.unique(label).shape[0]
seed_everything(35536)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

graph = graph.to(device)
label = label.to(device)
diff_graph = diff_graph.to(device)
feat = feat.to(device)
edge_weight = torch.tensor(edge_weight).float().to(device)

n_node = graph.number_of_nodes()


def TT(space):
    r = space
    model = MG(feat.size(1), 1024*2, space['p1'], space['p2'], space['beta'],
               space['beta1'], space['rate'], space['rate1'], space['alpha']).to(device)
    optimizer = create_optimizer("adam", model, space['lr'], space['w'])
    total_params = sum(p.numel() for p in model.parameters())
    print("Number of parameter: %.2fM" % (total_params/1e6))
    # num_finetune_params = [p.numel() for p in encoder.parameters() if p.requires_grad]
    # print(f"num parameters for finetuning: {sum(num_finetune_params)/1e6}")
    # scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / 100)) * 0.5
    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    a = []
    for epoch in range(1, 30 + 1):
        model.train()
        loss = model(graph, diff_graph, feat, edge_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        model.eval()
        z1, z2 = model.get_embed(graph, diff_graph, feat, edge_weight)
        acc = label_classification(z1 + z2,
                                   label, label_type)['Acc']['mean']
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Acc: {acc}")
        a.append(acc)
    r['acc'] = np.array(a).max()
    # r['a'] = np.array(a)
    print(r)

#     return {'loss': -round(r['acc'], 4), 'status': STATUS_OK}
#TT({'alpha': 0.9, 'beta': 3.0, 'beta1': 5.0, 'lr': 0.0005, 'p1': 0.1, 'p2': 0.7, 'rate': 0.5, 'rate1': 0.2, 'w': 0.0, 'acc': 0.9596475969591547})
#TT({'alpha': 0.3, 'beta': 4.0, 'beta1': 4.0, 'lr': 0.0008, 'p1': 0.3, 'p2': 0.6, 'rate': 0.6, 'rate1': 0.5, 'w': 5e-05, 'acc': 0.9558884164411803})
TT({'alpha': 0.7, 'beta': 5.0, 'beta1': 1.0, 'lr': 0.008, 'p1': 0.7, 'p2': 0.8, 'rate': 0.4, 'rate1': 0.7, 'w': 5e-05, 'acc': 0.9571575827857236})
#TT({'alpha': 0.8, 'beta': 3.0, 'beta1': 4.0, 'lr': 0.0008, 'p1': 0.0, 'p2': 0.4, 'rate': 0.7, 'rate1': 0.6, 'w': 0.0001, 'acc': 0.9586780054116739})
#TT({'alpha': 0.4, 'beta': 5.0, 'beta1': 2.0, 'lr': 0.05, 'p1': 0.5, 'p2': 0.4, 'rate': 0.4, 'rate1': 0.6, 'w': 5e-05, 'acc': 0.9585910320834945})
# trials = Trials()
# space = {
#     "lr": hp.choice('lr', [1e-5, 5e-5, 8e-5, 1e-4, 5e-4, 8e-4, 1e-3,
#                            5e-3, 8e-3, 1e-2, 5e-2, 8e-2]),
#     "w": hp.choice('w', [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
#     "p1": hp.quniform('p1', 0.0, 0.9, 0.1),
#     "p2": hp.quniform('p2', 0.0, 0.9, 0.1),
#     "beta": hp.quniform('beta', 1, 5, 1),
#     "rate": hp.quniform('rate', 0.0, 0.9, 0.1),
#     "rate1": hp.quniform('rate1', 0.0, 0.9, 0.1),
#     "alpha": hp.quniform('alpha', 0.0, 0.9, 0.1),
#     "beta1": hp.quniform('beta1', 1, 5, 1)
# }
# best = fmin(TT, space=space, algo=tpe.suggest, max_evals=200, trials=trials)
# print(best)
