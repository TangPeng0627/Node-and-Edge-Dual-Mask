import numpy as np
import functools

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


# Code reference resources https://github.com/flyingtango/DiGCL,
#                          https://github.com/CRIPAC-DIG/GRACE
def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            # print_statistics(statistics, f.__name__)
            return statistics

        return wrapper

    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


# def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None,
#                      test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
#     random_state = np.random.RandomState(seed)
#     train_indices, val_indices, test_indices = get_train_val_test_split(
#         random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size,
#         val_size, test_size)
#
#     # print('number of training: {}'.format(len(train_indices)))
#     # print('number of validation: {}'.format(len(val_indices)))
#     # print('number of testing: {}'.format(len(test_indices)))
#
#     train_mask = np.zeros((labels.shape[0], 1), dtype=int)
#     train_mask[train_indices, 0] = 1
#     train_mask = np.squeeze(train_mask, 1)
#     val_mask = np.zeros((labels.shape[0], 1), dtype=int)
#     val_mask[val_indices, 0] = 1
#     val_mask = np.squeeze(val_mask, 1)
#     test_mask = np.zeros((labels.shape[0], 1), dtype=int)
#     test_mask[test_indices, 0] = 1
#     test_mask = np.squeeze(test_mask, 1)
#     mask = {}
#     mask['train'] = train_mask
#     mask['val'] = val_mask
#     mask['test'] = test_mask
#     return mask


# type: Label division type, i.e. 0 represents fixed label and
#       1 represents 20 nodes for each type of sampling
@repeat(10)
def label_classification(embeddings, graph, type):
    X = embeddings.detach().cpu().numpy()
    Y = graph.ndata["label"].detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    if np.isinf(X).any() == True or np.isnan(X).any() == True:
        return {
            'F1Mi': 0,
            'F1Ma': 0,
            'Acc': 0
        }
    X = normalize(X, norm='l2')
    X_train = X[graph.ndata["train_mask"].cpu().numpy()]
    X_val = X[graph.ndata["val_mask"].cpu().numpy()]
    X_test = X[graph.ndata["test_mask"].cpu().numpy()]
    y_train = Y[graph.ndata["train_mask"].cpu().numpy()]
    y_val = Y[graph.ndata["val_mask"].cpu().numpy()]
    y_test = Y[graph.ndata["test_mask"].cpu().numpy()]
    # a = []
    # for i in range(20):
    #     X_train = X[train_mask[:, i].cpu().numpy()]
    #     X_val = X[val_mask[:, i].cpu().numpy()]
    #     X_test = X[test_mask.cpu().numpy()]
    #     y_train = Y[train_mask[:, i].cpu().numpy()]
    #     y_val = Y[val_mask[:, i].cpu().numpy()]
    #     y_test = Y[test_mask.cpu().numpy()]
    # X_train, X_test, y_train, y_test = train_test_split(X, Y,
    #                                                     test_size=1 - 0.1)
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                           param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                           verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)
    acc = accuracy_score(y_test, y_pred)
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
        # a.append(acc)
        # return np.array(a).mean()

    return {
        'F1Mi': micro,
        'F1Ma': macro,
        'Acc': acc
    }
