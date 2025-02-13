import numpy as np
import random
from collections import Counter, defaultdict


def stratified_group_k_fold(X, y, groups, k, seed=None):
    np.random.seed(seed)
    # ラベルの数をカウント
    labels_num = np.max(y) + 1
    # 各グループのラベルの数をカウントする
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1
    # 各フォールドのラベルの数をカウント
    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)

    for i in range(k):
        test_k = i
        # val_k = i+1 if i+1 != k else 0
        # print(val_k)
        train_groups = all_groups - groups_per_fold[test_k]  #  - groups_per_fold[val_k]
        # val_groups = groups_per_fold[val_k]
        test_groups = groups_per_fold[test_k]
        # print(test_groups)
        # train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        # val_indices = [i for i, g in enumerate(groups) if g in val_groups]
        # test_indices = {str(g): [i for i, g in enumerate(groups) if g in test_groups]}

        def choice_ind(group):
            indices = []
            n_g = None
            list_ = []
            for i, g in enumerate(groups):
                if g in group:
                    if n_g is not None and n_g != g:
                        indices.append(list_)
                        list_ = []
                    list_.append(i)
                    n_g = g

            indices = [np.random.choice(i) for i in indices]
            return indices

        train_indices = choice_ind(train_groups)
        test_indices = choice_ind(test_groups)
        # print(train_indices)
        yield train_indices, test_indices  # val_indices,


def get_distribution(y_vals):
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())
    return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]
