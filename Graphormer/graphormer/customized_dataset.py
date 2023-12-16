from graphormer.data import register_dataset
from dgl.data import QM9
import numpy as np
from sklearn.model_selection import train_test_split
import torch

import os

from graphormer.dataset_pyg import PygGraphPropPredDataset

@register_dataset("customized_qm9_dataset")
def create_customized_dataset():
    dataset = PygGraphPropPredDataset(name = 'dfg_dsp')
    print(dataset.num_classes)
    print(dataset[1])
    split_index = dataset.get_idx_split()
    # print(dataset[0].node_is_attributed)
    # print([dataset[i].x[1] for i in range(100)])
    # print(dataset[0].y)
    print(dataset[1].x)
    print(dataset[split_index['train']])
    print(dataset[split_index['valid']])
    print(dataset[split_index['test']])

    print("done done")
    # dataset = QM9(label_keys=["mu"])

    print(dataset)
    num_graphs = len(dataset)

    print(num_graphs)

    # customized dataset split
    train_valid_idx, test_idx = train_test_split(
        np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx, test_size=num_graphs // 5, random_state=0
    )
    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "pyg"
    }
