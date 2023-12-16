from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np

max_num_node = 259

def get_max_num_node():
    return max_num_node

def read_csv(file_name, dtype=int):
    with open(file_name, newline='') as file:
        reader = pd.read_csv(file, header=None, dtype=dtype)
        return reader.values

class GraphDataset(Dataset):
    def __init__(self, num_node_list_filename, node_feat_filename, num_edge_list_filename, edge_filename, graph_label_filename, device):
        global max_num_node

        num_node_list = read_csv(num_node_list_filename)
        node_feat = read_csv(node_feat_filename, np.float32)
        num_edge_list = read_csv(num_edge_list_filename)
        edges = read_csv(edge_filename)
        graph_label = read_csv(graph_label_filename, np.float32)

        self.node_features = []
        self.masks = []
        self.labels = torch.from_numpy(graph_label).to(device)

        for num_node in num_node_list:
            max_num_node = max(max_num_node, num_node[0])
        
        self.max_num_node = max_num_node
        self.num_features = len(node_feat[0])
        print("max_num_node: {}".format(self.max_num_node))
        print("num_features: {}".format(self.num_features))

        num_nodes = 0
        for num_node in num_node_list:
            feat = torch.full((self.max_num_node, self.num_features), 0.0, dtype=torch.float32)
            feat[:num_node[0]] = torch.from_numpy(node_feat[num_nodes : num_nodes + num_node[0]])
            self.node_features.append(feat.to(device))
            num_nodes += num_node[0]
        
        num_edges = 0
        for num_edge in num_edge_list:
            mask = torch.full((max_num_node, max_num_node), 0, dtype=torch.int32)

            for edge in edges[num_edges : num_edges + num_edge[0]]:
                mask[edge[0], edge[1]] = 1
                mask[edge[1], edge[0]] = 1
            
            for i in range(self.max_num_node):
                mask[i, i] = 1

            self.masks.append(mask.to(device))
            num_edges += num_edge[0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features = self.node_features[idx]
        label = self.labels[idx]
        mask = self.masks[idx]
        sample = {'features': features,'labels': label, 'mask': mask}
        return sample
