from utils import GraphDataset
from HLSTransformer import fit


path = "dfg_lut/raw/"
device = "cuda:0"

if __name__ == "__main__":
    
    train_data = GraphDataset(path + "num-node-list.csv", path + "node-feat.csv", path + "num-edge-list.csv", \
        path + "edge.csv", path + "graph-label.csv", device)
    
    losses = fit(train_data, 7, 100, 32, device)