if __name__ == "__main__":
    full_dataset = GraphDataset(path + "num-node-list.csv", path + "node-feat.csv", path + "num-edge-list.csv", \
        path + "edge.csv", path + "graph-label.csv", device)

    # Assuming GraphDataset is defined and dataset is loaded
    total_size = len(dataset)
    train_val_size = int(total_size * 0.9)
    test_size = total_size - train_val_size

    # Randomly split dataset into training + validation and test sets
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])
    
