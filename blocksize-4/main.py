from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
from utils import GraphDataset
from HLSTransformer import HLSTransformer
import torch.nn as nn
import pandas as pd
import csv


path = "dfg_lut/raw/"
device = "cuda:0"

lrate = 1e-7
num_features = 7
K = 5
save_interval = 20
early_stop_epochs = 10
epochs = 200
batch_size = 32

# if __name__ == "__main__":
    
#     train_data = GraphDataset(path + "num-node-list.csv", path + "node-feat.csv", path + "num-edge-list.csv", \
#         path + "edge.csv", path + "graph-label.csv", device)
    
#     losses = fit(train_data, 7, 100, 32, device)

# Function to load the model
def load_model(model_path, num_features, lrate, device):
    model = HLSTransformer(lrate, torch.nn.MSELoss(), num_features)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to make predictions
def make_predictions(model, dataset, device):
    predictions = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            features, mask = batch['features'].to(device), batch['mask'].to(device)
            prediction = model(features, mask)
            predictions.extend(prediction.cpu().numpy())
    
    return predictions  

if __name__ == "__main__":
    full_dataset = GraphDataset(path + "num-node-list.csv", path + "node-feat.csv", path + "num-edge-list.csv", \
                                path + "edge.csv", path + "graph-label.csv", device)
    
    # Assuming GraphDataset is defined and dataset is loaded
    total_size = len(full_dataset)
    train_val_size = int(total_size * 0.9)
    test_size = total_size - train_val_size

    # Randomly split dataset into training + validation and test sets
    train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size], \
        generator=torch.Generator().manual_seed(42))
    
    data_list = []
    for i, batch in enumerate(test_dataset):
        labels = batch['labels'].detach().cpu().numpy()
        data_list.append(labels)
    
    with open("test.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Target'])  # Write header
        csv_writer.writerows(data_list)
    
    kfold = KFold(n_splits=K, shuffle=True, random_state=42)
    net = HLSTransformer(lrate, nn.MSELoss(), num_features).to(device)

    best_fold = 0
    best_loss = float("inf")
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(train_val_dataset)))):
        print(f"Fold {fold + 1}/{K}")

        train_data = Subset(train_val_dataset, train_ids)
        val_data = Subset(train_val_dataset, val_ids)

        loss, _, _ = net.fit(train_data, val_data, num_features, epochs, \
            batch_size, device, save_interval, early_stop_epochs, fold)
        
        if best_loss > loss:
            best_loss = loss
            best_fold = fold
    
    model_path = f'model_best_{best_fold}.pth'  # Replace with your model's filename
    print(model_path)
    pred_model = load_model(model_path, num_features, lrate, device)
    predictions = make_predictions(pred_model, test_dataset, device)

    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    predictions_df.to_csv('predictions.csv', index=False)