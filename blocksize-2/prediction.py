import torch
from HLSTransformer import HLSTransformer
from utils import get_max_num_node

num_features = 7
lrate = 1e-7
device="cuda:0"

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

model_path = f'model_best_0.pth'  # Replace with your model's filename
print(model_path)
pred_model = load_model(model_path, num_features, lrate, device)
predictions = make_predictions(pred_model, test_dataset, device)

# Save predictions to CSV
predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
predictions_df.to_csv('predictions.csv', index=False)
