import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

from utils import get_max_num_node

hidden_size = 64

class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, softmax, x, mask, sqrt_d):
        ctx.save_for_backward(x, mask, sqrt_d)
        y = softmax((x + mask) / sqrt_d)
        y[y.isnan()] = 0

        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, mask, sqrt_d = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.isnan()] = 0

        return grad_input

class DropoutNan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.clone()
        y[y.isnan()] = 0

        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output)
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.isnan()] = 0

        return grad_input

class HLSTransformer(nn.Module):
    def __init__(self, lrate, loss_fn, num_features):
        super(HLSTransformer, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=-1)
        self.layernorm = nn.LayerNorm([get_max_num_node(), hidden_size])

        self.fc0 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.LeakyReLU()
        )

        self.lrate = lrate
        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(self.parameters(), self.lrate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
    

    def forward(self, x, mask):
        # print(x)
        x_emb = self.embedding(x)
        # print(x_emb)
        Q, K, V = x_emb, x_emb, x_emb
        QK = torch.matmul(Q, K.transpose(-1, -2))
        QK_max, _ = torch.max(QK, dim=-1, keepdim=True)
        QK = self.softmax((QK - QK_max + mask) / np.sqrt(get_max_num_node(), dtype=np.float32))

        # print(QK.isnan().sum())

        QKV = self.layernorm(torch.matmul(QK, V) + x_emb)

        # print(V)

        fc = self.layernorm(QKV + self.fc0(QKV))

        # second block starts

        Q, K, V = fc, fc, fc
        QK = torch.matmul(Q, K.transpose(-1, -2))
        QK_max, _ = torch.max(QK, dim=-1, keepdim=True)
        QK = self.softmax((QK - QK_max + mask) / np.sqrt(get_max_num_node(), dtype=np.float32))

        # print(QK.isnan().sum())

        QKV = self.layernorm(torch.matmul(QK, V) + x_emb)

        # print(V)

        fc = self.layernorm(QKV + self.fc1(QKV))

        # second block ends

        out = self.output_layer(fc.sum(dim=1))
        
        return out

    def step(self, x, mask, y):        
        self.optimizer.zero_grad()
        y_pred = self.forward(x, mask)
        loss = self.loss_fn(y_pred, y)
        loss.backward()

        self.optimizer.step()
        # self.scheduler.step(loss)

        return loss
    
    def validate(self, x, mask, y):
        y_pred = self.forward(x, mask)
        loss = self.loss_fn(y_pred, y)

        return loss
    
    def fit(self, train_data, val_data, num_features, epochs, batch_size, device, save_interval, early_stop_epochs, fold):
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        training_losses = []
        validation_losses = []
        best_loss = float('inf')
        best_params = None

        for epoch in range(epochs):
            self.train()
            total_train_loss = 0
            for batch in train_dataloader:
                x = batch['features']
                mask = batch['mask']
                y = batch['labels']
                loss = self.step(x, mask, y)
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_dataloader)

            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    x = batch['features']
                    mask = batch['mask']
                    y = batch['labels']
                    loss = self.validate(x, mask, y)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)

            scheduler.step(avg_val_loss)  # Update the learning rate based on validation loss

            training_losses.append(avg_train_loss)
            validation_losses.append(avg_val_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

            if (epoch + 1) % save_interval == 0:
                torch.save(self.state_dict(), f'model_epoch_{fold}_{epoch+1}.pth')
                print(f'Model saved at epoch {epoch+1}')
            
            # Early stopping logic
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = self.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == early_stop_epochs:
                print("Early stopping triggered")
                break

        torch.save(best_params, f'model_best_{fold}.pth')
        print(f'Model saved at epoch {epoch+1}')
        print(f"best_loss: {best_loss}")

        return best_loss, training_losses, validation_losses

    
"Alex's original code start"
# def fit(train_data, num_features, epochs, batch_size, device):
#     train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
#     net = HLSTransformer(lrate, nn.MSELoss(), num_features).to(device)

#     losses = []

#     while True:
#         total_loss = 0
#         for step, batch in enumerate(train_dataloader):
#             x = batch['features']
#             mask = batch['mask']
#             y = batch['labels']
#             loss = net.step(x, mask, y)
#             total_loss += loss
#         total_loss /= batch_size

#         print(total_loss)
#         losses.append(total_loss)

    # return losses
"Alex's original code end"

# def fit(train_data, val_data, num_features, epochs, batch_size, device):
#     train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
#     net = HLSTransformer(lrate, MSELoss(), num_features).to(device)

#     training_losses = []
#     validation_losses = []

#     for epoch in range(epochs):
#         net.train()  # Set the model to training mode
#         total_train_loss = 0
#         for batch in train_dataloader:
#             x = batch['features']
#             mask = batch['mask']
#             y = batch['labels']
#             loss = net.step(x, mask, y)
#             total_train_loss += loss.item()
        
#         avg_train_loss = total_train_loss / len(train_dataloader)
#         training_losses.append(avg_train_loss)

#         # Validation loop
#         net.eval()  # Set the model to evaluation mode
#         total_val_loss = 0
#         with torch.no_grad():
#             for batch in val_dataloader:
#                 x = batch['features']
#                 mask = batch['mask']
#                 y = batch['labels']
#                 loss = net.validate(x, mask, y)
#                 total_val_loss += loss.item()
        
#         avg_val_loss = total_val_loss / len(val_dataloader)
#         validation_losses.append(avg_val_loss)

#         print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

#     return net, training_losses, validation_losses

