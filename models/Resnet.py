# This code was inspired by https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
NUM_CLASS=7
import torch as torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torcheval.metrics.functional import multiclass_f1_score

from tqdm import tqdm

class Resnet:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model = self.model.cuda()
        # Adjust to conv
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASS)

    def _create_dataloader(self, train_x, train_y, batch_size):
        
        x_tensor = torch.tensor(train_x, dtype=torch.float32)
        x_tensor = x_tensor.reshape(x_tensor.shape[0], 1, x_tensor.shape[1], x_tensor.shape[1])
        y_tensor = torch.tensor(train_y)
        dataset = TensorDataset(x_tensor, y_tensor)
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    
    def train(self, train_x, train_y, val_x, val_y, optimizer='adam', batch_size=32, epochs=20, lr=5e-4, eval_freq=4, **kwargs):
        # First let's make dataloaders
        train_loader = self._create_dataloader(train_x, train_y, batch_size)
        val_loader = self._create_dataloader(val_x, val_y, batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_model = None
        best_perf  = 0
        # Let train it!
        self.model.cuda()
        for epoch in range(1, epochs+1):
            self.model.train()
            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.to(self.device))
                loss.backward()
                optimizer.step()
                
            if epoch == 1 or epoch%eval_freq==0:
                predictions = []
                gts = []
                self.model.eval()
                self.model.to(self.device)
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        outputs = self.model(inputs)
                        predicted = torch.argmax(outputs, 1)
                        predictions += list(predicted)
                        gts += list(labels)
                predictions = torch.tensor([a.item() for a in predictions])
                gts = torch.tensor([a.item() for a in gts])
                f1 =  multiclass_f1_score(predictions, gts, num_classes=7)
                print(f"EPOCH : {epoch}, f1 : {f1}")
                if best_perf <= f1:
                    best_perf = f1
                    best_model = self.model.state_dict()
        self.model.load_state_dict(best_model)
    
    def predict(self, X):
        self.model = self.model.to(self.device)
        predictions = []
        for i in range(0, X.shape[0], 32):
            batch = torch.tensor(X[i: i+32], dtype=torch.float32)
            batch_tensor = batch.reshape(batch.shape[0], 1, batch.shape[1], batch.shape[1]).to(self.device)
            outputs = self.model(batch_tensor)
            predicted = torch.argmax(outputs, 1)
            predictions += list(predicted)
        return [a.item() for a in predictions]

    #def score(self, X, y):
     #   return self.model.score(X, y)