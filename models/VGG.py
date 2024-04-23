NUM_CLASS = 7
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torcheval.metrics.functional import multiclass_f1_score

from tqdm import tqdm

class VGG:
    def __init__(self, device=None):
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=NUM_CLASS, bias=True)
        self.model = self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # conver to grayscale, if not
            transforms.Resize((224, 224)),               # resize to match dimesions for VGG expected input (224*224)
            transforms.ToTensor(),                        
            transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization
        ])

    def _create_dataloader(self, train_x, train_y, batch_size):
        processed_images = [self.transform(Image.fromarray(img)) for img in train_x]
        x_tensor = torch.stack(processed_images)
        y_tensor = torch.tensor(train_y, dtype=torch.int64)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def train(self, train_x, train_y, val_x, val_y, optimizer='adam', batch_size=32, epochs=5, lr=0.0005,):
        train_loader = self._create_dataloader(train_x, train_y, batch_size)
        val_loader = self._create_dataloader(val_x, val_y, batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_model, best_perf = None, float('-inf')

        # training
        for epoch in range(1, epochs+1):
            self.model.train()
            train_loss = 0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Evaluating after each epoch

            val_f1 = self.evaluate(val_loader)
            print(f"EPOCH: {epoch}, Train Loss: {train_loss / len(train_loader):.4f}, Val F1: {val_f1:.4f}")
            if val_f1 > best_perf:
                best_perf = val_f1
                best_model = self.model.state_dict()

        if best_model is not None:
            self.model.load_state_dict(best_model)

    def evaluate(self, val_loader):
        self.model.eval()
        predictions, gts = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                gts.extend(labels.cpu().numpy())

        predictions = torch.tensor(predictions)
        gts = torch.tensor(gts)
        f1 = multiclass_f1_score(predictions, gts, num_classes=NUM_CLASS)
        return f1.item()

    def predict(self, X):
        self.model.eval()
        predictions = []
        X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, X.shape[1], X.shape[1]).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predicted = torch.argmax(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
        return predictions