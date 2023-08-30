import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader

test_csv = pd.read_csv("csvTestImages.csv",header=None)
test_lab_csv = pd.read_csv("csvTestLabel.csv", header=None)
train_csv = pd.read_csv("csvTrainImages.csv", header=None)
train_lab_csv = pd.read_csv("csvTrainLabel.csv", header=None)

test_tensor = torch.tensor(test_csv.values)
test_lab_tensor = torch.tensor(test_lab_csv.values)
train_tensor = torch.tensor(train_csv.values)
train_lab_tensor = torch.tensor(train_lab_csv.values)

train_dataset = torch.utils.data.TensorDataset(train_tensor, train_lab_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(test_tensor, test_lab_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device_str = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = x.to(self.linear_relu_stack[0].weight.dtype)
        x = x.view(x.size(0), -1)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device_str)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #X_ = torch.Tensor(X)eeze()
        y = y.squeeze()
        X, y = X.to(device_str), y.to(device_str)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y = y.squeeze()
            X, y = X.to(device_str), y.to(device_str)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")

