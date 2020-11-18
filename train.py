import data_handler, tracking_nn
import torch
from torch.optim import Adam, SGD
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on", device)
model = tracking_nn.Net(device).to(device)
data = data_handler.LegDataLoader()
print("Loading dataset...")
train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = data.load(32)


# Train the nn

epochs = 100
patience = 10
learning_rate = 0.01
optimizer = Adam(model.parameters(), lr = learning_rate)
best_acc = 0
save_path = "/home/athdom/GaitTracking/model.pt"

print("Started training...")
for epoch in range(epochs):
    running_loss = 0
    for i in range(len(train_set_x)):
        inputs, labels = train_set_x[i].to(device), train_set_y[i].to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = model.loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print("epoch:{}, batch: {}, running loss: {}".format(epoch, i, running_loss / 50))
            running_loss = 0
"""
    if epoch >= patience:
        with torch.no_grad():
            acc = 0
            for i, j in zip(val_set_x, val_set_y):
                input = i.to(device)
                label = j.to(device)
                output = model.forward(input)
                acc += model.loss(output, label)
            if acc > best_acc:
                best_acc = acc
                torch.save(model, save_path)
"""
