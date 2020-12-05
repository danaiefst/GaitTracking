import data_handler, tracking_nn
import torch
from torch.optim import Adam, SGD
import os

#data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a","/home/danai/Desktop/GaitTracking/p5/2.a", "/home/danai/Desktop/GaitTracking/p11/2.a", "/home/danai/Desktop/GaitTracking/p11/3.a", "/home/danai/Desktop/GaitTracking/p16/3.a", "/home/danai/Desktop/GaitTracking/p17/3.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
data_paths = ["/gpu-data/athdom/p1/2.a"]
#data_paths = ["/home/danai/Desktop/GaitTracking/p1/2.a"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on", device)
model = tracking_nn.Net(device).to(device)
data = data_handler.LegDataLoader(data_paths)
print("Loading dataset...")
train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = data.load(32)


# Train the nn

epochs = 100
patience = 1
learning_rate = 0.0001
optimizer = Adam(model.parameters(), lr = learning_rate)
best_acc = float("Inf")
save_path = "/home/athdom/GaitTracking/model.pt"

print("Started training...")
for epoch in range(epochs):
    running_loss = 0
    for i in range(len(train_set_x)):
        model.init_hidden(1)
        #print("Training batch", i, "/", len(train_set_x))
        model.init_hidden(1)
        inputs, labels = train_set_x[i].to(device), train_set_y[i].to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = model.loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("epoch:{}, running loss: {}".format(epoch, running_loss / len(train_set_x)))
    running_loss = 0
    if epoch >= patience:
        with torch.no_grad():
            acc = 0
            for i, j in zip(val_set_x, val_set_y):
                input = i.to(device)
                label = j.to(device)
                output = model.forward(input)
                acc += model.loss(output, label)
            if acc < best_acc:
                best_acc = acc
                print("Saving model with acc", acc / len(val_set_x))
                torch.save(model.state_dict(), save_path)
