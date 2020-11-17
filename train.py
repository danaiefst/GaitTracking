import data_handler, tracking_nn
import torch
from torch.optim import Adam, SGD


model = tracking_nn.Net()
data = data_handler.LegDataLoader()
print("Loading dataset...")
train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = data.load(32)


# Train the nn

epochs = 50
patience = 5
learning_rate = 0.0001
optimizer = Adam(model.parameters(), lr = learning_rate)
best_acc = 0
save_path = "/home/danai/Desktop/mobot/model.pt"

print("Started training...")
for epoch in range(epochs):
    running_loss = 0
    for i in range(len(train_set_x)):
        inputs, labels = train_set_x[i], train_set_y[i]
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = model.loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print("epoch:{}, batch: {}, running loss: {}".format(epoch, i, running_loss / 50))
            running_loss = 0

        if epoch >= patience:
            with torch.no_grad():
                acc = 0
                for i, j in zip(val_set_x, val_set_y):
                    output = model.forward(i, j)
                    acc += model.loss(output, j)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model, save_path)
