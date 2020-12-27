import data_handler, classification
import torch
from torch.optim import Adam

data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a","/home/danai/Desktop/GaitTracking/p5/2.a", "/home/danai/Desktop/GaitTracking/p11/2.a", "/home/danai/Desktop/GaitTracking/p11/3.a", "/home/danai/Desktop/GaitTracking/p16/3.a", "/home/danai/Desktop/GaitTracking/p17/3.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
#data_paths = ["/gpu-data/athdom/p1/2.a"]
#data_paths = ["/home/danai/Desktop/GaitTracking/p1/2.a"]
#data_paths=["/home/shit/Desktop/GaitTracking/p1/2.a","/home/shit/Desktop/GaitTracking/p5/2.a", "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on", device)
model = classification.Net(device).to(device)
data = data_handler.LegDataLoader(cnn = 1)
print("Loading dataset...")
train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = data.load(32)


# Train the nn

grid = 7
epochs = 1000
patience = 1
learning_rate = 0.001
optimizer = Adam(model.parameters(), lr = learning_rate)
best_acc = float("Inf")
save_path = "/home/athdom/GaitTracking/model.pt"

def eucl_dist(out, labels):
    #Probability loss
    probh = out
    prob = torch.zeros(labels.shape[0], 2, grid, grid)
    prob[torch.arange(labels.shape[0]), 0, labels[:, 0, 0].long(), labels[:, 0, 1].long()] = 1
    prob[torch.arange(labels.shape[0]), 1, labels[:, 1, 0].long(), labels[:, 1, 1].long()] = 1
    prob_loss = ((prob - probh) ** 2).sum()

    return prob_loss

print("Started training...")
for epoch in range(epochs):
    running_loss = 0
    if epoch % 10 == 0:
        learning_rate *= 0.1
        optimizer = Adam(model.parameters(), lr = learning_rate)
    for i in range(len(train_set_x)):
        #print("Training batch", i, "/", len(train_set_x))
        inputs, labels = train_set_x[i].to(device), train_set_y[i].to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = model.loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() / train_set_x[i].shape[0]
    print("epoch:{}, running loss: {}".format(epoch, running_loss / len(train_set_x)))
    running_loss = 0
    if epoch >= patience:
        with torch.no_grad():
            acc = 0
            dist = 0
            for i, j in zip(val_set_x, val_set_y):
                input = i.to(device)
                label = j.to(device)
                output = model.forward(input)
                acc += model.loss(output, label) / i.shape[0]
                dist += eucl_dist(output, label)
            if acc < best_acc:
                best_acc = acc
                print("Saving model with acc:", acc / len(val_set_x), ", mean dist:", dist / len(val_set_x))
                torch.save(model.state_dict(), save_path)
