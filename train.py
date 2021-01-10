import data_handler, tracking_nn
import torch
from torch.optim import Adam

torch.cuda.manual_seed(1)
torch.manual_seed(1)

#data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a","/home/danai/Desktop/GaitTracking/p5/2.a", "/home/danai/Desktop/GaitTracking/p11/2.a", "/home/danai/Desktop/GaitTracking/p11/3.a", "/home/danai/Desktop/GaitTracking/p16/3.a", "/home/danai/Desktop/GaitTracking/p17/3.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
data_paths = ["/gpu-data/athdom/p1/2.a", "/gpu-data/athdom/p18/2.a", "/gpu-data/athdom/p18/3.a"]
#data_paths = ["/home/danai/Desktop/GaitTracking/p1/2.a"]
#data_paths=["/home/shit/Desktop/GaitTracking/p1/2.a","/home/shit/Desktop/GaitTracking/p5/2.a", "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Working on", device)
cnn = tracking_nn.CNN().to(device)
#cnn.load_state_dict(torch.load("/home/athdom/GaitTracking/cnn_model.pt", map_location = device))
#for param in cnn.parameters():
#    param.requires_grad = False
rnn = tracking_nn.RNN().to(device)
model = tracking_nn.Net(device, cnn, rnn).to(device)
data = data_handler.LegDataLoader(data_paths = data_paths)
print("Loading dataset...")
train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = data.load(32)


# Train the nn

epochs = 1000
patience = 1
learning_rate = 0.001
optimizer = Adam(model.parameters(), lr = learning_rate)
best_acc = float("Inf")
save_path = "/home/athdom/GaitTracking/model.pt"

def eucl_dist(out, labels):
    return (torch.sqrt(((out[:, :2] - labels[:, 0]) ** 2).sum(axis = 1)).sum() + torch.sqrt(((out[:, 2:] - labels[:, 1]) ** 2).sum(axis = 1)).sum()) / out.shape[0] / 2

print("Started training...")
for epoch in range(epochs):
    running_loss = 0
    if epoch == 20 or epoch == 50:
        learning_rate *= 0.1
        optimizer = Adam(model.parameters(), lr = learning_rate)
    for i in range(len(train_set_x)):
        model.init_hidden(1)
        inputs, labels = train_set_x[i].to(device), train_set_y[i].to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        print("labels", labels[0])
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
