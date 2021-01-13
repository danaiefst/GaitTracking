import data_handler, tracking_nn
import torch
import sys
from torch.optim import Adam

torch.cuda.manual_seed(1)
torch.manual_seed(1)

flag = int(sys.argv[1])
data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a","/home/danai/Desktop/GaitTracking/p5/2.a", "/home/danai/Desktop/GaitTracking/p11/2.a", "/home/danai/Desktop/GaitTracking/p11/3.a", "/home/danai/Desktop/GaitTracking/p16/3.a","/home/danai/Desktop/GaitTracking/p17/2.a", "/home/danai/Desktop/GaitTracking/p17/3.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
#data_paths = ["/gpu-data/athdom/p1/2.a", "/gpu-data/athdom/p18/2.a", "/gpu-data/athdom/p18/3.a"]
#data_paths = ["/home/danai/Desktop/GaitTracking/p1/2.a"]
#data_paths=["/home/shit/Desktop/GaitTracking/p1/2.a","/home/shit/Desktop/GaitTracking/p5/2.a", "/home/shit/Desktop/GaitTracking/p11/2.a", "/home/shit/Desktop/GaitTracking/p11/3.a", "/home/shit/Desktop/GaitTracking/p16/3.a", "/home/shit/Desktop/GaitTracking/p17/3.a", "/home/shit/Desktop/GaitTracking/p18/2.a", "/home/shit/Desktop/GaitTracking/p18/3.a"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on", device)
path = "/home/danai/Desktop/GaitTracking/"

cnn = tracking_nn.CNN().to(device)
if flag:
    cnn.load_state_dict(torch.load(path + "cnn_model.pt", map_location = device))
    for param in cnn.parameters():
        param.requires_grad = False
rnn = tracking_nn.RNN().to(device)
model = tracking_nn.Net(device, cnn, rnn).to(device)
data = data_handler.LegDataLoader(device = device, data_paths = data_paths)
print("Loading dataset...")
train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, test_set_y = data.load(32)

# Train the nn

epochs = 1000
patience = 1
learning_rate = 0.001
grid = 7
optimizer = Adam(model.parameters(), lr = learning_rate)
best_acc = float("Inf")
if flag:
    save_path = path + "model.pt"
else:
    save_path = path + "cnn_model.pt"

def eucl_dist(out, labels):
    ret = 0
    for i in range(out.shape[0]):
        yh = out[i]
        p1_h = yh[0, :, :]
        p2_h = yh[3, :, :]
        detect_cell1 = p1_h.reshape(-1).argmax(axis = 0)
        detect_cell2 = p2_h.reshape(-1).argmax(axis = 0)
        x1, y1 = detect_cell1 // grid, detect_cell1 % grid
        x2, y2 = detect_cell2 // grid, detect_cell2 % grid
        ret += (torch.sqrt((x1 + out[i, 1, x1, y1] - labels[i, 0, 0]) ** 2 + (y1 + out[i, 2, x1, y1] - labels[i, 0, 1]) ** 2) + torch.sqrt((x2 + out[i, 4, x2, y2] - labels[i, 1, 0]) ** 2 + (y2 + out[i, 5, x2, y2] - labels[i, 1, 1]) ** 2)).item()
    return ret / out.shape[0] / 2

print("Started training...")
for epoch in range(epochs):
    running_loss = 0
    if epoch % 10 == 0:
        learning_rate *= 0.1
        optimizer = Adam(model.parameters(), lr = learning_rate)
    for i in range(len(train_set_x)):
        model.init_hidden()
        inputs, labels = train_set_x[i], train_set_y[i]
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        #print("labels", labels[0])
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
            for input, label in zip(val_set_x, val_set_y):
                output = model.forward(input)
                acc += model.loss(output, label) / i.shape[0]
                dist += eucl_dist(output, label)
            if acc < best_acc:
                best_acc = acc
                print("Saving model with acc:", acc / len(val_set_x), ", mean dist:", dist / len(val_set_x) / grid * 100) #mean dist in cm
                if flag:
                    torch.save(model.state_dict(), save_path)
                else:
                    torch.save(cnn.state_dict(), save_path)
