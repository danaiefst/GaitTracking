import torch

torch.cuda.manual_seed(3)
torch.manual_seed(3)

import data_handler, tracking_nn
import sys
from torch.optim import Adam

flag = int(sys.argv[1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on", device)
#path = "/home/athdom/GaitTracking/"
#path = "/home/iral-lab/GaitTracking/"
path = "/home/danai/Desktop/GaitTracking/"
data_path = path + "data/" 
batch_size = 32

cnn = tracking_nn.CNN().to(device)
#if flag:
#    cnn.load_state_dict(torch.load(path + "cnn_model.pt", map_location = device))
#    for param in cnn.parameters():
#        param.requires_grad = False
rnn = tracking_nn.RNN().to(device)
model = tracking_nn.Net(device, cnn, rnn).to(device)
data = data_handler.LegDataLoader(batch_size = batch_size, data_path = data_path)
# Train the nn

epochs = 1000
patience = 0
learning_rate = 0.0001
grid = 7
optimizer = Adam(model.parameters(), lr = learning_rate)
best_acc = float("Inf")
if flag:
    save_path = path + "model.pt"
else:
    save_path = path + "cnn_model.pt"

def eucl_dist(out, labels):
    ret = 0
    m = 0
    for i in range(out.shape[0]):
        yh = out[i]
        p1_h = yh[0, :, :]
        p2_h = yh[3, :, :]
        detect_cell1 = p1_h.reshape(-1).argmax(axis = 0)
        detect_cell2 = p2_h.reshape(-1).argmax(axis = 0)
        x1, y1 = detect_cell1 // grid, detect_cell1 % grid
        x2, y2 = detect_cell2 // grid, detect_cell2 % grid
        d1 = (torch.sqrt((x1 + out[i, 1, x1, y1] - labels[i, 0, 0]) ** 2 + (y1 + out[i, 2, x1, y1] - labels[i, 0, 1]) ** 2)).item()
        d2 = (torch.sqrt((x2 + out[i, 4, x2, y2] - labels[i, 1, 0]) ** 2 + (y2 + out[i, 5, x2, y2] - labels[i, 1, 1]) ** 2)).item()
        if d1 > m:
            m = d1
        if d2 > m:
            m = d2
        ret += (d1 + d2) / 2
    return m, ret / out.shape[0]

print("Started training...")
for epoch in range(epochs):
    running_loss = 0
    if epoch == 10 or epoch == 23:
        learning_rate *= 0.1
        optimizer = Adam(model.parameters(), lr = learning_rate)
    f, input, label = data.load(0)
    model.init_hidden()
    c = 0
    while(True):
        if f:
            model.init_hidden()
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        output = model.forward(input)
        #print("labels", labels[0])
        loss = model.loss(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() / input.shape[0]
        c += 1
        if f == -1:
            break
        f, input, label = data.load(0)
        #model.init_hidden()
        model.detach_hidden()
    print("epoch:{}, running loss: {}, #n: {}".format(epoch, running_loss / c, c))
    running_loss = 0
    if epoch >= patience:
        with torch.no_grad():
            acc = 0
            dist = 0
            c = 0
            f, input, label = data.load(1)
            model.init_hidden()
            m = 0
            while(True):
                if f:
                    model.init_hidden()
                input, label = input.to(device), label.to(device)
                output = model.forward(input)
                acc += model.loss(output, label) / input.shape[0]
                m1, d = eucl_dist(output, label)
                dist += d
                if m1 > m:
                    m = m1
                c += 1
                if f == -1:
                    break
                f, input, label = data.load(1)
                #model.init_hidden()
            if acc < best_acc:
                best_acc = acc
                print("Saving model with acc:", acc / c, ", mean dist:", dist / c / grid * 100, ", max dist:", m / grid * 100) #mean dist in cm
                if flag:
                    torch.save(model, save_path)
                else:
                    torch.save(cnn.state_dict(), save_path)
