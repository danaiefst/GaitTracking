import torch

torch.cuda.manual_seed(3)
torch.manual_seed(3)

import data_handler, tracking_nn
import sys
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

flag = int(sys.argv[1])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on", device)
#path = "/home/athdom/GaitTracking/"
#path = "/home/iral-lab/GaitTracking/"
path = "/home/danai/Desktop/GaitTracking/"
paths = ["p1/2.a", "p11/2.a", "p17/2.a", "p17/3.a", "p18/2.a", "p18/3.a"]
data_path = path + "data/" 
batch_size = 32

cnn = tracking_nn.CNN().to(device)
#if flag:
#    cnn.load_state_dict(torch.load(path + "cnn_model.pt", map_location = device))
#    for param in cnn.parameters():
#        param.requires_grad = False
rnn = tracking_nn.RNN().to(device)
model = tracking_nn.Net(device, cnn, rnn).to(device)
model.load_state_dict(torch.load(path + "model.pt", map_location = device))
for param in model.parameters():
    param.requires_grad = False
gnet = tracking_nn.GNet(device).to(device)
data = data_handler.LegDataLoader(gait = 1, batch_size = batch_size, data_path = data_path, paths = paths)
# Train the nn

epochs = 1000
patience = 0
learning_rate = 0.0001
grid = 7
optimizer = Adam(gnet.parameters(), lr = learning_rate)
best_loss = float("Inf")
if flag == 1:
    save_path = path + "model.pt"
elif flag == 0:
    save_path = path + "cnn_model.pt"
else:
    save_path = path + "gmodel.pt"

def accuracy(out, states):
    classes = out.argmax(axis=1)
    return (classes == states).sum() / len(classes) * 100
    

print("Started training...")
for epoch in range(epochs):
    running_loss = 0
    if epoch == 3 or epoch == 7 or epoch == 10:
        learning_rate *= 0.1
        optimizer = Adam(gnet.parameters(), lr = learning_rate)
    #if epoch == 17:
    #    learning_rate /= 0.1
    #    optimizer = Adam(gnet.parameters(), lr = learning_rate)
    f, input, label, states = data.load(0)
    model.init_hidden()
    gnet.init_hidden()
    c = 0
    while(True):
        if f:
            model.init_hidden()
            gnet.init_hidden()
        input, label, states = input.to(device), label.to(device), states.to(device)
        optimizer.zero_grad()
        output = model.forward(input)
        output = gnet.forward(output)
        #print("labels", labels[0])
        loss = gnet.loss(output, states)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        c += 1
        if f == -1:
            break
        f, input, label, states = data.load(0)
        #model.init_hidden()
        #model.detach_hidden()
        gnet.detach_hidden()
    print("epoch:{}, running loss: {}".format(epoch, running_loss / c))
    running_loss = 0
    if epoch >= patience:
        with torch.no_grad():
            running_loss = 0
            acc = 0
            c = 0
            f, input, label, states = data.load(1)
            model.init_hidden()
            gnet.init_hidden()
            while(True):
                if f:
                    model.init_hidden()
                    gnet.init_hidden()
                input, label, states = input.to(device), label.to(device), states.to(device)
                output = model.forward(input)
                output = gnet.forward(output)
                running_loss += gnet.loss(output, states)
                acc += accuracy(output, states)
                c += 1
                if f == -1:
                    break
                f, input, label, states = data.load(1)
                #model.init_hidden()
            if running_loss < best_loss:
                best_loss = running_loss
                print("Saving model with loss:", best_loss / c, "acc:", acc / c)
                if flag == 1:
                    torch.save(model.state_dict(), save_path)
                elif flag == 0:
                    torch.save(cnn.state_dict(), save_path)
                else:
                    torch.save(gnet.state_dict(), save_path)
