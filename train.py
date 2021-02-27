import torch

torch.cuda.manual_seed(3)
torch.manual_seed(3)

import data_handler, tracking_nn
import sys
from torch.optim import Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Working on", device)
paths = ["p1/2.a", "p5/2.a", "p11/2.a", "p16/3.a", "p17/2.a", "p17/3.a", "p18/2.a", "p18/3.a"]
#paths = ["p1/2.a", "p18/2.a", "p18/3.a"]
batch_size = 64

model = torch.load("model.pt", map_location = device)
model.eval()
gnet = tracking_nn.GNet(device).to(device)
data = data_handler.LegDataLoader(batch_size = batch_size, paths = paths)
# Train the nn

epochs = 1000
patience = 0
learning_rate = 0.0001
grid = 7
optimizer = Adam(gnet.parameters(), lr = learning_rate)
best_loss = float("Inf")
save_path = "gmodel.pt"

def accuracy(out, states):
    classes = out.argmax(axis=1)
    return (classes == states).sum() / len(classes) * 100
    

print("Started training...")
for epoch in range(epochs):
    running_loss = 0
    if epoch == 1:# or epoch == 26:# or epoch == 30:
        learning_rate *= 0.1
        optimizer = Adam(gnet.parameters(), lr = learning_rate)
    #if epoch == 17:
    #    learning_rate /= 0.1
    #    optimizer = Adam(gnet.parameters(), lr = learning_rate)
    f, input, states = data.load(0)
    model.init_hidden()
    gnet.init_hidden()
    c = 0
    while(True):
        if f:
            model.init_hidden()
            gnet.init_hidden()
        #print(input.shape, states.shape)
        input, states = input.to(device), states.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            output = model(input)
        #print(output)
        output = gnet.forward(output)
        loss = gnet.loss(output, states)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        c += 1
        if f == -1:
            break
        f, input, states = data.load(0)
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
            f, input, states = data.load(1)
            model.init_hidden()
            gnet.init_hidden()
            while(True):
                if f:
                    model.init_hidden()
                    gnet.init_hidden()
                input, states = input.to(device), states.to(device)
                output = model(input)
                output = gnet.forward(output)
                running_loss += gnet.loss(output, states).item()
                acc += accuracy(output, states).item()
                c += 1
                if f == -1:
                    break
                f, input, states = data.load(1)
                #model.init_hidden()
            if running_loss < best_loss:
                best_loss = running_loss
                print("Saving model with loss:", best_loss / c, "acc:", acc / c)
                torch.save(gnet, save_path)
            else:
                print("Not saving, acc", acc / c)
