import torch
from matplotlib import pyplot as plt
import data_handler
import tracking_nn
import time

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


def printeval(true, pred):
    print(f"Recall score (weighted): {recall_score(true, pred, average='weighted')}")
    print(f"F1 score (weighted): {f1_score(true, pred, average='weighted')}")
    print(f"Precision score (weighted): {precision_score(true, pred, average='weighted')}")

img_side = 112
max_height = 1.2
min_height = 0.2
max_width = 0.5
min_width = -0.5
grid = 7

def print_data(img, found):
    y,x = torch.where(img)
    y = (img_side - y) / img_side * (max_height - min_height) + min_height
    x = x / img_side * (max_width - min_width) + min_width
    y1h,x1h,y2h,x2h = data_handler.find_center(found)
    y1h = (img_side - y1h) / img_side * (max_height - min_height) + min_height
    y2h = (img_side - y2h) / img_side * (max_height - min_height) + min_height
    x1h = x1h / img_side * (max_width - min_width) + min_width
    x2h = x2h / img_side * (max_width - min_width) + min_width
    plt.xlim(min_width, max_width)
    plt.ylim(min_height, max_height)
    plt.scatter(x,y, c = 'b', marker = '.')
    plt.scatter(x1h, y1h, c = 'r', marker = 'o')
    plt.scatter(x2h, y2h, c = 'y', marker = 'o')
    #plt.show()
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()

def check_out(batch, out, out1, states):
    for i in range(out.shape[0]):
        y = batch[i]
        yh = out[i]
        p1_h = yh[0, :, :]
        p2_h = yh[3, :, :]
        detect_cell1 = p1_h.reshape(-1).argmax(axis = 0)
        detect_cell2 = p2_h.reshape(-1).argmax(axis = 0)
        x_cell1, y_cell1 = detect_cell1 // grid, detect_cell1 % grid
        x_cell2, y_cell2 = detect_cell2 // grid, detect_cell2 % grid
        print("Found state", out1[i].argmax().item(), ", Real state", states[i].item())
        print_data(y, torch.tensor([[x_cell1 + out[i][1, x_cell1, y_cell1], y_cell1 + out[i][2, x_cell1, y_cell1]], [x_cell2 + out[i][4, x_cell2, y_cell2], y_cell2 + out[i][5, x_cell2, y_cell2]]]))


def eucl_dist(out, labels):
    ret = []
    for i in range(out.shape[0]):
        yh = out[i]
        p1_h = yh[0, :, :]
        p2_h = yh[3, :, :]
        detect_cell1 = p1_h.reshape(-1).argmax(axis = 0)
        detect_cell2 = p2_h.reshape(-1).argmax(axis = 0)
        x1, y1 = detect_cell1 // grid, detect_cell1 % grid
        x2, y2 = detect_cell2 // grid, detect_cell2 % grid
        ret.append(torch.sqrt((x1 + out[i, 1, x1, y1] - labels[i, 0, 0]) ** 2 + (y1 + out[i, 2, x1, y1] - labels[i, 0, 1]) ** 2).item())
        ret.append(torch.sqrt((x2 + out[i, 4, x2, y2] - labels[i, 1, 0]) ** 2 + (y2 + out[i, 5, x2, y2] - labels[i, 1, 1]) ** 2).item())
    return ret

def median(l):
    # l sorted list
    if len(l) % 2 == 0:
        return (l[len(l) // 2] + l[len(l) // 2 - 1]) / 2
    else:
        return l[len(l) // 2]

def find_classes(out):
    return out.argmax(axis=1)

def accuracy(out, states):
    classes = find_classes(out)
    return (classes[0] == states[0])
    
paths=["p18/2.a", "p18/3.a"]
data = data_handler.LegDataLoader(paths = paths, cg = 0)
print("Loading dataset...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = tracking_nn.CNN().to(device)
rnn = tracking_nn.RNN().to(device)
net = tracking_nn.Net(device, cnn, rnn).to(device)
net.load_state_dict(torch.load("model_p18_2a_p1_2a.pt", map_location=device).state_dict())
#net.to(device)
#net = torch.load("/home/danai/Desktop/GaitTracking/model_p18_2a_p1_2a.pt", map_location=device)
gnet = torch.load("gmodel.pt", map_location=device)
net.eval()
gnet.eval()
all_dists = []
f, input, states = data.load(2)
net.init_hidden()
gnet.init_hidden()
acc = 0
c = 1
all_out_states = None
all_states = None
with torch.no_grad():
    while True:
        if f:
            gnet.init_hidden()
            net.init_hidden()
        input, states = input.to(device), states.to(device)
        c += 1
        out = net(input)
        out1 = gnet(out)
        if all_states is not None:
            all_states = torch.cat([all_states, states])
            all_out_states = torch.cat([all_out_states, find_classes(out1)])
        else:
            all_out_states = find_classes(out1)
            all_states = states
        acc += accuracy(out1, states)
        #check_out(input.to(torch.device("cpu")), out.to(torch.device("cpu")), out1, states)
        if f == -1:
            break
        f, input, states = data.load(2)

all_dists.sort()
printeval(all_states.cpu(), all_out_states.cpu())
print("Gait acc:", acc / c * 100)
