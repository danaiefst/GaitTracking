import data_handler

data = data_handler.LegDataLoader(batch_size = 32, data_path = "/home/danai/Desktop/GaitTracking/data/", paths = ["p1/2.a", "p18/2.a", "p18/3.a"])
f, input, label = data.load(0)
c = 1
while(True):
    #for i in range(len(input)):
        #print(label[i])
    #    data_handler.print_data(input[i], label[i], fast = 0.1)
    if f == -1:
        break
    f, input, label = data.load(0)
    c += 1
print(c)
