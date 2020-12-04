import data_handler

data_paths=["/home/danai/Desktop/GaitTracking/p1/2.a"]#,"/home/danai/Desktop/GaitTracking/p5/2.a", "/home/danai/Desktop/GaitTracking/p11/2.a", "/home/danai/Desktop/GaitTracking/p11/3.a", "/home/danai/Desktop/GaitTracking/p16/3.a", "/home/danai/Desktop/GaitTracking/p17/3.a", "/home/danai/Desktop/GaitTracking/p18/2.a", "/home/danai/Desktop/GaitTracking/p18/3.a"]
data = data_handler.LegDataLoader(data_paths)
print("Loading dataset...")
_, _, val_set_x, val_set_y, _, _ = data.load(32)
for i in range(len(val_set_x)):
    print("Batch", i, "/", len(val_set_x))
    for j in range(len(val_set_x[i])):
        data_handler.print_data(val_set_x[i][j], val_set_y[i][j])
