with open("../../p5/2.a/centers.csv", 'r') as f:
    with open("../../p5/2.a/centers1.csv", 'w') as f1:
        for i in f:
            x = i.strip().split(",")
            f1.write(",".join([x[2],x[3],x[0],x[1]]) + "\n")
