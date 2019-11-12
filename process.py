import numpy as np
import os
with open("ceshi_val.txt") as f:
    lines = f.readlines()
label = []
for line in lines:
    line = line.rstrip()
    line = line.split(' ')
    label.append(line[3])
#print(label)
num = 0
for i in label:
    line = i.split(',')
    labelnew = np.zeros([21]).astype(int).reshape(1,21)
    for m in line:
        labelnew[0][int(m)] = 1
    if num == 0:
        new = labelnew
    else:
        new = np.concatenate((new, labelnew), axis=0)
    num = num + 1

print(new.shape)
np.save("ceshi_val.npy",new)
