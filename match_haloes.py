from read_rockstar import *

    
iout = 13
root = './test_data/biased/'

h, pid, header = read_binary_haloes(root, iout)

for ih in h:
    print(ih['m'])
