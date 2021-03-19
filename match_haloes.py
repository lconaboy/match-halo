# import os
# import sys
# import glob
# import numpy as np
# from rockstar_structs import head_struct, halo_struct

from read_rockstar import *

    
iout = 13
root = './test_data/biased/'

h, pid, header = read_binary_haloes(root, iout)

for ih in h:
    print(ih['m'])
