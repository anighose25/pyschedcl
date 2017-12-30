#!/usr/bin/env python
import pickle
import sys
import numpy as np
np.set_printoptions(threshold=np.inf)


with open(sys.argv[1], 'rb') as handle:
    buffer_info = pickle.load(handle)

for buf in buffer_info:
    print "OUTPUT BUFFER_SIZE : " + str(buf.size)
    print buf

