# minimum caller for feature extraction (only calls CMBDF (and np), nothing else)
import numpy as np
import cMBDF

def extract_MBDF():
    print(np.load("/users/baribowo/Code/Julia/qm/data/test_ncs.npy", allow_pickle=True))
    print(np.load("/users/baribowo/Code/Julia/qm/data/test_coors.npy", allow_pickle=True))


extract_MBDF()