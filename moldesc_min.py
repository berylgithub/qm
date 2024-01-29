# minimum caller for feature extraction (only calls CMBDF (and np), nothing else)
import numpy as np
import cMBDF_joblib
import time

def extract_MBDF():
    ncs = np.load("data/test_ncs.npy", allow_pickle=True)
    coors = np.load("data/test_coors.npy", allow_pickle=True)
    # timed extraction:
    start = time.time() # timer
    reps = cMBDF_joblib.generate_mbdf(ncs, coors, gradients=False, progress_bar = False, n_atm=2.0)
    print(time.time()-start) # end of timer
    print(reps)
    print(reps.shape)


extract_MBDF()