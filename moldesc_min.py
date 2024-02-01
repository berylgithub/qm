# minimum caller for feature extraction (only calls CMBDF (and np), nothing else)
import numpy as np
import cMBDF_joblib, cMBDF_300124, cMBDF_130823
import time

def extract_MBDF(x, version = "joblib"):
    print(x)
    ncs = np.load("data/qm9_ncs.npy", allow_pickle=True)  #np.load("data/test_ncs.npy", allow_pickle=True)
    coors = np.load("data/qm9_coors.npy", allow_pickle=True)  #np.load("data/test_coors.npy", allow_pickle=True)
    # timed extraction:
    start = time.time() # timer
    if version == "joblib":
        reps = cMBDF_joblib.generate_mbdf(ncs, coors, gradients=False, progress_bar = False, n_atm=2.0)
    elif version == "300124":
        reps = cMBDF_300124.generate_mbdf(ncs, coors, gradients=False, progress_bar = False, n_atm=2.0)
    else:
        reps = cMBDF_130823.generate_mbdf(ncs, coors)    
    print(time.time()-start) # end of timer
    print(reps)
    print(reps.shape)
    return reps


#extract_MBDF()