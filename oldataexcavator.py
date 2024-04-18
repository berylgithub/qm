import numpy as np
import pickle as pkl
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# save datas as json:
data = np.load("/users/baribowo/Code/Python/pes/data/hxoy_data.npy", allow_pickle=True)
json_dump = json.dumps(data)

#file = open('/users/baribowo/Code/Python/pes/result/cross_val_each_state_5fold_101221_031813.pkl', 'rb') 