import sys
import os
import json
import torch
from torch.utils.data import Dataset


class VisualSemTuplesDataset(torch.utils.data.Dataset):
    """
        Dataset class that can be used to iterate all tuples in VisualSem.
        Each tuple consists of a (h, r, t) entry denoting that
        head node `h` is related to tail node `t` through relation `r`.
    """
    def __init__(self, path_to_tuples):
        """
            path_to_tuples(str):        Path to JSON file containing VisualSem tuples.
        """
        super(VisualSemTuplesDataset).__init__()
        assert(os.path.isfile( path_to_tuples )), "File not found: %s"%path_to_tuples

        self.tuples = []
        with open(path_to_tuples, 'r') as fh:
            tuples_json = json.load(fh)
            for tuple_key, tuple_value in tuples_json.items():
                for entry in tuple_value:
                    self.tuples.append({
                        "head" : entry['s'],
                        "tail" : tuple_key,
                        "relation" : entry['r'],
                        "tuple_id" : entry['r_id']
                    })

        self.tuples = sorted(self.tuples, key=lambda kv:(kv["head"], kv["tail"]))

    def __getitem__(self, index):
        return self.tuples[ index ]

    def __len__(self):
        return len(self.tuples)


if __name__=="__main__":
    dir_path    = os.path.dirname(os.path.realpath(__file__))
    tuples_json  = os.path.join(dir_path, "dataset", "tuples.v2.json")
    # testing tuples dataset
    print("Testing tuple dataset...")
    vs = VisualSemTuplesDataset(tuples_json)
    print("len(vs): ", len(vs))
    print(vs[0])
