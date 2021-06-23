import sys
import os
import json
import torch
from torch.utils.data import Dataset
from utils import load_visualsem_bnids


class VisualSemNodesDataset(torch.utils.data.Dataset):
    """
        Dataset class that can be used to iterate all nodes in VisualSem (linking all data available in a node).
        Nodes are associated to images, multilingual glosses, and tuples
        (i.e. its tuples include all nodes with which it is a tail in VisualSem).
    """
    def __init__(self, path_to_nodes, path_to_glosses, path_to_tuples, path_to_images=None):
        """
            path_to_nodes(str):         Path to JSON file containing VisualSem nodes.
            path_to_glosses(str):       Path to JSON file containing VisualSem glosses.
            path_to_tuples(str):        Path to JSON file containing VisualSem tuples.
            path_to_images(str):        Path to directory containing VisualSem images. (Optional)
        """
        super(VisualSemNodesDataset).__init__()
        
        assert(os.path.isfile( path_to_nodes )), "File not found: %s"%path_to_nodes
        assert(os.path.isfile( path_to_tuples )), "File not found: %s"%path_to_tuples
        assert(os.path.isfile( path_to_glosses )), "File not found: %s"%path_to_glosses
        assert( path_to_images is None or os.path.isfile( path_to_images )), "File not found: %s"%path_to_images

        bnids = load_visualsem_bnids(path_to_nodes, path_to_images)

        self.nodes = {}
        with open(path_to_nodes, 'r') as fh:
            nodes_json = json.load(fh)
            for node_key, node_value in nodes_json.items():
                # initialize node
                self.nodes[ node_key ] = {
                    "ms" : node_value['ms'],
                    "se" : node_value['se'],
                    "images": node_value["ims"],
                }

        with open(path_to_tuples, 'r') as fh:
            tuples_json = json.load(fh)
            for tuple_key, tuple_value in tuples_json.items():
                if not "incoming_nodes" in self.nodes[ tuple_key ]:
                    self.nodes[ tuple_key ][ "incoming_nodes" ] = []

                for entry in tuple_value:
                    self.nodes[ tuple_key ][ "incoming_nodes" ].append({
                        "head" : entry['s'],
                        "tail" : tuple_key,
                        "relation" : entry['r'],
                        "tuple_id" : entry['r_id']
                    })

        with open(path_to_glosses, 'r') as fh:
            glosses_json = json.load(fh)
            for gloss_entry in glosses_json[1:]:
                assert(len(gloss_entry)==1)
                key = list(gloss_entry.keys())[0]
                self.nodes[ key ][ "glosses" ] = gloss_entry[key]
        
        self.bnids = bnids

    def __getitem__(self, index):
        return self.nodes[ self.bnids[index] ]

    def __len__(self):
        return len(self.bnids)


if __name__=="__main__":
    dir_path    = os.path.dirname(os.path.realpath(__file__))
    nodes_json   = os.path.join(dir_path, "dataset", "nodes.v2.json")
    glosses_json = os.path.join(dir_path, "dataset", "gloss_files", "nodes.glosses.json")
    tuples_json  = os.path.join(dir_path, "dataset", "tuples.v2.json")
    # testing node dataset
    print("Testing node dataset...")
    vs = VisualSemNodesDataset(nodes_json, glosses_json, tuples_json)
    print("len(vs): ", len(vs))
    print(vs[0])

