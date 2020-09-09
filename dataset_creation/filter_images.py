import json
import sys
import hashlib
import glob
from tqdm import tqdm
from collections import defaultdict
import os
import magic
import networkx as nx
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashes_storage", type = str, default = "data/hashes.json", help = "hashes file")
    parser.add_argument("--dict_hashes", type = str, default = "data/hashes_magic_dict.json", help = "Hashes conversion dict. ")
    parser.add_argument("--nodes_180k", type = str, default = "data/nodes_180k.json", help = "Initial nodes")
    parser.add_argument("--edges_180k", type = str, default = "data/edges_180k.json", help = "Initial edges. ")
    parser.add_argument("--marking_dict", type = str, default = "data/marking_dict.json", help = "Marking dict. ")
    parser.add_argument("--nodes_file", type = str, default = "data/nodes.json", help = "Output nodes file. ")
    parser.add_argument("--edges_file", type = str, default = "data/tuples.json", help = "Output edges/tuples file. ")
    args = parser.parse_args()

    with open(args.hashes_storage, "r") as f:
        hashes = json.loads(f.read())

    with open(args.dict_hashes, "r") as f:
        magic_dict = json.loads(f.read())

    with open(args.nodes_180k, "r") as f:
        nodes_180k = json.loads(f.read())

    with open(args.edges_180k, "r") as f:
        all_edges_180k = json.loads(f.read())

    with open(args.marking_dict, "r") as f:
        marking_dict = json.loads(f.read())

    counts = {}
    for key in tqdm(nodes_180k.keys(), mininterval=10):
        if key in magic_dict.keys():
            imgs = set(magic_dict[key])
            counts[key] = len(imgs.intersection(good_imgs))
        else:
            counts[key] = 0
    edggs = {}
    to_use = set([key for key, value in counts.items() if value >= 4])
    for key, value in all_edges_180k.items():
        if key in to_use:
            l = []
            for entry in value:
                if entry["s"] in to_use:
                    l.append(entry)
            edggs[key] = l


    with open(args.nodes_file, "w") as f:
        json.dump(new_nodes, f)

    with open(args.edges_file, "w") as f:
        json.dump(to_use_edg, f)
