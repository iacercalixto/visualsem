import urllib.request
import json
from utils import create_folder, from_lemma_to_ids, from_synsetID_to_images, get_edges_from_synset, return_core_graph, process_sense_info, process_imgs
from tqdm import tqdm
import time
import networkx as nx
#%matplotlib inline
#import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import os
import subprocess
import argparse

def process_sense_info_n(id_syn, placement, key=None):
    synset_info, synset_images = from_synsetID_to_images(id_syn, placement, key)
    senses = list(set([entry["properties"]["fullLemma"] for entry in synset_info["senses"]]))
    glosses = list(set([entry["gloss"] for entry in synset_info["glosses"]]))
    main_sense = synset_info["mainSense"]
    ims = [img["url"] for img in synset_images]
    source_ims = {img["url"].split("/")[-1]: img["urlSource"] for img in synset_images}
    #synset_id = id_syn
    #if "wn:" in id_syn:
    #    synset_id = list(set([sen["properties"]["synsetID"]["id"] for sen in synset_info["senses"]]))[0]
    return senses, glosses, main_sense, ims, source_ims

def get_nodes(file_name, current_nodes):
    with open(file_name + "_lts", "r") as f:
        mapping = json.loads(f.read())
    with open(file_name + ".sorted", "r") as f:
        new_lines = [l.split() for l in f.read().split("\n")][:-1]
    print("amount new extra nodes: ", str(len(new_lines)))
    for line in new_lines:
        current_nodes.add(mapping[line[0]])
    return current_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes_file", type = str, default = "data/nodes_180k.json", help = "Stored nodes file ")
    parser.add_argument("--edges_file", type = str, default = "data/edges_180k.json", help = "Initial 1000 nodes json file. ")
    parser.add_argument("--sources_file", type = str, default = "data/img_sources.json", help = "A json file with relation mapping; does not have to be there initially")
    parser.add_argument("--store_steps_nodes", type = str, default = "data/testing_n1000_15_", help = "Where to store the steps of each nodes getting")
    parser.add_argument("--k", type = int, default = 50, help = "How many neigboring nodes are considered at max per relation")
    parser.add_argument("--min_ims", type = int, default = 0, help = "Mimimum amount of images per node")
    parser.add_argument("--which_iter", type = int, default = 4, help = "How many iterations to get nodes")
    parser.add_argument("--placement", type = str, default = "localhost:8080", help = "Nabelnet api location")
    args = parser.parse_args()

    with open(args.nodes_file, 'r') as f:
        nodes = json.loads(f.read())

    curr_n = set(nodes.keys())
    for i in range(args.which_iter + 1):
        complete_line = args.store_steps_nodes + str(args.k) + "_" + str(args.min_ims) + "_r" + str(i)
        print("Previous unique nodes: ", str(len(curr_n)))
        curr_n = get_nodes(complete_line, curr_n)
        print("Next unique nodes: ", str(len(curr_n)))

    # For the images and the nodes
    img_dict = {}
    nodes_dict = {}
    img_sources = {}
    for n in tqdm(curr_n, mininterval=10):
        senses, glosses, main_sense, ims, source_ims = process_sense_info_n(n, args.placement)
        img_dict[n] = ims
        # gl = glosses, ms = main sense, se=senses
        nodes_dict[n] = {"gl": glosses, "ms": main_sense, "se": senses}
        img_sources[n] = source_ims

    with open(args.nodes_file, "w") as f:
        json.dump(nodes_dict, f)

    with open(args.sources_file, "w") as f:
        json.dump(img_sources, f)

    # Get the edges
    edges = []
    for i in [which_iter]:
        complete_line = args.store_steps_nodes + str(args.k) + "_" + str(args.min_ims) + "_r" + str(i) + "_edges"
        with open(complete_line, "r") as f:
            edg = json.loads(f.read())
        edges.append(edg)

    # restore edges
    all_edges = defaultdict(list)
    r_id = 0
    for edg in edges:
        for key, value in tqdm(edg.items(), mininterval=10):
            for rel, s in value:
                all_edges[key].append({"s": s, "r": rel, "r_id": r_id})
                r_id += 1
    # remove possible duplicates
    all_edges = {key:[dict(t) for t in {tuple(d.items()) for d in value}] for key, value in all_edges.items()}
    with open(args.edges_file, "w") as f:
        json.dump(all_edges, f)
