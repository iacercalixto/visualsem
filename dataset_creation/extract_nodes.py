import urllib.request
import json
from utils import from_lemma_to_ids, from_synsetID_to_images, get_edges_from_synset, return_core_graph, process_sense_info, process_imgs
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

rel_mapping = {"is_a": "IsA", "is-a": "IsA", "has-kind": "HasA", "has_kind": "HasA", "related": "RelatedTo",
               "has-part": "HasA", "has_part": "HasA", "use": "UsedFor", "used_by": "UsedBy", "used-by": "UsedBy",
               "subject_of": "SubjectOf", "subject-of": "SubjectOf", "depicts": "DefinedAs", "depicts_by": "DefinedAs",
               "depicts-by": "DefinedAs", "interaction": "ReceivesAction", "oath_made_by": "MadeOf",
               "also-see": "AlsoSee", "gloss-related": "GlossRelated", "taxon-synonym": "Synonym", "part_of": "PartOf",
               "part-of": "PartOf"}
part_rel_mapping = {"location": "AtLocation", "has_": "HasProperty", "located_": "AtLocation"}
set_of_rels = set(rel_mapping.values()).union(set(part_rel_mapping.values()))

def normalize_relation(old_rel):
    if old_rel in rel_mapping:
        return rel_mapping[old_rel]
    else:
        for k, v in part_rel_mapping.items():
            if k in old_rel:
                return v
        return None

def get_nodes(file_name, current_nodes):
    with open(file_name + "_lts", "r") as f:
        mapping = json.loads(f.read())
    with open(file_name + ".sorted", "r") as f:
        new_lines = [l.split() for l in f.read().split("\n")][:-1]
    print("amount new extra nodes: ", str(len(new_lines)))
    for line in new_lines:
        current_nodes.add(mapping[line[0]])
    return current_nodes

def create_mapping(set_of_rels):
    # Map the textual relations to a number
    if os.path.exists(relations_file):
        with open(relations_file, "r") as f:
            mapping = json.loads(f.read())
    else:
        mapping = {}
        for index, rel in enumerate(set_of_rels):
            mapping[rel] = "k" + str(index + 1)
        with open(relations_file, "w") as f:
            json.dump(mapping, f)
    return mapping

def to_file(out, u_rels, k, min_ims, complete_line):
    # Convert the found relations and synsets back to a file
    line_to_synset = {}
    with open(complete_line, "w") as f:
        for i, (key, value) in enumerate(out.items()):
            if value:
                line_to_synset[i] = key
                f.write(str(i) + " " + " ".join([str(v) for v in value.values()]) + "\n")
    with open(complete_line + "_lts", "w") as f:
        json.dump(line_to_synset, f)
    with open(complete_line + "_edges", "w") as f:
        json.dump(u_rels, f)
    print("done")

def edge_information_small(synset_id, placement, run=None, nns=None):
    edges_info = get_edges_from_synset(synset_id, placement)
    if run:
        new =  [{"s":synset_id, "e": edg["target"], "rts": normalize_relation(edg["pointer"]["shortName"]),
                 "rtl": edg["pointer"]["name"], "w": edg["weight"]} for edg in edges_info if edg["target"] not in nns]
    else:
        new =  [{"s":synset_id, "e": edg["target"], "rts": normalize_relation(edg["pointer"]["shortName"]),
                 "rtl": edg["pointer"]["name"], "w": edg["weight"]} for edg in edges_info]
    new = [edg for edg in new if edg["rts"]]
    return new

def return_sets(list_of_neighbors, k, stats, u_rels):
    each_set = defaultdict(set)

    for neigh in list_of_neighbors:
        each_set[neigh["rts"]].add((neigh["e"], neigh["s"]))

    for key, v in each_set.items():
        v_l = list(v)
        np.random.shuffle(v_l)
        for (end, start) in v_l[:k]:
            stats[end][key] += 1
            u_rels[end].append((key, start))
    # Clean var
    each_set = None

def process_sense_info_small(id_syn, placement, key=None):
    _, synset_images = from_synsetID_to_images(id_syn, placement, key)
    num_ims = len(synset_images)
    return num_ims

def create_entries(stats, u_rels, mapping, min_ims, placement):
    # Calculate statistics per node
    entry_dict = defaultdict(dict)
    l = len(set_of_rels)
    l_1 = l - 2
    for key, new_dict in tqdm(stats.items(), mininterval=10):
        new_d = {}
        sum_1 = 0
        sum_2 = 0
        sum_3 = 0
        sum_4 = 0
        for r in set_of_rels:
            curr_val = new_dict[r]
            new_d[mapping[r]] = curr_val
            if curr_val > 0:
                if r != "HasA" and r != "RelatedTo":
                    sum_2 += 1
                sum_1 += 1
            if r != "HasA" and r != "RelatedTo":
                sum_4 += curr_val
            sum_3 += curr_val

        new_d["rels_1_with"] = sum_1
        new_d["rels_1_without"] = sum_2
        new_d["k_avg_with"] = sum_3/l
        new_d["k_avg_without"] = sum_4/l_1
        new_d["imgs"] = process_sense_info_small(key, placement)
        entry_dict[key] = new_d
    return entry_dict


def do_steps(nodes, set_of_rels, k, min_ims, placement):
    mapping = create_mapping(set_of_rels)
    print("step 1 done...")
    # Get all the neighboring information
    stats = defaultdict(lambda: defaultdict(int))
    u_rels = defaultdict(list)
    for n in tqdm(nodes, mininterval=10):
        new = edge_information_small(n, placement)
        return_sets(new, k, stats, u_rels)
    print("step 2 done.....")
    entry_dict = create_entries(stats, u_rels, mapping, min_ims, placement)
    print("final step done...")
    return entry_dict, u_rels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_node_file", type = str, default = "data/nodes_1000.json", help = "Initial 1000 nodes json file. ")
    parser.add_argument("--relations_file", type = str, default = "data/rel_to_k.json", help = "A json file with relation mapping; does not have to be there initially. ")
    parser.add_argument("--store_steps_nodes", type = str, default = "data/testing_n1000_15_", help = "Where to store the steps of each nodes getting. ")
    parser.add_argument("--k", type = int, default = 50, help = "How many neigboring nodes are considered at max per relation.")
    parser.add_argument("--min_ims", type = int, default = 0, help = "Mimimum amount of images per node. ")
    parser.add_argument("--M", type = int, default = 6, help = "How many iterations to get nodes. ")
    parser.add_argument("--placement", type = str, default = "localhost:8080", help = "Nabelnet api location. ")
    args = parser.parse_args()

    with open(args.initial_node_file) as f:
        nodes = json.loads(f.read())

    curr_n = set(nodes.keys())
    for i in range(args.M):
        out, u_rels = do_steps(curr_n, set_of_rels, args.k, args.min_ims, args.placement)
        complete_line = args.store_steps_nodes + str(args.k) + "_" + str(args.min_ims) + "_r" + str(i)
        complete_line_s = args.store_steps_nodes + str(args.k) + "_" + str(args.min_ims) + "_r" + str(i) + ".sorted"
        to_file(out, u_rels, args.k, args.min_ims, complete_line)
        cmd = 'cat ' + complete_line + " | awk '{ if (($21 >= 1) && ($18 + $17 >= 2)) { print } }' | sort -shr -k18,18 -k20,20 -k17,17 -k19,19 -k3,3 -k4,4 -k5,5 -k6,6 -k8,8 -k9,9 -k10,10 -k11,11 -k12,12 -k13,13 -k14,14 -k15,15 -k16,16 -k7,7h -k2,2h -k21,21hr > " + complete_line_s
        p = subprocess.Popen(['/bin/bash','-c', cmd], stdout=subprocess.PIPE)
        p.wait()
        print("Previous unique nodes: ", str(len(curr_n)))
        curr_n = get_nodes(complete_line, curr_n)
        print("Next unique nodes: ", str(len(curr_n)))
