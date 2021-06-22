from PIL import Image
import numpy as np
import glob, os
#from cairosvg import svg2png
from tqdm import tqdm
import json
from utils import create_folder, from_lemma_to_ids, from_synsetID_to_images, get_edges_from_synset, return_core_graph, process_sense_info, process_imgs
import socket
import urllib.request
from multiprocessing.dummy import Pool as ThreadPool
import argparse

socket.setdefaulttimeout(5)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes_file", type = str, default = "data/nodes_180k.json", help = "Nodes file json")
    parser.add_argument("--where_to_store_images", type = str, default = "data/visualsem_images_before/", help = "First download of images")
    parser.add_argument("--placement", type = str, default = "localhost:8080", help = "Url for babelnet api")
    args = parser.parse_args()

    with open(args.nodes_file, "r") as f:
        nodes_180k = json.loads(f.read())
    print(len(nodes_180k))

    img_dict = {}
    total_count_imgs = 0
    extensions = set()
    for n in tqdm(nodes_180k.keys(), mininterval=10):
        _, synset_images = from_synsetID_to_images(n, args.placement)
        ims = []
        extensions_sub = []
        for img in synset_images:
            ims.append(img["url"])
            extensions_sub.append(img["url"].split(".")[-1])
        img_dict[n] = ims
        total_count_imgs += len(ims)
        extensions.update(extensions_sub)

    print(total_count_imgs)
    print(extensions)

    # Only keep correct extensions
    new_count = 0
    new_img_dict = {}
    for n in tqdm(nodes_180k.keys(), mininterval=10):
        new = []
        for i in img_dict[n]:
            ext = i.split(".")[-1].lower()
            if "jpeg" in ext or "png" in ext or "jpg" in ext or "svg" in ext or "tif" in ext or "gif" in ext:
                new.append(i)
        new_count += len(new)
        new_img_dict[n] = new
    img_dict = None
    print(new_count)

    tuple_list_urls = [(url, args.where_to_store_images + key + "/" + url.split("/")[-1])
                       for key, value in new_img_dict.items()
                        for url in value]

    batch_urls = [tuple_list_urls[i:i + 209845] for i in range(0, len(tuple_list_urls), 209845)]
    strings = []
    for batch in batch_urls:
        strr = ''
        for entry in batch:
            strr += entry[0] + '\n \t dir=' + args.where_to_store_images +  + entry[1].split("/")[-2] + ' \n \t out=' + entry[1].split("/")[-1] + '\n'
        strings.append(strr)

    for i, strr in enumerate(strings):
        with open("data/urls_" + str(i), "w") as f:
            f.write(strr)
