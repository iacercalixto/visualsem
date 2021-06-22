import sys
import hashlib
import glob
from tqdm import tqdm
from collections import defaultdict
import os
import json

BUF_SIZE = 65536

# Hash a file with the sha1 hash
def hash_file(file_name):
    sha1 = hashlib.sha1()
    with open(file_name, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

# Make sure that folders are present
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--FOLDER", type = str, default = "data/visualsem_images_before/*", help = "Initial downloaded folder with /*")
    parser.add_argument("--NEW_FOLDER", type = str, default = "data/visualsem_images/", help = "New folder for visualsem images")
    parser.add_argument("--sources_file", type = str, default = "data/img_sources.json", help = "Image sources")
    parser.add_argument("--hash_to_source", type = str, default = "data/hash_to_source.json", help = "Where to store hash to source dict ")
    parser.add_argument("--hashes_magic_dict", type = str, default = "data/hashes_magic_dict.json", help = "Where to store hashes dictionary")
    parser.add_argument("--hashes", type = str, default = "data/hashes.json", help = "Where to store hashes")
    args = parser.parse_args()

    images = [j for i in glob.glob(args.FOLDER) for j in glob.glob(i + '/*')]

    with open(args.sources_file, 'r') as f:
        sources_img = json.loads(f.read())

    # if starting from no earlier files, comment below two lines
    hashes = set()
    new_d = defaultdict(set)
    hash_to_source = defaultdict(list)

    # Now start the duplicate removal
    for im in tqdm(images, mininterval = 10):
        synset = im.split("/")[-2]
        hashh = hash_file(im)
        new_d[synset].add(hashh)
        hash_to_source[hashh].append(sources_img[synset][im.split("/")[-1]])
        if hashh not in hashes:
            hashes.add(hashh)
            create_folder(args.NEW_FOLDER + hashh[:2])
            os.rename(im, args.NEW_FOLDER + hashh[:2] + '/' + hashh)
        else:
            os.remove(im)

    hash_to_source = {k : list(set(v)) for k, v in hash_to_source.items()}

    with open(args.hash_to_source, "w") as f:
        json.dump(hash_to_source, f)

    with open(args.hashes_magic_dict, "w") as f:
        json.dump(new_d, f)

    with open(args.hashes, "w") as f:
        json.dump(hashes, f)
