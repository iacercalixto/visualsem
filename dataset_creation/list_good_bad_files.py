import json
import glob
from tqdm import tqdm
from PIL import Image
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder", type = str, default = "data/visualsem_images/*", help = "Where the visualsem images are stored; ending with the *. ")
    parser.add_argument("--good_examples_filename", type = str, default = "data/good_examples.json", help = "Where to store the valid image paths. ")
    parser.add_argument("--bad_examples_filename", type = str, default = "data/bad_examples.json", help = "Where to store the invalid image paths. ")
    args = parser.parse_args()

    good_examples = []
    bad_examples = []
    files = [j for i in glob.glob(args.images_folder) for j in glob.glob(i + "/*")]
    for file in tqdm(files, mininterval=10):
        try:
            im = Image.open(file)
            good_examples.append(file)
        except:
            bad_examples.append(file)

    print(len(good_examples))
    print(len(bad_examples))
    print(len(good_examples)/(len(good_examples) + len(bad_examples))*100)

    with open(args.good_examples_filename, "w") as f:
        json.dump(good_examples, f)

    with open(args.bad_examples_filename, "w") as f:
        json.dump(bad_examples, f)
