import os
import subprocess
import glob
import argparse
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_images(files):
    for file in tqdm(files, mininterval = 10):
        if ".jpg" in file:
            continue
        try:
            im = Image.open(file)
            process_one_img(file, im)
        except:
            print(file)

def process_one_img(file, im):
    if im.mode == "RGB":
        im.thumbnail((400, 400), Image.ANTIALIAS)
        im.save(file + ".jpg")
        os.remove(file)
    else:
        im = im.convert("RGB")
        im.thumbnail((400, 400), Image.ANTIALIAS)
        im.save(file + ".jpg")
        os.remove(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder", type = str, default = "data/visualsem_images/*", help = "Where the visualsem images are stored; ending with the *. ")
    args = parser.parse_args()
    files_all = glob.glob(args.images_folder)
    for files in files_all:
        print(files)
        process_images(glob.glob(files + "/*"))
