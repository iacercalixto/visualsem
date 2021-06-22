from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import numpy as np
import glob, os
import argparse
from load_data import load_image, get_data
from tqdm import tqdm
from lenet import LeNet
from resnet import ResNet152
from vgg import VGG19_BN
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

MODELS = {"lenet": 1, "vgg": 2, "resnet": 3}
TRANSFORMS = {1: transforms.Compose([transforms.ToTensor(), normalize]),
              2: transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize]),
              3: transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize])}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type = int, default = 80, help = "batch size")
    parser.add_argument("--width", type = int, default = 300, help = "width images") # Only used for lenet
    parser.add_argument("--height", type = int, default = 300, help = "height images") # Only used for lenet
    parser.add_argument("--model", type = str, default = "resnet", help = "chosen model, lenet, resnet, vgg")
    parser.add_argument("--file_data", type = str, default = 'data/good_examples.json', help = "valid images files")
    parser.add_argument("--resize", type = bool, default = False, help = "resize images, only for lenet is true.")
    parser.add_argument("--img_store", type = str, default = "data/visualsem_images")
    parser.add_argument("--marking_dict", type = str, default = "data/marking_dict.json", help = "marking dict")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.model == "lenet":
        net = LeNet(args.width).to(device)
        net.load_state_dict(torch.load("data/lenet_test3000_19"))
    elif args.model == "resnet":
        net = ResNet152().to(device)
        net.load_state_dict(torch.load("data/resnet_test3000_26"))
    elif args.model == "vgg":
        net = VGG19_BN().to(device)
        net.load_state_dict(torch.load("data/vgg_test3000_28"))

    net.eval()
    transform = TRANSFORMS[MODELS[args.model]]

    with open(args.file_data, "r") as f:
        val = json.loads(f.read())

    #with open("../marking_dict.json", "r") as f:
#        marking_dict = json.loads(f.read())

    #print(val[:10])
    #val = list(set(val).difference(set(marking_dict.keys())))
    #val.remove("c8377613eae3302038e0f9844d2b3691cd20c529")
    #val.remove("cb96674e1a626d386ff96ceed15072c64bf837b6")
    #val.remove("b3f7aefa58fd4d2c1cbd107f1dc89b8f5604ae7d")
    #print(len(val))
    #val.remove("193a1c95ee51356e1c02c229b9b83b2b57759f5e")
    #val.remove("10053d8e432a031b7610c9bd4549f99285afdde4")
    val_load = [val[i:i + args.batch] for i in range(0, len(val), args.batch)]
    print(val_load[0])

    for batch_id, batch in tqdm(enumerate(val_load), mininterval=10):
        inputs = torch.stack(tuple([load_image(args.img_store + i[:2] + "/" + i + ".jpg", args.width, args.height, args.resize, transform) for i in batch]), 0).to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        for i, elem in enumerate(batch):
            marking_dict[elem] = predicted[i].item()

        with open(args.marking_dict, "w") as f:
            json.dump(marking_dict, f)
