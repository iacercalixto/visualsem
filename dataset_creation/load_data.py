from torchvision import transforms

from PIL import Image
import torch
import numpy as np
import glob, os

def load_image(image_path, width=300, height=300, resize=False, transform=None):
    """Load an image and convert it to a torch tensor.
        Input:
            - Path to the image
            - Desired width of the image; default 300
            - Desired height of the image; default 300 
            - Whether resizing is necessary; default False
        Output: 
            - Image that is converted to rgb and resized/transformed if given as parameters 
    """
    image = Image.open(image_path).convert("RGB")
    if resize:
        size = np.array([width, height])
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if transform:
        image = transform(image)

    return image

def get_data(pos_file, neg_file, batch_size, ratio, partial=None):
    # Pos file '../../../../images/data_cnn/positive_examples/*'
    # Neg file '../../../../images/data_cnn/negative_examples/*'
    # ratio = 0.9, partial = 3000, 2500, 2000 for example
    """
        Create a division of image data based on their file locations. 

        Input:
            - Location of the positive images, ending in /*
            - Location of the negative images, ending in /*
            - Desired batch size 
            - Ratio of train/test between 0 and 1; given ratio is for training, remainder for validation 
            - partial value; when you do not want to use all the data in a run. 

        Output:
            - List of batches containing image locations for training
            - List of batches containing image locations for validation
    """
    # Get file names
    pos_examples = [(i, 1) for i in glob.glob(pos_file)]
    neg_examples = [(i, 0) for i in glob.glob(neg_file)]

    if partial: # Use subset when necessary
        pos_examples = pos_examples[:partial]
        neg_examples = neg_examples[:partial]

    np.random.shuffle(pos_examples)
    np.random.shuffle(neg_examples)

    bound = int(len(pos_examples) * ratio)
    # Equal amount of positive and negative examples
    train, val = pos_examples[:bound] + neg_examples[:bound], pos_examples[bound:] + neg_examples[bound:]

    # Create randomized batches
    np.random.shuffle(train)
    np.random.shuffle(val)
    train_load = [train[i:i + batch_size] for i in range(0, len(train), batch_size)]
    val_load = [val[i:i + batch_size] for i in range(0, len(val), batch_size)]

    return train_load, val_load
