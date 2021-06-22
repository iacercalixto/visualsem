import sys, os
import json
from collections import defaultdict
import tqdm
import argparse
import numpy
import torch
from clip import clip
from PIL import Image
import spacy
#from spacy.lang.en import English
nlp = spacy.load("en_core_web_trf")
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
spacy_tokenizer = nlp.tokenizer

#MODEL_NAME = "ViT-B/32"
MODEL_NAME = "RN50x4"


def get_nlines(fname):
    with open(fname, 'r') as fh:
        lines = 0
        for line in fh:
            lines += 1
        return lines


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)


def load_visualsem_english_glosses(split_idx, max_splits):
    """ Returns a dictionary with VisualSem node idxs as keys and a list of English glosses as values.

        The parameter `split_idx` and `max_splits` tell us what to include.
        We split nodes in equal `max_split` parts, and only return nodes from `split_idx`.
    """
    en_glosses_fname = os.path.join(path_to_vs, 'dataset', "gloss_files/glosses.en.txt")
    en_glosses_nodes_fname = os.path.join(path_to_vs, 'dataset', "gloss_files/glosses.en.txt.bnids")
    nodes_fname = os.path.join(path_to_vs, 'dataset', "nodes.json")

    print("Loading glosses...")
    # load nodes JSON (we only load the nodes JSON to get the overall number of nodes and split examples per GPU)
    with open(nodes_fname, 'r') as fh:
        nodes_json = json.load(fh)

    n_nodes = len(nodes_json)
    # last split could have less nodes than others, but that's negligeable
    n_nodes_per_split = (n_nodes // max_splits) + 1

    print("Including glosses from idx %i (inclusive) to %i (exclusive) out of %i total number of nodes..."%(
        n_nodes_per_split * split_idx, n_nodes_per_split * (split_idx+1), n_nodes))

    # load all glosses
    glosses_dict_all = defaultdict(list)
    with open(en_glosses_fname, 'r') as fh_gl:
        with open(en_glosses_nodes_fname, 'r') as fh_nodes:
            for gloss, node in tqdm.tqdm(zip(fh_gl, fh_nodes), total=get_nlines(en_glosses_fname)):
                gloss = gloss.strip()
                node = node.strip()
                glosses_dict_all[ node ].append( gloss )

    # add filtered glosses to dictionary
    glosses_dict = defaultdict(list)
    for idx, (node,_) in tqdm.tqdm(enumerate(nodes_json.items()), total=n_nodes):
        if idx >= n_nodes_per_split * split_idx and idx < n_nodes_per_split * (split_idx+1):
            glosses_dict[ node ].extend( glosses_dict_all[ node ] )
    print()
    return glosses_dict


def load_visualsem_image_fnames(split_idx, max_splits):
    """ Returns a dictionary with VisualSem node idxs as keys and a list of (the full path to) image file names.

        The parameter `split_idx` and `max_splits` tell us what to include.
        We split nodes in equal `max_split` parts, and only return nodes from `split_idx`.
    """
    images_path = os.path.join(path_to_vs, 'dataset', "images")
    nodes_fname = os.path.join(path_to_vs, 'dataset', "nodes.v2.json")

    print("Loading image file names...")
    # load nodes JSON
    with open(nodes_fname, 'r') as fh:
        nodes_json = json.load(fh)

    n_nodes = len(nodes_json)
    # last split could have less nodes than others, but that's negligeable
    n_nodes_per_split = (n_nodes // max_splits) + 1

    print("Including image from idx %i (inclusive) to %i (exclusive) out of %i total number of nodes..."%(
        n_nodes_per_split * split_idx, n_nodes_per_split * (split_idx+1), n_nodes))

    # retrieve images for each node
    images_dict = defaultdict(list)
    for idx, node in tqdm.tqdm(enumerate(nodes_json), total=n_nodes):
        if idx >= n_nodes_per_split * split_idx and idx < n_nodes_per_split * (split_idx+1):
            node_idx, node_entry = node, nodes_json[ node ]
            #rint(node_idx, node_entry['ims'])
            for im in node_entry['ims']:
                fullpath = os.path.join(images_path, im[0:2], "%s.jpg"%im)
                #assert( os.path.isfile( fullpath )), "Could not find file: %s"%fullpath
                images_dict[ node_idx ].append( fullpath )
    print()

    return images_dict


def load_visualsem_image_splits(split_idx, max_splits):
    images_path = os.path.join(path_to_vs, 'dataset', "images")
    nodes_fname = os.path.join(path_to_vs, 'dataset', "nodes.v2.json")
    train_split_path = os.path.join(path_to_vs, 'dataset', "train_images.json")
    valid_split_path = os.path.join(path_to_vs, 'dataset', "valid_images.json")
    test_split_path  = os.path.join(path_to_vs, 'dataset', "test_images.json")

    with open(train_split_path, 'r') as fh:
        train_splits = json.load(fh)
        train_splits = set(train_splits)

    with open(valid_split_path, 'r') as fh:
        valid_splits = json.load(fh)
        valid_splits = set(valid_splits)

    with open(test_split_path, 'r') as fh:
        test_splits = json.load(fh)
        test_splits = set(test_splits)

    print("Loading image file names...")
    # load nodes JSON
    with open(nodes_fname, 'r') as fh:
        nodes_json = json.load(fh)

    n_nodes = len(nodes_json)
    # last split could have less nodes than others, but that's negligeable
    n_nodes_per_split = (n_nodes // max_splits) + 1

    print("Including image from idx %i (inclusive) to %i (exclusive) out of %i total number of nodes..."%(
        n_nodes_per_split * split_idx, n_nodes_per_split * (split_idx+1), n_nodes))

    # retrieve images for each node
    train_images_dict = defaultdict(list)
    valid_images_dict = defaultdict(list)
    test_images_dict  = defaultdict(list)
    for idx, node in tqdm.tqdm(enumerate(nodes_json), total=n_nodes):
        if idx >= n_nodes_per_split * split_idx and idx < n_nodes_per_split * (split_idx+1):
            node_idx, node_entry = node, nodes_json[ node ]
            #rint(node_idx, node_entry['ims'])
            for im in node_entry['ims']:
                fullpath = os.path.join(images_path, im[0:2], "%s.jpg"%im)
                #assert( os.path.isfile( fullpath )), "Could not find file: %s"%fullpath
                if im in train_splits:
                    train_images_dict[ node_idx ].append( fullpath )
                elif im in valid_splits:
                    valid_images_dict[ node_idx ].append( fullpath )
                elif im in test_splits:
                    test_images_dict[ node_idx ].append( fullpath )
                else:
                    raise Exception()

    print()

    return train_images_dict, valid_images_dict, test_images_dict


def load_clip_model(device):
    print("Loading pretrained CLIP model on %s..."%device, end=" ")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    print()
    return model, preprocess


def process_image_to_gloss(model, device, preprocess, gls, gls_bnids, valid_ims, test_ims, one_gloss_per_node, split_names):
        # load English gloss features and idxs
        #split_name = "valid"
        n_english_glosses = sum([len(gs) for bnidx, gs in gls.items()])

        if one_gloss_per_node:
            # load/compute split image features
            fname_split_gloss_feats = os.path.join(path_to_vs, 'dataset', "visualsem-gloss-features.train.CLIP-%s.one-gloss-per-node.npz"%(MODEL_NAME))
            # load features if they have already been computed
            if os.path.isfile( fname_split_gloss_feats ):
                with numpy.load(fname_split_gloss_feats) as data:
                    split_glosses = data['features'][:]
            else:
                with torch.no_grad():
                    # encode first english gloss for all nodes in split (valid/test) with CLIP
                    split_glosses = numpy.empty( shape=(len(gls), 640) )
                    iidx = 0
                    for node_idx in tqdm.tqdm(gls):
                        MAX_CONTEXT_LEN = 77
                        # sentences with length greater than 77 break the model
                        # this is a hyperparameter from pretraining CLIP and apparently cannot be changed unless retraining the model
                        sent_len = [len( spacy_tokenizer(gls[node_idx][0]) )]
                        tok_glosses = [spacy_tokenizer(gls[node_idx][0])[:MAX_CONTEXT_LEN]]
                        tok_glosses = [g.text for g in tok_glosses]

                        # CLIP's maxixum subword context length (for textual inputs) is 77
                        # since there is no easy way to find out what's the length of an input after being tokenized,
                        # we just try until we find the right length on a per-case basis
                        while True:
                            try:
                                tok_glosses = [spacy_tokenizer(g)[:MAX_CONTEXT_LEN] for g in tok_glosses]
                                tok_glosses = [g.text for g in tok_glosses]
                                text = clip.tokenize( tok_glosses ).to(device)
                                break
                            except RuntimeError:
                                MAX_CONTEXT_LEN -= 5

                        text_features = model.encode_text(text)
                        split_glosses[iidx, :] = text_features.cpu()
                        iidx += 1
                    numpy.savez_compressed(fname_split_gloss_feats, features=split_glosses)

            # we regenerate the gloss bnids to include a single bnid per gloss (since we only use the first english gloss)
            gls_bnids = [bnidx for bnidx, gs in gls.items()]

        else:
            # load/compute split (valid/test) image features
            fname_split_gloss_feats = os.path.join(path_to_vs, 'dataset', "visualsem-gloss-features.CLIP-%s.npz"%(MODEL_NAME))
            # load features if they have already been computed
            if os.path.isfile( fname_split_gloss_feats ):
                with numpy.load(fname_split_gloss_feats) as data:
                    split_glosses = data['features'][:]
            else:
                with torch.no_grad():
                    # encode all glosses for split with CLIP
                    split_glosses = numpy.empty( shape=(n_english_glosses, 640) )
                    iidx = 0
                    for node_idx in tqdm.tqdm(gls):
                        MAX_CONTEXT_LEN = 77
                        # sentences with length greater than 77 break the model
                        # this is a hyperparameter from pretraining CLIP and apparently cannot be changed unless retraining the model
                        sent_lens = [len( spacy_tokenizer(g) ) for g in gls[node_idx]]
                        tok_glosses = [spacy_tokenizer(g)[:MAX_CONTEXT_LEN] for g in gls[node_idx]]
                        tok_glosses = [g.text for g in tok_glosses]

                        # CLIP's maxixum subword context length (for textual inputs) is 77
                        # since there is no easy way to find out what's the length of an input after being tokenized,
                        # we just try until we find the right length on a per-case basis
                        while True:
                            try:
                                tok_glosses = [spacy_tokenizer(g)[:MAX_CONTEXT_LEN] for g in tok_glosses]
                                tok_glosses = [g.text for g in tok_glosses]
                                text = clip.tokenize( tok_glosses ).to(device)
                                break
                            except RuntimeError:
                                MAX_CONTEXT_LEN -= 5

                        text_features = model.encode_text(text)
                        split_glosses[iidx:iidx+len(tok_glosses), :] = text_features.cpu()
                        iidx += len(tok_glosses)
                    numpy.savez_compressed(fname_split_gloss_feats, features=split_glosses)

        split_glosses = torch.tensor(split_glosses)
        split_glosses = split_glosses.to(torch.float16)
        # normalize
        split_glosses = split_glosses / split_glosses.norm(dim=-1, keepdim=True)
        split_glosses = split_glosses.cuda()
        print("english training glosses: ", split_glosses.shape)
        print("one gloss per node?: ", one_gloss_per_node)

        # cosine similarity as logits
        logit_scale = torch.nn.Parameter(torch.ones([]) * numpy.log(1 / 0.07))
        logit_scale = logit_scale.exp()
        logit_scale = logit_scale.cuda()
        #logit_scale = logit_scale.to(torch.float16)

        #for split_name in ['valid', 'test']:
        for split_name in split_names:
            print("Processing %s ..."%split_name)
            fname_out = os.path.join(path_to_vs, 'dataset', "visualsem-nodes.images-to-glosses-scores.CLIP-%s.%s"%(MODEL_NAME, split_name))
            if one_gloss_per_node:
                fname_out += ".one-gloss-per-node.txt"
            else:
                fname_out += ".txt"

            if split_name == 'valid':
                ims = valid_ims
            elif split_name == 'test':
                ims = test_ims
            else:
                raise Exception("Split name not allowed: '%s'. Choose from: 'valid', 'test'."%split_name)

            with open(fname_out, 'w') as fh_out:
                # iterate node idxs for queries
                for node_idx in tqdm.tqdm(ims, total=len(ims)):

                    # retrieve targets gloss bnids for node idx
                    targets = numpy.array([bnid == node_idx for bnid in gls_bnids])
                    targets = targets.nonzero()[0]

                    # retrieve glosses for image (node)
                    for im in ims[ node_idx ]:
                        image = preprocess(Image.open(im)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_features = model.encode_image(image)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                            logits_per_query  = logit_scale * image_features @ split_glosses.t()
                            logits_per_images = logit_scale * split_glosses @ image_features.t()
                            g_probs = logits_per_query.softmax(dim=-1)
                            g_ranks = torch.argsort(g_probs, descending=True)
                            g_ranks = g_ranks.squeeze()
                            pred_prob       = g_probs[ :, g_ranks[0] ].item()
                            pred_gloss_bnid = gls_bnids[ g_ranks[0] ]

                            pred_rank = []
                            for target in targets:
                                pred_rank.append( numpy.where( target == g_ranks.cpu().numpy() )[0] )
                            pred_rank = numpy.vstack(pred_rank)

                            fh_out.write("%i\t%s\t%.2f\n"%(pred_rank.min().item(), pred_gloss_bnid, pred_prob))


if __name__=="__main__":
    path_to_vs = os.path.dirname(os.path.realpath(__file__))

    p = argparse.ArgumentParser()
    #p.add_argument('--gpu_idx', type=int, required=False, default=0,
    #        help="""GPU card idx to use, starting from 0. We split input examples deterministically
    #        using the GPU idx so that multiple jobs can be started simultaneously on multiple GPUs.""")
    #p.add_argument('--image_to_gloss', action='store_true',
    #        help="""Whether to run retrieval using images as queries against a node representation constituted of glosses.
    #        This works first by building a database of validation/test English glosses associated to each node,
    #        and then using images from the respective validation/test sets as queries.""")
    #p.add_argument('--image_to_image', action='store_true',
    #        help="""Whether to run retrieval using images as queries against a node representation constituted of images.
    #        This works first by building a database of training images associated to each node, and then using images from
    #        the validation and test sets as queries.""")
    p.add_argument('--one_gloss_per_node', action='store_true',
            help="""When running retrieval, wether to represent each node in the index with a single english gloss,
            or with all iavailable glosses for the node.""")
    p.add_argument('--split_name', type=str, choices=['valid', 'test'], nargs="+", default=['valid', 'test'],
            help="""Image splits to perform retrieval for.""")
    args = p.parse_args()

    #if args.image_to_image:
    #    raise NotImplemented()

    #threshold = 0.5
    #ngpus = torch.cuda.device_count()
    #assert(ngpus >= 1 and ngpus >= args.gpu_idx+1), "No GPU card found / problem with requested GPU idx %i!"%args.gpu_idx
    assert( torch.cuda.is_available() ), "At least one GPU card is required."

    #print("Creating output file images_filtered.json using threshold %.2f ..."%threshold)
    #print("%i GPU card(s) available, using GPU idx %i."%(ngpus, args.gpu_idx))

    train_ims, valid_ims, test_ims = load_visualsem_image_splits(0, 1)
    gls = load_visualsem_english_glosses(0, 1)
    #train_ims, valid_ims, test_ims = load_visualsem_image_splits(args.gpu_idx, ngpus)
    #gls = load_visualsem_english_glosses(args.gpu_idx, ngpus)
    gls_bnids = [[bnidx]*len(gs) for bnidx, gs in gls.items()]
    # function to flatten nested lists of depth 2
    flatten = lambda t: [item for sublist in t for item in sublist]
    gls_bnids = flatten(gls_bnids)

    assert( len(train_ims) == len(gls) ), "Nodes in glosses and images dictionary do not match. Lengths: %i (ims), %i (gls)"%(
            len(train_ims), len(gls))

    assert( len([g for k,gs in gls.items() for g in gs]) == len(gls_bnids) ), "Glosses and gloss BNids do not match: %i, %i"%(
            len([g for k,gs in gls.items() for g in gs]), len(gls_bnids))

    print("Number of nodes in train split: %i"%(len(train_ims)))
    print("Number of nodes in valid split: %i"%(len(valid_ims)))
    print("Number of nodes in test split: %i"%(len(test_ims)))

    #print("Number of nodes in train split (GPU idx %i): %i"%(args.gpu_idx, len(train_ims)))
    #print("Number of nodes in valid split (GPU idx %i): %i"%(args.gpu_idx, len(valid_ims)))
    #print("Number of nodes in test split (GPU idx %i): %i"%(args.gpu_idx, len(test_ims)))

    # load model on GPU requested
    #device = 'cuda:%i'%args.gpu_idx
    device = 'cuda'
    model, preprocess = load_clip_model(device)

    # images to glosses (retrieve nodes via glosses given image queries)
    #if args.image_to_gloss:
    process_image_to_gloss(model, device, preprocess, gls, gls_bnids, valid_ims, test_ims, args.one_gloss_per_node, args.split_names)


