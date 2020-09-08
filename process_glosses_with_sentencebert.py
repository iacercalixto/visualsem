import torch
import sys
import os
import json
from collections import defaultdict
import h5py
from sentence_transformers import SentenceTransformer, util
import numpy
import tqdm
from itertools import zip_longest
import argparse
from utils import grouper, load_sentences, load_bnids, load_visualsem_bnids


def evaluate_glosses(gloss_fname, bnid_fname, batch_size, all_gloss_sentences, all_gloss_bnids, all_bnids):
    """
        gloss_fname(str):                   Name of file containing glosses for a specific language.
        bnid_fname(str):                    Name of file containing BNids (e.g. unique indices) for each gloss in `gloss_fname`.
        batch_size(int):                    Batch size for Sentence BERT.
        all_sentences(list[str]):           All glosses loaded from `gloss_fname`.
        all_gloss_bnids(list[str]):         All gloss BNids loaded from `bnid_fname`.
        all_bnids(list[str]):               All BNids/nodes in Visualsem.
    """
    # evaluate on random subset of 2000 glosses
    valid_idxs = numpy.random.permutation( numpy.array( range(all_gloss_bnids.shape[0]) ) )[:2000]
    n_examples = len(valid_idxs)
    print("... Number of examples randomly selected for evaluation: ", n_examples)

    with h5py.File(gloss_fname+".sentencebert.h5", 'r') as fh_gloss:
        glosses_feats = fh_gloss["features"][:]
        glosses_feats = torch.tensor(glosses_feats)
        ranks_predicted = []
        for idxs_ in grouper(batch_size, valid_idxs):
            idxs = []
            for i in idxs_:
                if not i is None:
                    idxs.append(i)

            queries = glosses_feats[idxs]
            scores = util.pytorch_cos_sim(queries, glosses_feats)
            scores = scores.cpu().numpy()
            # query vs. itself is in main diagonal. fill it with large negative not to retrieve it.
            other_glosses_with_same_bnid_minibatch = []
            for x,y in zip(list(range(batch_size)), idxs):
                scores[x,y] = -10.0
                # retrieve ground-truth bnids for each example
                this_gloss_bnid = all_gloss_bnids[ y ]
                other_glosses_with_same_bnid = numpy.where(all_gloss_bnids==this_gloss_bnid)[0]
                other_glosses_with_same_bnid_minibatch.append( other_glosses_with_same_bnid )

            ranks = numpy.argsort(scores) # sort scores by cosine similarity (low to high)
            ranks = ranks[:,::-1] # sort by cosine similarity (high to low)
            for r,index,bnid in zip(range(ranks.shape[0]), idxs, other_glosses_with_same_bnid_minibatch):
                # uncomment the print statement below to debug/gather details on the predicted nodes.
                #print(
                #        "bnid: ", all_gloss_bnids[bnid[0]], ", bnid indices: ",  bnid,
                #        ", numpy.in1d(ranks[r], bnid_indices): ",
                #        numpy.in1d(ranks[r], bnid),
                #        numpy.where( numpy.in1d(ranks[r], bnid) )[0],
                #        "bnid rank 0: ", all_gloss_bnids[ranks[r,0]]
                #)
                rank_predicted = numpy.where( numpy.in1d(ranks[r], bnid) )[0][0]
                ranks_predicted.append( rank_predicted )

        ranks_predicted = numpy.array(ranks_predicted)
        #print("ranks_predicted: ", ranks_predicted)
        print("... Rank mean/std: ", ranks_predicted.mean().item(), ranks_predicted.std().item())
        for k in [1,3,5,10]:
            print("... Accuracy (hits@%i): %.2f%%"%(k, (ranks_predicted<=(k-1)).sum()*1.0/ranks_predicted.shape[0]*100))


def compute_glosses(gloss_fname, bnid_fname, batch_size, all_sentences, all_bnids):
    """
        gloss_fname(str):                   Name of file containing glosses for a specific language.
        bnid_fname(str):                    Name of file containing BNids (e.g. unique indices) for each gloss in `gloss_fname`.
        batch_size(int):                    Batch size for Sentence BERT.
        all_sentences(list[str]):           All glosses loaded from `gloss_fname`.
        all_bnids(list[str]):               All gloss BNids loaded from `bnid_fname`.
    """
    n_lines = len(all_sentences)
    model = SentenceTransformer('distiluse-base-multilingual-cased')
    shape_features = (n_lines, 512)
    with h5py.File(gloss_fname+".sentencebert.h5", 'w') as fh_out:
        fh_out.create_dataset("features", shape_features, dtype='float32', chunks=(1,512), maxshape=(None, 512), compression="gzip")

        for from_idx in tqdm.trange(0,n_lines,batch_size):
            to_idx = from_idx+batch_size if from_idx+batch_size <= n_lines else n_lines
            #print(from_idx, to_idx)
            batch_sentences = all_sentences[ from_idx: to_idx ]
            emb_sentences = model.encode(batch_sentences, convert_to_tensor=True)
            #test_queries(emb_sentences, all_sentences, model)
            fh_out["features"][from_idx:to_idx] = emb_sentences.cpu().numpy()


if __name__=="__main__":
    visualsem_path = "/misc/vlgscratch5/ChoGroup/icalixto/visualsem_krakatoa"
    visualsem_nodes_path = "%s/dataset/nodes.json"%visualsem_path
    visualsem_images_path = "%s/dataset/images/"%visualsem_path
    visualsem_gloss_languages = ['ar', 'de', 'en', 'es', 'fr', 'it', 'ko', 'nl', 'pt', 'ru', 'zh', 'fa', 'pl', 'sv']
    p = argparse.ArgumentParser()
    p.add_argument('--visualsem_path', type=str, default=visualsem_path,
            help="Path to directory containing VisualSem knowledge graph.")
    p.add_argument('--visualsem_nodes_path', type=str, default=visualsem_nodes_path,
            help="Path to file containing VisualSem nodes.")
    p.add_argument('--visualsem_images_path', type=str, default=visualsem_images_path,
            help="Path to directory containing VisualSem images.")
    p.add_argument('--gloss_languages', type=str, nargs="+", default=visualsem_gloss_languages,
            help="Process glosses in these languages. Requires glosses to have been extracted previously from BabelNet v4.0.")
    p.add_argument('--batch_size', type=int, default=1000)
    args = p.parse_args()
    os.makedirs('%s/dataset/gloss_files/'%args.visualsem_path, exist_ok=True)

    # load all nodes in VisualSem
    all_bnids = load_visualsem_bnids(args.visualsem_nodes_path, args.visualsem_images_path)
    gloss_fnames = ['%s/dataset/gloss_files/glosses.%s.txt'%(args.visualsem_path, l) for l in args.gloss_languages]
    #batch_size = 1000

    for gloss_fname in gloss_fnames:
        # load all glosses for the language
        all_gloss_sentences = load_sentences( gloss_fname )
        # load all BNids for each gloss
        bnid_fname = gloss_fname + ".bnids"
        all_gloss_bnids = load_bnids( bnid_fname )
        all_gloss_bnids = numpy.array(all_gloss_bnids, dtype='object')

        if not os.path.isfile(gloss_fname+".sentencebert.h5"):
            # if it still has not been created, create the gloss index using Sentence BERT.
            # https://github.com/UKPLab/sentence-transformers
            compute_glosses(gloss_fname, bnid_fname, args.batch_size, all_gloss_sentences, all_gloss_bnids)

        else:
            # randomly select a subset of the glosses for the language and evaluate node retrieval.
            evaluate_glosses(gloss_fname, bnid_fname, args.batch_size, all_gloss_sentences, all_gloss_bnids, all_bnids)

