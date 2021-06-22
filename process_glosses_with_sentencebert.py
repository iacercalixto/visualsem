import torch
import sys
import os
import random
import json
from collections import defaultdict
import h5py
from sentence_transformers import SentenceTransformer, util
import numpy
import tqdm
from itertools import zip_longest
import argparse
from utils import grouper, load_sentences, load_bnids, load_visualsem_bnids


def compute_glosses(gloss_fname, bnid_fname, batch_size, all_sentences, all_bnids, lang_idx):
    """
        gloss_fname(str):                   Name of file containing glosses for a specific language.
        bnid_fname(str):                    Name of file containing BNids (e.g. unique indices) for each gloss in `gloss_fname`.
        batch_size(int):                    Batch size for Sentence BERT.
        all_sentences(list[str]):           All glosses loaded from `gloss_fname`.
        all_bnids(list[str]):               All gloss BNids loaded from `bnid_fname`.
        lang_idx(int):                      Language idx of the glosses (integer).
    """
    n_lines = len(all_sentences)

    # reserve 2,000 glosses for valid and 2,000 glosses for test set
    assert(n_lines > 10000), "Gloss file has less than 10000 lines: %s"%gloss_fname
    all_idxs = list(range(n_lines))
    random.shuffle(all_idxs)
    valid_idxs = all_idxs[:2000]
    test_idxs  = all_idxs[2000:4000]
    train_idxs = all_idxs[4000:]

    #model = SentenceTransformer('distiluse-base-multilingual-cased')
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    shape_features = (n_lines, 768)
    shape_splits   = (n_lines, 1)
    with h5py.File(gloss_fname+".sentencebert.h5", 'w') as fh_out:
        fh_out.create_dataset("features", shape_features, dtype='float32', chunks=(1,768), maxshape=(None, 768), compression="gzip")
        # train -> 0, valid -> 1, test-> 2
        fh_out.create_dataset("split_idxs", shape_splits, dtype='uint8', chunks=(1,1), maxshape=(None, 1), compression="gzip")
        # by running this script using the default languages, we have:
        # ar -> 0, de -> 1, en -> 2, es -> 3, fr -> 4, it -> 5, ko -> 6, nl -> 7, pt -> 8, ru -> 9, zh -> 10, fa -> 11, pl -> 12, sv -> 13
        fh_out.create_dataset("language_idxs", shape_splits, dtype='uint8', chunks=(1,1), maxshape=(None, 1), compression="gzip")

        for from_idx in tqdm.trange(0, n_lines, batch_size):
            to_idx = from_idx+batch_size if from_idx+batch_size <= n_lines else n_lines
            #print(from_idx, to_idx)
            batch_sentences = all_sentences[ from_idx: to_idx ]
            emb_sentences = model.encode(batch_sentences, convert_to_tensor=True)
            #test_queries(emb_sentences, all_sentences, model)

            split_idxs = []
            for curr_idx in range(from_idx, to_idx):
                if curr_idx in train_idxs:
                    curr_idx = 0
                elif curr_idx in valid_idxs:
                    curr_idx = 1
                elif curr_idx in test_idxs:
                    curr_idx = 2
                else:
                    raise Exception()
                split_idxs.append( curr_idx )

            fh_out["split_idxs"][from_idx:to_idx] = numpy.array(split_idxs, dtype='uint8')[:,None]
            fh_out["features"][from_idx:to_idx] = emb_sentences.cpu().numpy()
            fh_out["language_idxs"][from_idx:to_idx] = numpy.array([lang_idx]*(to_idx-from_idx))[:,None]


if __name__=="__main__":
    visualsem_path = os.path.dirname(os.path.realpath(__file__))
    visualsem_glosses_path = "%s/dataset/gloss_files/"%visualsem_path
    visualsem_nodes_path = "%s/dataset/nodes.v2.json"%visualsem_path
    visualsem_images_path = "%s/dataset/images/"%visualsem_path
    visualsem_gloss_languages = ['ar', 'de', 'en', 'es', 'fr', 'it', 'ko', 'nl', 'pt', 'ru', 'zh', 'fa', 'pl', 'sv']
    #visualsem_gloss_languages = ['en']
    p = argparse.ArgumentParser()
    p.add_argument('--visualsem_path', type=str, default=visualsem_path,
            help="Path to directory containing VisualSem knowledge graph.")
    p.add_argument('--visualsem_glosses_path', type=str, default=visualsem_glosses_path,
            help="Path to directory containing VisualSem glosses.")
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
    gloss_fnames = ['%s/glosses.%s.txt'%(args.visualsem_glosses_path, l) for l in args.gloss_languages]
    #batch_size = 1000

    for lang_idx, gloss_fname in enumerate(gloss_fnames):
        #lang = args.gloss_languages[lang_idx]
        #lang_idx = visualsem_gloss_languages.index(lang)

        # load all glosses for the language
        all_gloss_sentences = load_sentences( gloss_fname )
        # load all BNids for each gloss
        bnid_fname = gloss_fname + ".bnids"
        all_gloss_bnids = load_bnids( bnid_fname )
        all_gloss_bnids = numpy.array(all_gloss_bnids, dtype='object')

        assert(not os.path.isfile(gloss_fname+".sentencebert.h5")), \
                "File found: '%s'. Please remove it manually to avoid tampering."%gloss_fname+".sentencebert.h5"

        #if not os.path.isfile(gloss_fname+".sentencebert.h5"):
        # if it still has not been created, create the gloss index using Sentence BERT.
        # https://github.com/UKPLab/sentence-transformers
        compute_glosses(gloss_fname, bnid_fname, args.batch_size, all_gloss_sentences, all_gloss_bnids, lang_idx)

        #else:
        #    # randomly select a subset of the glosses for the language and evaluate node retrieval.
        #    evaluate_glosses(gloss_fname, bnid_fname, args.batch_size, all_gloss_sentences, all_gloss_bnids, all_bnids)

