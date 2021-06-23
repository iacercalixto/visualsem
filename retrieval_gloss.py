import argparse
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
from utils import grouper, load_sentences, load_bnids, load_visualsem_bnids


def retrieve_nodes_given_sentences(out_fname, batch_size, all_input_sentences, glosses_bnids, glosses_feats, topk):
    """
        out_fname(str):                     Output file to write retrieved node ids to.
        batch_size(int):                    Batch size for Sentence BERT.
        all_input_sentences(list[str]):     All input sentences loaded from `input_file`.
        glosses_bnids(list[str]):           All gloss BNids loaded from `args.glosses_bnids`. Aligned with `glosses_feats`.
        glosses_feats(numpy.array):         Numpy array with VisualSem gloss features computed with Sentence BERT.
        topk(int):                          Number of nodes to retrieve for each input sentence.
    """
    if os.path.isfile(out_fname):
        raise Exception("File already exists: '%s'. Please remove it manually to avoid tampering."%out_fname)

    n_examples = len(all_input_sentences)
    print("Number of input examples to extract BNIDs for: ", n_examples)
    model_name = "paraphrase-multilingual-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    with open(out_fname, 'w', encoding='utf8') as fh_out:
        ranks_predicted = []
        for idxs_ in grouper(batch_size, range(n_examples)):
            idxs = []
            queries = []
            for i in idxs_:
                if not i is None:
                    idxs.append(i)
                    queries.append( all_input_sentences[i] )

            queries_embs = model.encode(queries, convert_to_tensor=True)
            if torch.cuda.is_available():
                queries_embs = queries_embs.cuda()
            scores = util.pytorch_cos_sim(queries_embs, glosses_feats)
            scores = scores.cpu().numpy()

            ranks = numpy.argsort(scores) # sort scores by cosine similarity (low to high)
            ranks = ranks[:,::-1] # sort by cosine similarity (high to low)
            for rank_idx in range(len(idxs[:ranks.shape[0]])):
                bnids_predicted = []
                for rank_predicted in range(topk*10):
                    bnid_pred = glosses_bnids[ ranks[rank_idx,rank_predicted] ]
                    bnid_pred_score = scores[rank_idx, ranks[rank_idx, rank_predicted]]
                    if not bnid_pred in bnids_predicted:
                        bnids_predicted.append((bnid_pred,bnid_pred_score))
                    if len(bnids_predicted)>=topk:
                        break

                # write top-k predicted BNids
                for iii, (bnid, score) in enumerate(bnids_predicted[:topk]):
                    fh_out.write(bnid+"\t"+"%.4f"%score)
                    if iii < topk-1:
                        fh_out.write("\t")
                    else: # iii == topk-1
                        fh_out.write("\n")


def encode_query(out_fname, batch_size, all_sentences):
    """
        out_fname(str):                     Output file to write SBERT features for query.
        batch_size(int):                    Batch size for Sentence BERT.
        all_sentences(list[str]):           Sentences to be used for retrieval.
    """
    n_lines = len(all_sentences)
    model_name = "paraphrase-multilingual-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    shape_features = (n_lines, 768)
    with h5py.File(out_fname, 'w') as fh_out:
        fh_out.create_dataset("features", shape_features, dtype='float32', chunks=(1,768), maxshape=(None, 768), compression="gzip")

        for from_idx in tqdm.trange(0,n_lines,batch_size):
            to_idx = from_idx+batch_size if from_idx+batch_size <= n_lines else n_lines
            batch_sentences = all_sentences[ from_idx: to_idx ]
            emb_sentences = model.encode(batch_sentences, convert_to_tensor=True)
            #test_queries(emb_sentences, all_sentences, model)
            fh_out["features"][from_idx:to_idx] = emb_sentences.cpu().numpy()


if __name__=="__main__":
    visualsem_path             = os.path.dirname(os.path.realpath(__file__))
    visualsem_nodes_path       = "%s/dataset/nodes.v2.json"%visualsem_path
    visualsem_images_path      = "%s/dataset/images/"%visualsem_path
    glosses_sentence_bert_path = "%s/dataset/gloss_files/glosses.en.txt.sentencebert.h5"%visualsem_path
    glosses_bnids_path         = "%s/dataset/gloss_files/glosses.en.txt.bnids"%visualsem_path
    os.makedirs("%s/dataset/gloss_files/"%visualsem_path, exist_ok=True)

    p = argparse.ArgumentParser()
    p.add_argument('--input_files', type=str, nargs="+", default=["example_data/queries.txt"],
            help="""Input file(s) to use for retrieval. Each line in each file should contain a detokenized sentence.""")
    p.add_argument('--topk', type=int, default=1, help="Retrieve topk nodes for each input sentence.")
    p.add_argument('--batch_size', type=int, default=1000)
    p.add_argument('--visualsem_path', type=str, default=visualsem_path,
            help="Path to directory containing VisualSem knowledge graph.")
    p.add_argument('--visualsem_nodes_path', type=str, default=visualsem_nodes_path,
            help="Path to file containing VisualSem nodes.")
    p.add_argument('--visualsem_images_path', type=str, default=visualsem_images_path,
            help="Path to directory containing VisualSem images.")
    p.add_argument('--glosses_sentence_bert_path', type=str, default=glosses_sentence_bert_path,
            help="""HDF5 file containing glosses index computed with Sentence BERT (computed with `extract_glosses_visualsem.py`).""")
    p.add_argument('--glosses_bnids_path', type=str, default=glosses_bnids_path,
            help="""Text file containing glosses BabelNet ids, one per line (computed with `extract_glosses_visualsem.py`).""")
    p.add_argument('--input_valid', action='store_true',
            help="""Perform retrieval for the glosses in the validation set. (See paper for reference)""")
    p.add_argument('--input_test', action='store_true',
            help="""Perform retrieval for the glosses in the test set. (See paper for reference)""")
    args = p.parse_args()

    # load all nodes in VisualSem
    all_bnids = load_visualsem_bnids(args.visualsem_nodes_path, args.visualsem_images_path)
    gloss_bnids = load_bnids( args.glosses_bnids_path )
    gloss_bnids = numpy.array(gloss_bnids, dtype='object')

    with h5py.File(args.glosses_sentence_bert_path, 'r') as fh_glosses:
        glosses_feats  = fh_glosses["features"][:]
        glosses_feats  = torch.tensor(glosses_feats)
        if torch.cuda.is_available():
            glosses_feats = glosses_feats.cuda()

        # load train/valid/test gloss splits
        glosses_splits = fh_glosses["split_idxs"][:]
        train_idxs = (glosses_splits==0).nonzero()[0]
        valid_idxs = (glosses_splits==1).nonzero()[0]
        test_idxs  = (glosses_splits==2).nonzero()[0]

        # load gloss language splits
        language_splits = fh_glosses["language_idxs"][:]

        for input_file in args.input_files:
            print("Processing input file: %s ..."%input_file)
            sbert_out_fname = input_file+".sentencebert.h5"
            if os.path.isfile( sbert_out_fname ):
                raise Exception("File already exists: '%s'. Please remove it manually to avoid tampering."%sbert_out_fname)

            input_sentences = load_sentences( input_file )
            encode_query(sbert_out_fname, args.batch_size, input_sentences)
            out_fname = input_file+".bnids"
            retrieve_nodes_given_sentences(out_fname, args.batch_size, input_sentences, gloss_bnids, glosses_feats, args.topk)
            # remove temporary SBERT index created for input file(s)
            os.remove( sbert_out_fname )

            print("Retrieved glosses: %s"%out_fname)

