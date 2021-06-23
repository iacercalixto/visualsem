import argparse
import torch
import sys
import os
import json
from collections import defaultdict
import h5py
from sentence_transformers import SentenceTransformer, util
import numpy
import pandas
import tqdm
from itertools import zip_longest
from utils import grouper, load_sentences, load_bnids, load_visualsem_bnids
import torchmetrics


def retrieve_nodes_given_sentences(out_fname, batch_size, mapping_bnids_idxs_to_gloss_idxs, glosses_feats, query_bnids_idxs_for_each_gloss, query_feats, topk, bnids_idxs_for_each_gloss, gloss_bnids, query_languages):
    """
        out_fname(str):                     Output file to write retrieved node ids to.
        batch_size(int):                    Batch size for Sentence BERT.
        glosses_feats(numpy.array):         Numpy array with VisualSem gloss features (to be used in search) computed with Sentence BERT.
        query_bnids_idxs_for_each_gloss(numpy.array(int)): Gloss BNID ids for each gloss in query.
        query_feats(numpy.array):           Numpy array with VisualSem gloss features (to be used as queries) computed with Sentence BERT.
        topk(int):                          Number of nodes to retrieve for each input sentence.
        bnids_idxs_for_each_gloss(numpy.array(int)): Gloss BNID ids for each gloss in searchable nodes, aligned to `glosses_feats`.
    """
    if os.path.isfile(out_fname):
        raise Exception("File already exists: '%s'. Please remove it manually to avoid tampering."%out_fname)

    n_examples = query_feats.shape[0]
    print("Number of input/query examples: ", n_examples)

    lang_codes = sorted([i.item() for i in numpy.unique(query_languages)])
    print("Number of languages: %i"%(len(lang_codes)))

    # shape: [n_queries, 1]
    # for each query, we store the rank we predicted the correct bnid (ranges from 0/best to number of nodes/worst)
    true_predicted_ranks = numpy.empty(shape=(len(query_bnids_idxs_for_each_gloss), 1))
    print("true_predicted_ranks.shape: ", true_predicted_ranks.shape)

    def first_nonzero_idxs_2dtensor(t):
        """ Given a 2D tensor, returns the first non-zero index in each row.
            If the input tensor `t` has shape `[n,m]`, the resulting tensor has shape `[n]`.
        """
        idx = torch.arange(t.shape[1], 0, -1)
        tmp2= t * idx.cuda()
        indices = torch.argmax(tmp2, 1, keepdim=True)
        return indices

    all_idxs = []
    with open(out_fname, 'w', encoding='utf8') as fh_out:
        ranks_predicted = []
        for idxs_ in tqdm.tqdm(grouper(batch_size, range(n_examples)), total=(n_examples//batch_size)):
            idxs = []
            for i in idxs_:
                if not i is None:
                    idxs.append(i)
            all_idxs.extend( idxs )

            # run search on CPU to avoid out-of-memory issues
            queries_embs = query_feats[ idxs ]
            queries_embs = queries_embs.cpu()
            glosses_feats = glosses_feats.cpu()
            scores = util.pytorch_cos_sim(queries_embs, glosses_feats)
            ranks = torch.argsort(scores, descending=True) # sort by cosine similarity (high to low)
            scores = scores.cuda()
            ranks = ranks.cuda()

            # shape: [n_queries, n_glosses]
            bnids_idxs_for_each_gloss_pred = bnids_idxs_for_each_gloss[ ranks ]
            # query idxs include all queries for valid/test set
            # first slice only the queries that apply to the current minibatch
            batch_query_bnids_idxs_for_each_gloss = query_bnids_idxs_for_each_gloss[ idxs ]

            # shape: [len(idxs), n_nodes]
            bnids_idxs_for_each_gloss_ranks_onehot = torch.where(
                    bnids_idxs_for_each_gloss_pred == batch_query_bnids_idxs_for_each_gloss.unsqueeze(1),
                    torch.tensor(1, device=torch.device('cuda')),
                    torch.tensor(0, device=torch.device('cuda'))
            )

            # shape: [len(idxs), 1]
            bnids_idxs_for_each_gloss_ranks_ = first_nonzero_idxs_2dtensor( bnids_idxs_for_each_gloss_ranks_onehot )
            true_predicted_ranks[ idxs, : ] = bnids_idxs_for_each_gloss_ranks_.cpu().numpy()

            # write retrieval results to output file
            for rank_idx in range(len(idxs[:ranks.shape[0]])):
                bnids_predicted = []
                for rank_predicted in range(topk*10):
                    bnid_pred = gloss_bnids[ ranks[rank_idx,rank_predicted] ]
                    bnid_pred_score = scores[rank_idx, ranks[rank_idx, rank_predicted]].item()
                    if not bnid_pred in bnids_predicted:
                        bnids_predicted.append((bnid_pred,bnid_pred_score))

                    if len(bnids_predicted)>=topk:
                        break

                # write top-k predicted BNids, their scores and ranks
                for iii, (bnid, score) in enumerate(bnids_predicted[:topk]):
                    fh_out.write(bnid+"\t"+"%.4f"%score)
                    if iii < topk-1:
                        fh_out.write("\t")
                    else: # iii == topk-1
                        fh_out.write("\n")

    print("Processed %i queries"%len(all_idxs))
    #print(true_predicted_ranks.shape)
    print("Mean ranks (std): ", true_predicted_ranks.mean(), "(", true_predicted_ranks.std(), ")")
    for k in [1,2,3,5,10]:
        p_at_k = (true_predicted_ranks[:] < k).sum() / true_predicted_ranks.shape[0]
        print("Hits@%i: %.4f"%(k, p_at_k))

    for lidx in lang_codes:
        print("Language: %i"%lidx)
        print("... Mean ranks (std): ", true_predicted_ranks[query_languages==lidx].mean(), "(", true_predicted_ranks[query_languages==lidx].std(), ")")
        for k in [1,2,3,5,10]:
            p_at_k = (true_predicted_ranks[query_languages==lidx] < k).sum() / true_predicted_ranks[query_languages==lidx].shape[0]
            print("... Hits@%i: %.4f"%(k, p_at_k))


if __name__=="__main__":
    visualsem_path             = os.path.dirname(os.path.realpath(__file__))
    visualsem_nodes_path       = "%s/dataset/nodes.v2.json"%visualsem_path
    visualsem_images_path      = "%s/dataset/images/"%visualsem_path
    glosses_sentence_bert_path = "%s/dataset/gloss_files/glosses.en.txt.sentencebert.h5"%visualsem_path
    glosses_bnids_path         = "%s/dataset/gloss_files/glosses.en.txt.bnids"%visualsem_path
    os.makedirs("%s/dataset/gloss_files/"%visualsem_path, exist_ok=True)

    p = argparse.ArgumentParser()
    g = p.add_argument_group()
    g.add_argument('--input_valid', action='store_true',
            help="""Perform retrieval for the glosses in the validation set. (See paper for reference)""")
    g.add_argument('--input_test', action='store_true',
            help="""Perform retrieval for the glosses in the test set. (See paper for reference)""")
    p.add_argument('--topk', type=int, default=1, help="Retrieve topk nodes for each input sentence.")
    p.add_argument('--batch_size', type=int, default=128)
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
    args = p.parse_args()

    assert(torch.cuda.is_available()), "Must have at least one GPU available."

    if not args.input_valid and not args.input_test:
        p.print_usage()
        sys.exit(1)

    print(args)
    # load all nodes (bnids) in VisualSem
    all_bnids = load_visualsem_bnids(args.visualsem_nodes_path, args.visualsem_images_path)
    # load all glosses (bnids) in VisualSem
    gloss_bnids = load_bnids( args.glosses_bnids_path )

    mapping_bnids_strs_to_bnids_idxs = {}
    mapping_gloss_idxs_to_bnids_idxs = {}
    mapping_bnids_idxs_to_gloss_idxs = defaultdict(list)
    # the can be multiple glosses per node/bnid
    # create mappings from gloss idxs to node idxs (and bnids) and vice-versa
    for idx, bnid in enumerate(all_bnids):
        mapping_bnids_strs_to_bnids_idxs[ bnid ] = idx

    for idx, bnid_str in enumerate(gloss_bnids):
        mapping_gloss_idxs_to_bnids_idxs[ idx ] = mapping_bnids_strs_to_bnids_idxs[ bnid_str ]
        mapping_bnids_idxs_to_gloss_idxs[ mapping_bnids_strs_to_bnids_idxs[bnid_str] ].append( idx )

    # vector with all gloss idxs (~1M)
    gloss_idxs = torch.tensor(list(mapping_gloss_idxs_to_bnids_idxs.keys()), dtype=torch.int32)
    # numpy array the same size as `gloss_idxs` but where instead of the gloss idx we directly have the corresponding node idx (bnid idx)
    bnids_idxs_for_each_gloss = torch.tensor(list(mapping_gloss_idxs_to_bnids_idxs.values()))
    if torch.cuda.is_available():
        gloss_idxs = gloss_idxs.cuda()
        bnids_idxs_for_each_gloss = bnids_idxs_for_each_gloss.cuda()

    with h5py.File(args.glosses_sentence_bert_path, 'r') as fh_glosses:
        # load sentence bert features for each gloss
        glosses_feats  = fh_glosses["features"][:]
        glosses_feats  = torch.tensor(glosses_feats)
        if torch.cuda.is_available():
            glosses_feats = glosses_feats.cuda()

        # load train/valid/test gloss splits
        glosses_splits = fh_glosses["split_idxs"][:]
        train_idxs = (glosses_splits==0).nonzero()[0]
        train_feats = glosses_feats[train_idxs]
        train_bnids_idxs_for_each_gloss = bnids_idxs_for_each_gloss[train_idxs]

        # load gloss language splits
        language_splits = fh_glosses["language_idxs"][:]

        if args.input_valid:
            print("Processing validation set glosses ...")
            valid_idxs = (glosses_splits==1).nonzero()[0]
            valid_feats = glosses_feats[valid_idxs]
            valid_languages = language_splits[valid_idxs]
            valid_bnids_idxs_for_each_gloss = bnids_idxs_for_each_gloss[ valid_idxs ]

            # file names, input/output
            input_file = "valid."+ args.glosses_bnids_path.rsplit("/", 1)[-1].replace(".h5", "")
            out_fname = os.path.join(args.visualsem_path, 'dataset', input_file+".bnids.retrieved_nodes")

            retrieve_nodes_given_sentences(out_fname, args.batch_size, mapping_bnids_idxs_to_gloss_idxs,
                    train_feats, valid_bnids_idxs_for_each_gloss, valid_feats, args.topk,
                    train_bnids_idxs_for_each_gloss, gloss_bnids, valid_languages)
            print("Retrieved glosses: %s"%out_fname)

        if args.input_test:
            print("Processing test set glosses ...")
            test_idxs  = (glosses_splits==2).nonzero()[0]
            test_feats = glosses_feats[test_idxs]
            test_languages = language_splits[test_idxs]
            test_bnids_idxs_for_each_gloss  = bnids_idxs_for_each_gloss[ test_idxs ]

            # file names, input/output
            input_file = "test."+ args.glosses_bnids_path.rsplit("/", 1)[-1].replace(".h5", "")
            out_fname = input_file+".bnids.retrieved_nodes"

            retrieve_nodes_given_sentences(out_fname, args.batch_size, mapping_bnids_idxs_to_gloss_idxs,
                    train_feats, test_bnids_idxs_for_each_gloss, test_feats, args.topk,
                    train_bnids_idxs_for_each_gloss, gloss_bnids, test_languages)

            print("Retrieved glosses: %s"%out_fname)

