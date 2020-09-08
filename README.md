# VisualSem Knowledge Graph

VisualSem is a knowledge graph designed and curated to support research in vision and language.
It is built using [BabelNet v4.0](https://babelnet.org) and [ImageNet](http://www.image-net.org) as a starting point and it contains over 101k nodes, 1.9M tuples, and 1.5M glosses and 1.5M images associated to nodes. It is described in detail in [our resource paper](https://arxiv.org/abs/2008.09150).

In a nutshell, VisualSem includes:

- 101,244 nodes which are linked to BabelNet ids, and therefore linkable to Wikipedia article ids, WordNet ids, etc (through BabelNet).
- 13 _visually relevant_ relation types: _is-a_, _has-part_, _related-to_, _used-for_, _used-by_, _subject-of_, _receives-action_, _made-of_, _has-property_, _gloss-related_, _synonym_, _part-of_, and _located-at_.
- 1.9M tuples, where each tuple consists of a pair of nodes connected by a relation type.
- 1.5M glosses linked to nodes which are available in up to 14 different languages.
- 1.5M images associated to nodes.


## Downloading VisualSem

If you wish to download VisualSem, please [fill in this form](https://forms.gle/dPPxMfY9QKAuCo2L6) with your full name, the name of your institution and a valid institutional e-mail address and we will send you further instructions asap. After you download the data (`nodes.json`, `tuples.json`, `glosses.tgz` and `images.tgz`), make sure all these files are available in `./dataset`.

    mv nodes.json tuples.json glosses.tgz images.tgz ./dataset/
    cd dataset
    tar zxvf glosses.tgz
    tar zxvf images.tgz

VisualSem is publicly and fully available for researchers and is released under [BabelNet's non-commercial license](https://babelnet.org/license). We are not supported/endorsed by the BabelNet project in anyway. The only reason VisualSem is released with the same license as BabelNet is because it uses (among other tools) the BabelNet API in its construction and therefore we need to comply to the original license (see [our paper](https://arxiv.org/abs/2008.09150) and [BabelNet's license](https://babelnet.org/license) for details).

## Requirements

Use python 3 (we use python 3.7) and install the required packages.

    pip install -r requirements.txt

## Retrieval

We release a multi-modal retrieval framework that allows one to retrieve nodes from the KG given sentences or images.


### Sentence retrieval

We use [Sentence BERT](https://github.com/UKPLab/sentence-transformers) (SBERT) as the multilingual encoder in our sentence retrieval model. We encode all glosses in VisualSem using SBERT, and also the query. Retrieval is implemented as a simple k-NN algorithm that computes a dot-product between the query vector representing the input sentence and the nodes' gloss matrix. We directly retrieve the top-k unique nodes associated to the most relevant glosses as the results.

Assuming the file `/path/to/queries.txt` contains one English sentence per line consisting of multiple queries,  by running `retrieval_gloss.py` as below you will generate `/path/to/queries.txt.bnids` with the retrieved nodes. The generated file contains the retrieved nodes (i.e. BNid) followed by their score (i.e. cosine similarity with the query). You can retrieve nodes from VisualSem for each sentential query by running:

    python retrieval_gloss.py --input_file /path/to/queries.txt

You can also directly run the script without any flags, in which case it uses example data under `example_data/queries.txt`.

    python retrieval_gloss.py

If you want to retrieve using glosses in other languages, you can do as below (e.g. using German glosses).

    python retrieval_gloss.py
        --input_files example_data/queries.txt
        --glosses_sentence_bert_path dataset/gloss_files/glosses.de.txt.sentencebert.h5
        --glosses_bnids_path dataset/gloss_files/glosses.de.txt.bnids 

If you want to retrieve using glosses in multiple languages, you can first combine glosses together into a single index and retrieve as below.

    # use flag --help to see what each option entails.
    python combine_sentencebert_glosses.py --strategy {all, top8, all_but_swedish_and_farsi}

    python retrieval_gloss.py
        --input_files example_data/queries.txt
        --glosses_sentence_bert_path dataset/gloss_files/glosses.combined-top8.h5
        --glosses_bnids_path dataset/gloss_files/glosses.combined-top8.bnids 

The above command will build an index using glosses for the 8 best performing languages (according to experiments in our paper) instead of the 14 languages supported. This gloss matrix is then ranked according to gloss similarity to each sentential query in `queries.txt`, and the associated nodes are retrieved. Among other options, you can set the number of nodes to retrieve for each sentence (`--topk` parameter). 

### Image retrieval

*Coming soon.*


## Generating VisualSem from scratch

Please refer to the dataset creation [README.md](dataset_creation/README.md) for instructions on how to generate VisualSem from scratch.

### Enabling sentence retrieval with your locally generated VisualSem

If you have generated VisualSem from scratch, you will need to extract glosses again for the current node set in your version. To do that, simply run:

    python extract_glosses_visualsem.py --extract_glosses --extract_glosses_languages

In order to have sentence retrieval work against your locally generated VisualSem, you need to create `*.sentencebert.h5` files for each set of glosses in each language you support. To do that, simply run:

    python process_glosses_with_sentencebert.py

### Enabling image retrieval with your locally generated VisualSem

*Coming soon.*


## Example code

For examples on how to include VisualSem in your code base, please run:

    # iterate nodes and print all information available for each node (around 101k)
    python visualsem_dataset_nodes.py

    # iterate each tuple in the dataset (around 1.9M)
    python visualsem_dataset_tuples.py

## Citing our work

If you use VisualSem, please consider citing our paper.

    @article{alberts2020visualsem,
      title={VisualSem: a high-quality knowledge graph for vision and language},
      author={Alberts, Houda and Huang, Teresa and Deshpande, Yash and Liu, Yibo and Cho, Kyunghyun and Vania, Clara and Calixto, Iacer},
      journal={arXiv preprint arXiv:2008.09150},
      year={2020}
    }

If you use BabelNet, please refer to [BabelNet publications](https://babelnet.org/papers) on how to properly credit their work.


## License

VisualSem is publicly available for researchers and is released under [BabelNet's non-commercial license](https://babelnet.org/license).


[babelnet-license]: https://babelnet.org/full-license
[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
