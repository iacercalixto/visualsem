# VisualSem Knowledge Graph

VisualSem is a multilingual and multi-modal knowledge graph designed and curated to support research in vision and language.
It is built using different publicly available resources (e.g., [Wikipedia](https://www.wikipedia.org), [ImageNet](http://www.image-net.org), [BabelNet v4.0](https://babelnet.org)) and it contains around 90k nodes, 1.5M tuples, and 1.3M glosses and 930k images associated to nodes.

In a nutshell, VisualSem includes:

- 89,896 nodes which are linked to Wikipedia articles, WordNet ids, and BabelNet ids.
- 13 _visually relevant_ relation types: _is-a_, _has-part_, _related-to_, _used-for_, _used-by_, _subject-of_, _receives-action_, _made-of_, _has-property_, _gloss-related_, _synonym_, _part-of_, and _located-at_.
- 1.5M tuples, where each tuple consists of a pair of nodes connected by a relation type.
- 1.3M glosses linked to nodes which are available in up to 14 different languages.
- 930k images associated to nodes.


## Downloading VisualSem

VisualSem is publicly and fully available for researchers and is released under [BabelNet's non-commercial license](https://babelnet.org/license). We are not supported/endorsed by the BabelNet project in anyway. The only reason VisualSem is released with the same license as BabelNet is because it uses (among other tools) the BabelNet API in its construction and therefore we comply with the original license (see [BabelNet's license](https://babelnet.org/license) for details).

- [nodes.v2.json](https://surfdrive.surf.nl/files/index.php/s/06AFB1LsJV9yt5N) (83MB): All nodes in VisualSem.
- [tuples.v2.json](https://surfdrive.surf.nl/files/index.php/s/P37QRCWDJVRqcWG) (83MB): All tuples in VisualSem.
- [glosses.v2.tgz](https://surfdrive.surf.nl/files/index.php/s/gQLULr5ElOEiafx) (125MB): All 1.5M glosses in 14 different languages.
- [images.tgz](https://surfdrive.surf.nl/files/index.php/s/KXmZTm4hNaXoYfO) (31GB): All 1.5M images.

In addition to the dataset files, you can also download pre-extracted features (used in retrieval experiments).
- [glosses.sentencebert.v2.tgz](https://surfdrive.surf.nl/files/index.php/s/7PDiEKQapk4dhlW) (9.8GB): Sentence BERT features extracted for all glosses as well as gloss training/validation/test splits.
- [images_features_splits.tgz](https://surfdrive.surf.nl/files/index.php/s/nuzVxSfhSH91MSv) (82MB): Image training/validation/test splits.
- [visualsem-image-features.valid.CLIP-RN50x4.npz](https://surfdrive.surf.nl/files/index.php/s/SvWgg9RZNEaXHls) (31MB) and [visualsem-image-features.test.CLIP-RN50x4.npz](https://surfdrive.surf.nl/files/index.php/s/pRsiPCuDLpUxmmZ) (31MB): CLIP features for all images in validation/test splits.


After you download the data (`nodes.v2.json`, `tuples.v2.json`, `glosses.v2.tgz`, `images.tar`, `glosses.sentencebert.v2.tgz`, `images_features_splits.tgz`, `visualsem-image-features.valid.CLIP-RN50x4.npz`, `visualsem-image-features.test.CLIP-RN50x4.npz`), make sure all these files are available in `./dataset`. These files are password protected, please send us an email (calixto[dot]iacer[at]gmail[dot]com) requesting the password using your **institutional email address (e.g. university email)** and we will provide you with the password asap.


Untar the (compressed) tarballs as indicated below.

    mkdir ./dataset && cd ./dataset
    tar zxvf glosses.v2.tgz
    tar zxvf glosses.sentencebert.v2.tgz
    tar zxvf images_features_splits.tgz
    tar xvf images.tar


## Requirements

Use python 3 (we use python 3.7) and install the required packages.

    pip install -r requirements.txt

## Retrieval

We release a multi-modal retrieval framework that allows one to retrieve nodes from the KG given sentences or images.


### Sentence retrieval

We use [Sentence BERT](https://github.com/UKPLab/sentence-transformers) (SBERT) as the multilingual encoder in our sentence retrieval model. We encode all glosses in VisualSem using SBERT, and also the query. Retrieval is implemented with k-NN where we compute the dot-product between the query vector representing the input sentence and the nodes' gloss matrix. We directly retrieve the top-k unique nodes associated to the most relevant glosses as the results.

#### Reproduce paper results

To reproduce the sentence retrieval results in our paper (metric scores obtained on validation and test gloss splits), run the command below.

    python retrieval_gloss_paper.py

If your VisualSem files are in non-standard directories, run `python retrieval_gloss_paper.py --help` to see the arguments to use to provide their locations.

#### Retrieve nodes for an arbitrary sentence

Assuming the file `/path/to/queries.txt` contains one (detokenized) English sentence per line consisting of multiple queries,  by running `retrieval_gloss.py` as below you will generate `/path/to/queries.txt.bnids` with the retrieved nodes. The generated file contains the retrieved nodes (i.e. BNid) followed by their score (i.e. cosine similarity with the query). You can retrieve nodes from VisualSem for each query by running:

    python retrieval_gloss.py --input_file /path/to/queries.txt

You can also directly run the script without any flags, in which case it uses example sentence queries under `example_data/queries.txt`.

    python retrieval_gloss.py

If you want to retrieve using glosses in other languages, you can do as below (e.g. using German glosses).

    python retrieval_gloss.py
        --input_files example_data/queries.txt
        --glosses_sentence_bert_path dataset/gloss_files/glosses.de.txt.sentencebert.h5
        --glosses_bnids_path dataset/gloss_files/glosses.de.txt.bnids

If you want to retrieve using glosses in multiple languages, you can first combine glosses together into a single index and retrieve as below.

    # use flag --help to see what each option entails.
    python combine_sentencebert_glosses.py --strategy {all, top8}

    python retrieval_gloss.py
        --input_files example_data/queries.txt
        --glosses_sentence_bert_path dataset/gloss_files/glosses.combined-top8.h5
        --glosses_bnids_path dataset/gloss_files/glosses.combined-top8.bnids

The above command will build an index using glosses for the 8 best performing languages (according to experiments in our paper) instead of all the 14 supported languages. This gloss matrix is then ranked according to gloss similarity to each query in `queries.txt`, and the associated nodes are retrieved. Among other options, you can set the number of nodes to retrieve for each sentence (`--topk` parameter).

### Image retrieval

We use [Open AI's CLIP](https://github.com/openai/CLIP) as our image retrieval model. CLIP has a bi-encoder architecture with one text and one image encoder. We encode all English glosses in VisualSem using CLIP's text encoder, and we encode the image we are using to query the KG with CLIP's image encoder. Retrieval is again implemented as k-NN where we compute the dot-product between the query vector representing the input image and the nodes' gloss matrix. We directly retrieve the top-k unique nodes associated to the highest scoring glosses as the results.

#### Reproduce paper results

First, if you have not downloaded the validation and test image features extracted with CLIP ([visualsem-image-features.valid.CLIP-RN50x4.npz](https://surfdrive.surf.nl/files/index.php/s/SvWgg9RZNEaXHls) and [visualsem-image-features.test.CLIP-RN50x4.npz](https://surfdrive.surf.nl/files/index.php/s/pRsiPCuDLpUxmmZ)), run the script below.

    python encode_images_with_CLIP.py

To reproduce the image retrieval results in our paper (metric scores obtained on validation and test image splits), run the script below.

    python retrieval_image_paper.py

#### Retrieve nodes for an arbitrary image

Assuming the file `/path/to/queries.txt` contains the full path to one image file per line,  by running `retrieval_image.py` as below you will generate `/path/to/queries.txt.bnids` with the retrieved nodes. The generated file contains the retrieved nodes (i.e. BNid) followed by their score (i.e. cosine similarity with the query image). You can retrieve nodes from VisualSem for each image query by running:

    python retrieval_image.py --input_file /path/to/queries.txt

You can also directly run the script without any flags, in which case it uses example image file queries under `example_data/queries.txt`.

    python retrieval_image.py


## Generating VisualSem from scratch

Please refer to the dataset creation [README.md](dataset_creation/README.md) for instructions on how to generate VisualSem from scratch.

### Enabling sentence and image retrieval with your locally generated VisualSem

If you have generated VisualSem from scratch, you will need to extract glosses again for the current node set in your version. To do that, simply run:

    python extract_glosses_visualsem.py --extract_glosses --extract_glosses_languages

In order to have sentence and image retrieval work against your locally generated VisualSem, you need to create `*.sentencebert.h5` files for each set of glosses in each language you support. To do that, simply run:

    python process_glosses_with_sentencebert.py


## Example code

For examples on how to include VisualSem in your code base, please run:

    # iterate nodes and print all information available for each node (around 101k)
    python visualsem_dataset_nodes.py

    # iterate each tuple in the dataset (around 1.5M)
    python visualsem_dataset_tuples.py


## License

VisualSem is publicly available for research and is released under [BabelNet's non-commercial license](https://babelnet.org/license).


[babelnet-license]: https://babelnet.org/full-license
[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
