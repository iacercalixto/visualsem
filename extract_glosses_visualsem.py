import sys
import json
import urllib.request
import os
import utils
import tqdm
import argparse
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor

# set the variable below to where you are serving your local BabelNet index.
BN_SERVER_URI="http://localhost:8080"


def get_glosses_from_synset(synset_id, key=None):
    """
        synset_id(str):                     BabelNet synset BNid.
        key(str):                           If not None, this must be the BabelNet key of the researcher.

        Returns a list of multilingual glosses for a given nodel.

        In order to call this function with key==None you must have an instance of BabelNet running on the local machine!
    """
    if key:
        req_link = "https://babelnet.io/v5/getSynset?id=" + synset_id + "&key=" + key
        content = urllib.request.urlopen(req_link).read().decode('utf8').replace("'", '"')
    else:
        req_link = BN_SERVER_URI + "/getSynset?id=" + synset_id
        content = urllib.request.urlopen(req_link).read().decode('utf8')

    content = json.loads(content)
    return content["glosses"]


def save_glosses(visualsem_nodes_json, out_file, available_languages):
    """
        visualsem_nodes_json(dict):         VisualSem nodes dictionary.
        out_file(str):                  File to write glosses to.
        available_languages(list[str]):     Glosses languages to include.

        For each VisualSem node id in `visualsem_nodes_json`, retrieve all its glosses
        and keep only those in `available_languages`. Write all glosses to output file `out_file`.
    """
    glosses_out = [ {'available_languages': [l.lower() for l in available_languages]} ]
    # iterate nodes' bn idxs
    for idx,synset_id in enumerate(tqdm.tqdm(visualsem_nodes_json.keys())):
        targets = get_glosses_from_synset( synset_id )
        languages = [target["language"] for target in targets]
        glosses   = [{target["language"]:target['gloss']} for target in targets]
        glosses_dict = {}
        for l, kv in zip(languages, glosses):
            assert(len(kv.items())==1)
            k,v = list(kv.keys())[0], list(kv.values())[0]
            assert(l==k)

            # only include glosses for languages in `available_languages` list
            if not l in available_languages:
                continue

            if not k.lower() in glosses_dict.keys():
                glosses_dict[k.lower()] = []
            glosses_dict[k.lower()].append( v )
        
        #print(synset_id, "glosses: ", glosses)
        #print("glosses_dict: ", glosses_dict)
        glosses_out.append( {synset_id : glosses_dict} )
    #print( "glosses_out: ", glosses_out )
    with open(out_file, 'w', encoding='utf8') as fh:
        json.dump(glosses_out, fh, ensure_ascii=False)


def load_glosses(in_file):
    """
        in_file(str):                       File containing VisualSem glosses.

        This file is the file created with `save_glosses`.
    """
    with open(in_file, 'r') as fh:
        x = json.load(fh)
    return x


def save_glosses_languages(visualsem_nodes_json, out_file):
    """
        visualsem_nodes_json(dict):         VisualSem nodes dictionary.
        out_file(str):                      File to write gloss languages to.

        For each VisualSem node id in `visualsem_nodes_json`, retrieve its glosses languages.
        Write all languages to output file `out_file`.
    """
    available_languages = {}
    # iterate nodes' bn idxs
    for idx,synset_id in enumerate(tqdm.tqdm(visualsem_nodes_json.keys())):
        targets = get_glosses_from_synset( synset_id )
        languages = [target["language"] for target in targets]
        #glosses = " ".join([target["gloss"] for target in targets])
        available_languages[synset_id] = languages
        #if (idx+1)%500==0:
        #     break
    with open(out_file, 'w', encoding='utf8') as fh:
        json.dump(available_languages, fh, ensure_ascii=False)


def load_glosses_languages(in_file):
    """
        in_file(str):                       File containing VisualSem gloss languages.

        This file is the file created with `save_glosses_languages`.
    """
    assert( os.path.isfile(in_file) )
    with open(in_file, 'r') as fh:
         x = json.load(fh)
    return x


def write_glosses_per_language(dict_available_languages, glosses_dict, path_to_visualsem):
    """
        dict_available_languages(dict):             Dictionary including a list of the available languages for glosses.
        glosses_dict(dict):                         Glosses available for each VisualSem node.
        path_to_visualsem(str):                     Path to the directory where to find VisualSem.

        This method will create two files for each language in `dict_available_languages`:
        `glosses.{language_code}.txt` and `glosses.{language_code}.txt.bnids`.

        The first file will include all glosses for the given language (across all nodes), whereas the second file
        is sentence-aligned to the first and has the BNid (i.e. VisualSem node) for each gloss.

        This is done to simplify retrieval where one wants to retrieve using a single language only.
    """
    os.makedirs('%s/dataset/gloss_files/'%path_to_visualsem, exist_ok=True)

    all_bnids = []
    glosses_to_bnids = []
    glosses_list = []

    # variable `n_glosses` is used to count all glosses across all BN ids and languages
    n_glosses_added = 0

    # for each language, create an output file and write glosses to it
    for lang_input in dict_available_languages['available_languages']:
        with open('%s/dataset/gloss_files/glosses.%s.txt.bnids'%(path_to_visualsem, lang_input), 'w') as fh_glosses_bnids:
            with open('%s/dataset/gloss_files/glosses.%s.txt'%(path_to_visualsem, lang_input), 'w') as fh_glosses:
                n_glosses_per_lang_added = 0
                n_empty_glosses_per_lang = 0
                glosses_list_per_lang = []
                for bnid_glosses in glosses_dict:
                    for bnid, all_glosses in bnid_glosses.items():
                        all_bnids.append( bnid )

                        for lang, lang_glosses in all_glosses.items():
                            if lang.lower() == lang_input: 
                                for gloss in lang_glosses:
                                    gloss = gloss.strip()
                                    if gloss.strip()=="" or gloss.strip().split(" ")==[]:
                                        n_empty_glosses_per_lang += 1
                                        continue

                                    if "\n" in gloss:
                                        print(n_glosses_added)
                                        print(n_glosses_per_lang_added)
                                        print(gloss)
                                        raise Exception()

                                    fh_glosses.write( gloss + "\n" )
                                    fh_glosses_bnids.write( bnid + "\n" )

                                    # save to which BNID this gloss refers to
                                    glosses_to_bnids.append( bnid )
                                    glosses_list.append( gloss )
                                    glosses_list_per_lang.append( gloss )
                                    n_glosses_added += 1
                                    n_glosses_per_lang_added += 1

                                    assert(len(glosses_list)==n_glosses_added)
                                    assert(len(glosses_list_per_lang)==n_glosses_per_lang_added)

                print("%s: %i/%i non-empty/empty glosses (acc. %i glosses)."%(lang_input, n_glosses_per_lang_added, n_empty_glosses_per_lang, n_glosses_added))

    return glosses_to_bnids, glosses_list, list(set(all_bnids))


if __name__=="__main__":
    # Path to VisualSem
    visualsem_path = os.path.dirname(os.path.realpath(__file__))
    # VisualSem nodes file
    path_to_visualsem_json = "%s/dataset/nodes.v2.json"%visualsem_path
    # VisualSem glosses file (to be created)
    path_to_glosses_json = "%s/dataset/gloss_files/nodes.glosses.json"%visualsem_path
    # VisualSem gloss languages file (to be created)
    path_to_glosses_languages_json = "%s/dataset/gloss_files/nodes.glosses_languages.json"%visualsem_path
    # List of languages to include glosses for
    langs_list = ["EN", "DE", "FR", "ZH", "IT", "PT", "ES", "NL", "KO", "FA", "AR", "RU", "PL", "SV"]

    #count_languages_per_node( path_to_glosses_languages_json )
    p = argparse.ArgumentParser()
    p.add_argument('--babelnet_server_uri', type=str, default=BN_SERVER_URI,
            help="""URI of an active BabelNet v4.0 server to use to retrieve glosses.""")
    p.add_argument('--path_to_visualsem_json', type=str, default=path_to_visualsem_json,
            help="Path to JSON file with VisualSem nodes.")
    p.add_argument('--extract_glosses_languages', action='store_true',
            help="Whether to iterate VisualSem BabelNet ids and query BabelNet for all glosses' languages for all nodes.")
    p.add_argument('--path_to_glosses_languages', type=str, default=path_to_glosses_languages_json,
            help="""Where to store/load glosses languages. If `--extract_glosses_languages`, create file. 
            If the flag is not set, load this file instead.""")
    p.add_argument('--extract_glosses', action='store_true',
            help="Whether to iterate VisualSem BabelNet ids and query BabelNet for all glosses for all nodes.")
    p.add_argument('--path_to_glosses', type=str, default=path_to_glosses_json,
            help="""Where to store glosses for all nodes in VisualSem.""")
    p.add_argument('--language_codes_list', type=str, nargs="+", default=langs_list,
            help="""List of language codes to include. Must be supported by BabelNet v4.0""")
    args = p.parse_args()
    # BabelNet expects uppercased language codes
    args.language_codes_list = [l.upper() for l in args.language_codes_list]
    os.makedirs('%s/dataset/gloss_files/'%path_to_visualsem, exist_ok=True)

    # Load VisualSem nodes
    with open(args.path_to_visualsem_json, 'r') as fh:
        visualsem_nodes_json = json.load(fh)
 
    # Extract glosses
    if args.extract_glosses:
        print("Extracting glosses for all BN ids in VisualSem (%i) and saving in %s..."%(
            len(visualsem_nodes_json), args.path_to_glosses))
        save_glosses( visualsem_nodes_json, args.path_to_glosses, args.language_codes_list )

    # Extract/load glosses languages
    if args.extract_glosses_languages:
        print("Extracting glosses languages' for all BN idxs in VisualSem (%i)..."%len(visualsem_nodes_json))
        save_glosses_languages( visualsem_nodes_json, args.path_to_glosses_languages )
    assert( os.path.isfile(args.path_to_glosses_languages) ), \
            "Glosses languages file not found: %s"%args.path_to_glosses_languages

    available_languages_per_bnid = load_glosses_languages( args.path_to_glosses_languages )

    # Print some information
    visualsem_nodes = set(visualsem_nodes_json.keys())
    nodes_included = set()
    nodes_included_counts = {}
    for l in args.language_codes_list:
        lang_available = [l for bnid,bnid_langs in available_languages_per_bnid.items() if l in bnid_langs]
        bnid_lang_available = set([bnid for bnid,bnid_langs in available_languages_per_bnid.items() if l in bnid_langs])
        nodes_included = nodes_included.union( bnid_lang_available )
        for node_name in bnid_lang_available:
            if not node_name in nodes_included_counts:
                nodes_included_counts[ node_name ] = 0
            nodes_included_counts[ node_name ] += 1
        print("%s is available in %i/%i entries."%(l,len(lang_available),len(visualsem_nodes)))
        print("%i/%i nodes included insofar."%(len(nodes_included),len(visualsem_nodes)))
        #print("... example gloss: ")

    nodes_missing = visualsem_nodes.difference( nodes_included )
    inv_nodes_included_counts = {}
    for k, v in nodes_included_counts.items():
        inv_nodes_included_counts[v] = inv_nodes_included_counts.get(v, [])
        inv_nodes_included_counts[v].append(k)

    inv_nodes_included_counts = {k:len(v) for k,v in inv_nodes_included_counts.items()}
    print("dictionary {number of languages for glosses: number of nodes}: ", sorted(inv_nodes_included_counts.items(), key=lambda kv: kv[0])[::-1])

    # Write individual ".txt" files for each language
    glosses_dict = load_glosses(args.path_to_glosses)
    # the first entry in the glosses dictionary are the languages covered.
    glosses_to_bnids, glosses_list, all_bnids = write_glosses_per_language(glosses_dict[0], glosses_dict[1:], visualsem_path)

