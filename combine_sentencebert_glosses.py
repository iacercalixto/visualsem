import sys
import os
import argparse
from utils import load_bnids
import h5py

if __name__=="__main__":
    strategies_list = ['all', 'all_but_swedish_and_farsi', 'top8']
    # For more details and discussion on VisualSem gloss quality/coverage, refer to: https://arxiv.org/pdf/2008.09150.pdf
    # We support the following strategies:
    # 'all': We simply use glosses in all languages concatenated as one large index.
    # 'all_but_swedish_and_farsi': As the name suggests, we use all languages but Swedish and Farsi.
    #       This is because the model we use (Sentence BERT) is not trained with sentences in these languages.
    # 'top8': We use the top-8 best-performing languages according to our experiments. These are: en, es, de, it, fr, pt, pl, nl.
    p = argparse.ArgumentParser()
    p.add_argument('--strategy', default='top8', choices=strategies_list, type=str,
            help="""For more details and discussion on VisualSem gloss quality/coverage, refer to: https://arxiv.org/pdf/2008.09150.pdf.
                    We support the following strategies:
                    'all': We simply use glosses in all languages concatenated as one large index.
                    'all_but_swedish_and_farsi': As the name suggests, we use all languages but Swedish and Farsi.
                        This is because the model we use (Sentence BERT) is not trained with sentences in these languages.
                    'top8': We use the top-8 best-performing languages according to our experiments. These are: en, es, de, it, fr, pt, pl, nl.""")
    args = p.parse_args()

    if args.strategy == 'all':
        languages = ['ar', 'de', 'en', 'es', 'fr', 'it', 'ko', 'nl', 'pt', 'ru', 'zh', 'fa', 'pl', 'sv']
    elif args.strategy == 'all_but_swedish_and_farsi':
        languages = ['ar', 'de', 'en', 'es', 'fr', 'it', 'ko', 'nl', 'pt', 'ru', 'zh', 'pl']
    elif args.strategy == 'top8':
        languages = ['de', 'en', 'es', 'fr', 'it', 'nl', 'pt', 'pl']
    languages = sorted(languages)
    gloss_bnids_fnames = ["glosses.%s.txt.bnids"%lang for lang in languages]
    gloss_hdf5_fnames  = ["glosses.%s.txt.sentencebert.h5"%lang for lang in languages]

    # concatenate BNids into single file
    gloss_bnid_outfname = os.path.join("dataset", "gloss_files", "glosses.combined-%s.bnids"%args.strategy)
    with open(gloss_bnid_outfname, 'w') as fh_out:
        n_lines = 0
        for gloss_bnid_fname in gloss_bnids_fnames:
            bnids = load_bnids( gloss_bnid_fname )
            for bnid in bnids:
                fh_out.write( bnid + "\n" )
                n_lines += 1
        print("Combined %i overall glosses into: %s"%(n_lines, gloss_bnid_outfname))

    # concatenate HDF5 features into single file
    gloss_hdf5_outfname = os.path.join("dataset", "gloss_files", 'glosses.combined-%s.h5'%args.strategy)
    with h5py.File(gloss_hdf5_outfname, 'w') as fh_out:
        shape_features = (n_lines, 512)
        fh_out.create_dataset("features", shape_features, dtype='float32', chunks=(1,512), maxshape=(None, 512), compression="gzip")

        from_idx = 0
        for gloss_hdf5_fname in gloss_hdf5_fnames:
            with h5py.File(gloss_hdf5_fname, 'r') as fh_in:
                gloss_feats = fh_in["features"][:]
                to_idx = from_idx + gloss_feats.shape[0]
                #gloss_feats = torch.tensor(gloss_feats)
                fh_out['features'][from_idx:to_idx] = gloss_feats
            from_idx = to_idx

        assert( from_idx == n_lines ), "from_idx: %i  -  n_lines: %i"%(from_idx, n_lines)
        print("Created %s"%gloss_hdf5_outfname)

