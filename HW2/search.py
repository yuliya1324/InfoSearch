import numpy as np
import argparse
from pathlib import Path
from data_base import DataBase

def parse_args():
    parser = argparse.ArgumentParser(description="Perform search in corpus")
    parser.add_argument("--build_corpus", action="store_true")
    parser.add_argument("--no-build_corpus", dest="build_corpus", action="store_false")
    parser.set_defaults(build_corpus=True)
    parser.add_argument("--query", type=str, required=True, help="String with query")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to directory with data")
    parser.add_argument("--vectorizer_filename", type=Path, required=False, default="vectorizer.pkl", help="Path to file with vectorizer")
    parser.add_argument("--matrix_filename", type=Path, required=False, default="matrix.pkl", help="Path to file with corpus matrix")
    parser.add_argument("--names_filename", type=Path, required=False, default="names.txt", help="Path to file with document names")
    args = parser.parse_args()
    return args

def main(args):
    db = DataBase(args.data_dir, args.vectorizer_filename, args.matrix_filename, args.build_corpus, args.names_filename)
    query = db.get_query(args.query)
    doc_idx = db.count_similarity(query)
    idx = list(np.argsort(-doc_idx))
    print("\n".join([db.names[i] for i in idx]))

if __name__ == '__main__':
    args = parse_args()
    main(args)