import numpy as np
import argparse
from pathlib import Path
from database import DataBase

def parse_args():
    parser = argparse.ArgumentParser(description="Perform search in corpus")
    parser.add_argument("--build_corpus", action="store_true")
    parser.add_argument("--no-build_corpus", dest="build_corpus", action="store_false")
    parser.set_defaults(build_corpus=True)
    parser.add_argument("--query", type=str, required=True, help="String with query")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to directory with data")
    parser.add_argument("--tfidf_vectorizer_filename", type=Path, required=False, default="data_hw3/tfidf_vectorizer.pkl", help="Path to file with vectorizer")
    parser.add_argument("--count_vectorizer_filename", type=Path, required=False, default="data_hw3/count_vectorizer.pkl", help="Path to file with vectorizer")
    parser.add_argument("--matrix_filename", type=Path, required=False, default="data_hw3/matrix.pkl", help="Path to file with corpus matrix")
    parser.add_argument("--answers_filename", type=Path, required=False, default="data_hw3/answers.txt", help="Path to file with document names")
    args = parser.parse_args()
    return args

def main(args):
    db = DataBase(
        args.data_dir, 
        args.tfidf_vectorizer_filename,
        args.count_vectorizer_filename,
        args.matrix_filename, 
        args.build_corpus, 
        args.answers_filename,
        )
    query = db.get_query(args.query)
    doc_idx = db.count_similarity(query)
    sorted_scores_indx = np.argsort(doc_idx, axis=0)[::-1]
    print("\n".join(np.array(db.answers)[sorted_scores_indx.ravel()]))

if __name__ == '__main__':
    args = parse_args()
    main(args)