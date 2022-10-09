import numpy as np
import argparse
from pathlib import Path
from database_bert import DataBaseBert
from database_bm25 import DataBaseBM25

def parse_args():
    parser = argparse.ArgumentParser(description="Perform search in corpus")
    parser.add_argument("--build_corpus", action="store_true")
    parser.add_argument("--no-build_corpus", dest="build_corpus", action="store_false")
    parser.set_defaults(build_corpus=True)
    parser.add_argument("--query", type=str, required=True, help="String with query")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to directory with data")
    parser.add_argument("--dir_model", type=Path, required=False, default="sbert_large_nlu_ru", help="Path to directory with cached model")
    parser.add_argument("--questions_matrix_filename_bert", type=Path, required=False, default="bert/questions_matrix.pkl", help="Path to file with question matrix for Bert")
    parser.add_argument("--answers_matrix_filename_bert", type=Path, required=False, default="bert/answers_matrix.pkl", help="Path to file with corpus matrix for Bert")
    parser.add_argument("--questions_matrix_filename_bm25", type=Path, required=False, default="bm25/questions_matrix.pkl", help="Path to file with question matrix for BM25")
    parser.add_argument("--answers_matrix_filename_bm25", type=Path, required=False, default="bm25/answers_matrix.pkl", help="Path to file with corpus matrix for BM25")
    parser.add_argument("--answers_filename", type=Path, required=False, default="answers.txt", help="Path to file with document names")
    parser.add_argument("--n_answers", type=int, required=False, help="number of answers", default=None)
    parser.add_argument("--device", type=str, required=False, help="device", default="cpu")
    parser.add_argument("--count_vectorizer_filename", type=Path, required=False, default="bm25/count_vectorizer.pkl", help="Path to file with vectorizer")
    
    args = parser.parse_args()
    return args

def find_ans(db, query):
    query = db.get_query(query)
    doc_idx = db.count_similarity(query)
    return np.argsort(doc_idx, axis=0)[::-1]


def main(args):
    db_bert = DataBaseBert(
        args.data_dir,
        args.questions_matrix_filename_bert, 
        args.answers_matrix_filename_bert,  
        args.build_corpus, 
        args.answers_filename,
        args.dir_model,
        args.device
        )
    db_bm25 = DataBaseBM25(
        args.data_dir,
        args.count_vectorizer_filename,
        args.questions_matrix_filename_bm25, 
        args.answers_matrix_filename_bm25, 
        args.build_corpus, 
        args.answers_filename,
    )

    sorted_scores_indx = find_ans(db_bert, args.query)

    print(f"Task 1.\n\nYour query: {args.query}\nAnswers:\n")
    if args.n_answers:
        print("\n".join(np.array(db_bert.answers)[sorted_scores_indx.ravel()][:args.n_answers]))
    else:
        print("\n".join(np.array(db_bert.answers)[sorted_scores_indx.ravel()]))

    metrics_bm25 = db_bm25.evaluate(args.n_answers)
    metrics_bert = db_bert.evaluate(args.n_answers)
    print(f"Task 2.\n\nMetrics for BM25: {metrics_bm25}\nMetrics for Bert: {metrics_bert}\n")

if __name__ == '__main__':
    args = parse_args()
    main(args)