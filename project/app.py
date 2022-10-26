import streamlit as st
import argparse
from pathlib import Path
from database_bert import DataBaseBert
from database_bm25 import DataBaseBM25
from database_tfidf import DataBaseTFIDF
import numpy as np
from time import time

def parse_args():
    parser = argparse.ArgumentParser(description="Perform search in corpus")
    parser.add_argument("--dir_model", type=Path, required=False, default="sbert_large_nlu_ru", help="Path to directory with cached model")
    parser.add_argument("--matrix_filename_bert", type=Path, required=False, default="data/bert_matrix.pkl", help="Path to file with question matrix for Bert")
    parser.add_argument("--matrix_filename_bm25", type=Path, required=False, default="data/bm25_matrix.pkl", help="Path to file with question matrix for BM25")
    parser.add_argument("--matrix_filename_tfidf", type=Path, required=False, default="data/tfidf_matrix.pkl", help="Path to file with question matrix for TF-IDF")
    parser.add_argument("--answers_filename", type=Path, required=False, default="data/answers.txt", help="Path to file with document names")
    parser.add_argument("--device", type=str, required=False, help="device", default="cpu")
    parser.add_argument("--count_vectorizer_filename", type=Path, required=False, default="data/count_vectorizer.pkl", help="Path to file with vectorizer")
    parser.add_argument("--tfidf_vectorizer_filename", type=Path, required=False, default="data/tfidf_vectorizer.pkl", help="Path to file with vectorizer")
    
    args = parser.parse_args()
    return args

def find_ans(db, query, n_answers, answers):
    doc_idx = db.count_similarity(query)
    sorted_scores_indx = np.argsort(doc_idx, axis=0)[::-1]
    return np.array(answers)[sorted_scores_indx.ravel()][:n_answers]

def main(args):
    with open(args.answers_filename, encoding="utf-8") as file:
            answers = file.read().split("\n")
    db_bert = DataBaseBert(
        args.matrix_filename_bert,  
        args.dir_model,
        args.device
        )
    db_bm25 = DataBaseBM25(
        args.count_vectorizer_filename,
        args.matrix_filename_bm25, 
    )
    db_tfidf = DataBaseTFIDF(
        args.tfidf_vectorizer_filename,
        args.matrix_filename_tfidf, 
    )
    algoritms = {
        "Bert": db_bert,
        "BM25": db_bm25,
        "TF-IDF": db_tfidf,
    }

    st.title("Поисковик")

    query = st.text_input("Введите запрос")
    algorithm = st.selectbox("Выберете метод", ["TF-IDF", "BM25", "Bert"])
    n_answers = st.slider("Количество ответов", min_value=1, max_value=100)
    if st.button("Поиск"):
        start = time()
        answers = find_ans(algoritms[algorithm], query, n_answers, answers)
        st.write("Ответы")
        for i, answer in enumerate(answers):
            st.write(f"{i+1}) {answer}")
        st.write(f"Поиск занял {round(time() - start, 2)} секунд")

if __name__ == '__main__':
    args = parse_args()
    main(args)
