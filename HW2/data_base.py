from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import pymorphy2
from tqdm import tqdm
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import scipy

class DataBase:
    def __init__(self, data_dir: Path, vectorizer_filename: Path, matrix_filename: Path, build_corpus: bool, names_filename: Path):
        self.morph = pymorphy2.MorphAnalyzer()
        self.stopwords = stopwords.words('russian') + list(punctuation)
        self.names = []

        if not build_corpus:
            self.vectorizer = pickle.load(open(vectorizer_filename, "rb"))
            self.matrix = pickle.load(open(matrix_filename, "rb"))
            with open(names_filename, encoding="utf-8") as file:
                self.names = file.read().split("\n")
        else:
            self.vectorizer = TfidfVectorizer(
                analyzer='word',
                tokenizer=self.do_nothing,
                preprocessor=None,
                lowercase=False)
            self.corpus = self.get_corpus(data_dir)
            self.matrix = self.get_index(vectorizer_filename, matrix_filename)
            with open(names_filename, "w", encoding="utf-8") as file:
                file.write("\n".join(self.names))

    @staticmethod
    def do_nothing(tokens: list) -> list:
        return tokens

    def normalize_text(self, text: str) -> list:
        return [
            self.morph.parse(word)[0].normal_form
            for word in word_tokenize(text.lower()) 
            if (re.search(r"[^a-zа-я ]", word) is None) and word not in self.stopwords
            ]

    def get_corpus(self, data_dir: Path) -> list:
        corpus = []
        for root, dirs, files in tqdm(os.walk(data_dir)):
            for name in tqdm(files):
                self.names.append(name)
                with open(os.path.join(root, name), 'r', encoding="utf-8") as f:  
                    corpus.append(self.normalize_text(f.read()))
        return corpus

    def get_index(self, vectorizer_filename: Path, matrix_filename: Path) -> scipy.sparse.csr.csr_matrix:
        matrix = self.vectorizer.fit_transform(self.corpus)
        pickle.dump(self.vectorizer, open(vectorizer_filename, "wb"))
        pickle.dump(matrix, open(matrix_filename, "wb"))
        return matrix

    def get_query(self, query: str) -> scipy.sparse.csr.csr_matrix:
        return self.vectorizer.transform([self.normalize_text(query)])
    
    def count_similarity(self, query: scipy.sparse.csr.csr_matrix):
        return cosine_similarity(self.matrix, query).reshape(self.matrix.shape[0])