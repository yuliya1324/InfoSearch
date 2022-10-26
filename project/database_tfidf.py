from string import punctuation
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import pymorphy2
import pickle
from pathlib import Path
import scipy
from database_base import DataBase
import jsonlines
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class DataBaseTFIDF(DataBase):
    def __init__(self, vectorizer_filename: Path, matrix_filename: Path):
        self.morph = pymorphy2.MorphAnalyzer()
        self.stopwords = stopwords.words('russian') + list(punctuation)
        self.vectorizer = pickle.load(open(vectorizer_filename, "rb"))
        self.matrix_answers = pickle.load(open(matrix_filename, "rb"))
        # self.questions = []
        # self.answers = []
        # self.vectorizer = TfidfVectorizer(
        #         analyzer='word',
        #         tokenizer=self.do_nothing,
        #         preprocessor=None,
        #         lowercase=False)
        # self.get_index(vectorizer_filename, matrix_filename)

    @staticmethod
    def do_nothing(tokens: list) -> list:
        return tokens

    def normalize_text(self, text: str) -> list:
        return [
            self.morph.parse(word)[0].normal_form
            for word in word_tokenize(text.lower()) 
            if (re.search(r"[^a-zа-я ]", word) is None) and word not in self.stopwords
            ]
    
    # def get_corpus(self) -> list:
    #     with jsonlines.open('data/data.jsonl') as reader:
    #         i = 0
    #         for item in tqdm(reader, total=50000):
    #             if i == 50000:
    #                 break
    #             q = item["question"]
    #             ans = item["answers"]
    #             if q not in self.questions and ans:
    #                 self.questions.append(self.normalize_text(q))
    #                 values = [a["author_rating"]["value"] for a in ans]
    #                 self.answers.append(ans[np.argmax(values)]["text"])
    #                 i += 1

    # def get_index(self, vectorizer_filename: Path, matrix_filename: Path) -> scipy.sparse.csr.csr_matrix:
    #     self.get_corpus()
    #     matrix = self.vectorizer.fit_transform(self.questions)
    #     pickle.dump(self.vectorizer, open(vectorizer_filename, "wb"))
    #     pickle.dump(matrix, open(matrix_filename, "wb"))

    def get_query(self, query: str) -> scipy.sparse.csr.csr_matrix:
        return self.vectorizer.transform([self.normalize_text(query)])