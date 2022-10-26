from string import punctuation
import re
import nltk
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

nltk.download('stopwords')

class DataBaseBM25(DataBase):
    def __init__(self, count_vectorizer_filename: Path, answers_matrix_filename: Path):
        self.morph = pymorphy2.MorphAnalyzer()
        self.stopwords = stopwords.words('russian') + list(punctuation)
        self.count_vectorizer = pickle.load(open(count_vectorizer_filename, "rb"))
        self.matrix_answers = pickle.load(open(answers_matrix_filename, "rb"))
        # self.count_vectorizer = CountVectorizer(
        #         analyzer='word',
        #         tokenizer=self.do_nothing,
        #         preprocessor=None,
        #         lowercase=False
        #         )
        # self.tfidf_vectorizer = TfidfVectorizer(
        #         use_idf=True, 
        #         norm='l2',
        #         analyzer='word',
        #         tokenizer=self.do_nothing,
        #         preprocessor=None,
        #         lowercase=False
        #         )
        # self.questions = []
        # self.answers = []
        # self.k = 2
        # self.b = 0.75
        # self.get_index(count_vectorizer_filename, answers_matrix_filename)

    @staticmethod
    def do_nothing(tokens: list) -> list:
        return tokens
    
    def normalize_text(self, text: str) -> list:
        return [
            self.morph.parse(word)[0].normal_form
            for word in word_tokenize(text.lower()) 
            if (re.search(r"[^a-zа-я ]", word) is None) and word not in self.stopwords
            ]

    # def get_corpus(self):
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

    # def get_index(self, count_vectorizer_filename: Path, matrix_filename: Path) -> scipy.sparse.csr.csr_matrix:
    #     self.get_corpus()
    #     tf = self.count_vectorizer.fit_transform(self.questions)
    #     tfidf = self.tfidf_vectorizer.fit_transform(self.questions)
    #     idf = self.tfidf_vectorizer.idf_

    #     pickle.dump(self.count_vectorizer, open(count_vectorizer_filename, "wb"))

    #     len_d = tf.sum(axis=1)
    #     avdl = len_d.mean()

    #     values = []
    #     row = []
    #     col = []
    #     for i, j in zip(*tf.nonzero()):
    #         values.append(
    #             (tf[i,j] * idf[j] * (self.k+1))/(tf[i,j] + self.k * (1 - self.b + self.b * len_d[i,0] / avdl))
    #             )
    #         row.append(i)
    #         col.append(j)

    #     self.matrix = scipy.sparse.csr_matrix((values, (row, col)), shape=tf.shape)
    #     pickle.dump(self.matrix, open(matrix_filename, "wb"))
    
    def get_query(self, query: str) -> scipy.sparse.csr.csr_matrix:
        return self.count_vectorizer.transform([self.normalize_text(query)])
