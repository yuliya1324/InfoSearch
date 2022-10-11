from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from string import punctuation
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import pymorphy2
from tqdm import tqdm
import pickle
from pathlib import Path
import scipy
import jsonlines
import numpy as np
from scipy import sparse

nltk.download('stopwords')

class DataBaseBM25:
    def __init__(self, data_dir: Path, count_vectorizer_filename: Path, questions_matrix_filename: Path, answers_matrix_filename: Path, build_corpus: bool, answers_filename: Path):
        self.morph = pymorphy2.MorphAnalyzer()
        self.stopwords = stopwords.words('russian') + list(punctuation)

        self.k = 2
        self.b = 0.75

        if not build_corpus:
            self.count_vectorizer = pickle.load(open(count_vectorizer_filename, "rb"))
            self.matrix_questions = pickle.load(open(questions_matrix_filename, "rb"))
            self.matrix_answers = pickle.load(open(answers_matrix_filename, "rb"))
            with open(answers_filename, encoding="utf-8") as file:
                self.answers = file.read().split("\n")
        else:
            self.questions = []
            self.answers = []
            self.answers_norm = []
            self.matrix_questions = None
            self.matrix_answers = None
            self.count_vectorizer = CountVectorizer(
                analyzer='word',
                tokenizer=self.do_nothing,
                preprocessor=None,
                lowercase=False
                )
            self.tfidf_vectorizer = TfidfVectorizer(
                use_idf=True, 
                norm='l2',
                analyzer='word',
                tokenizer=self.do_nothing,
                preprocessor=None,
                lowercase=False
                )
            self.get_corpus(data_dir, answers_filename)
            self.get_index(count_vectorizer_filename, answers_matrix_filename, questions_matrix_filename)

    @staticmethod
    def do_nothing(tokens: list) -> list:
        return tokens

    def normalize_text(self, text: str) -> list:
        return [
            self.morph.parse(word)[0].normal_form
            for word in word_tokenize(text.lower()) 
            if (re.search(r"[^a-zа-я ]", word) is None) and word not in self.stopwords
            ]

    def get_corpus(self, data_dir: Path, answers_filename: Path):
        with jsonlines.open(data_dir) as reader:
            i = 0
            for item in tqdm(reader, total=50000):
                if i == 50000:
                    break
                q = item["question"]
                ans = item["answers"]
                if q not in self.questions and ans:
                    self.questions.append(self.normalize_text(q))
                    values = [a["author_rating"]["value"] for a in ans]
                    a = ans[np.argmax(values)]["text"]
                    self.answers.append(a)
                    self.answers_norm.append(self.normalize_text(a))
                    i += 1
        
        with open(answers_filename, "w", encoding="utf-8") as file:
            file.write("\n".join(self.answers))

    def get_index(self, count_vectorizer_filename: Path, answers_matrix_filename: Path, questions_matrix_filename: Path) -> scipy.sparse.csr.csr_matrix:
        tf = self.count_vectorizer.fit_transform(self.answers_norm)
        tfidf = self.tfidf_vectorizer.fit_transform(self.answers_norm)
        idf = self.tfidf_vectorizer.idf_
        pickle.dump(self.count_vectorizer, open(count_vectorizer_filename, "wb"))

        len_d = tf.sum(axis=1)
        avdl = len_d.mean()

        values = []
        row = []
        col = []
        for i, j in zip(*tf.nonzero()):
            values.append(
                (tf[i,j] * idf[j] * (self.k+1))/(tf[i,j] + self.k * (1 - self.b + self.b * len_d[i,0] / avdl))
                )
            row.append(i)
            col.append(j)

        self.matrix_answers = sparse.csr_matrix((values, (row, col)), shape=tf.shape)
        pickle.dump(self.matrix_answers, open(answers_matrix_filename, "wb"))

        tf = self.count_vectorizer.transform(self.questions)
        len_d = tf.sum(axis=1)
        avdl = len_d.mean()

        values = []
        row = []
        col = []
        for i, j in zip(*tf.nonzero()):
            values.append(
                (tf[i,j] * idf[j] * (self.k+1))/(tf[i,j] + self.k * (1 - self.b + self.b * len_d[i,0] / avdl))
                )
            row.append(i)
            col.append(j)

        self.matrix_questions = sparse.csr_matrix((values, (row, col)), shape=tf.shape)
        pickle.dump(self.matrix_questions, open(questions_matrix_filename, "wb"))

    def get_query(self, query: str) -> scipy.sparse.csr.csr_matrix:
        return self.count_vectorizer.transform([self.normalize_text(query)])
    
    def count_similarity(self, query: scipy.sparse.csr.csr_matrix):
        return np.dot(self.matrix_answers, query.T).toarray()

    def evaluate(self, n: int) -> int:
        matrix = np.dot(self.matrix_answers, self.matrix_questions.T)
        A = [
            1 
            if i in np.argsort(matrix[:, i], axis=0)[::-1][:n] 
            else 0 
            for i in range(matrix.shape[-1])
            ]
        return sum(A) / matrix.shape[0]