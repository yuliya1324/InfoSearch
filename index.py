import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import os
import pymorphy2
import argparse
from pathlib import Path
from tqdm import tqdm

morph = pymorphy2.MorphAnalyzer()
STOPWORDS = stopwords.words('russian') + list(punctuation)

def parse_args():
    parser = argparse.ArgumentParser(description="Build index on a corpus")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to directory with data")
    args = parser.parse_args()
    return args

def normalize_text(text):
    return [
        morph.parse(word)[0].normal_form
        for word in word_tokenize(text.lower()) 
        if (re.search(r"[^a-zа-я ]", word) is None) and word not in STOPWORDS
        ]

def get_corpus(data_dir):
    corpus = []

    curr_dir = os.getcwd()
    filepath = os.path.join(curr_dir, data_dir)
    for root, dirs, files in tqdm(os.walk(filepath)):
        for name in tqdm(files):
            with open(os.path.join(root, name), 'r', encoding="utf-8") as f:  
                corpus.append(normalize_text(f.read()))
    return corpus

def get_index_matrix(corpus):
    def do_nothing(tokens):
        return tokens

    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=do_nothing,
        preprocessor=None,
        lowercase=False)
    matrix = vectorizer.fit_transform(corpus).toarray()
    return vectorizer, matrix

def get_index_dict(words, matrix):
    return {
        word: matrix[:, i]
        for i, word in enumerate(words)
        }

def count_freq(matrix, words):
    matrix_freq = np.asarray(matrix.sum(axis=0)).ravel()
    most_freq = matrix_freq.argmax()
    least_freq = matrix_freq.argmin()
    return words[most_freq], words[least_freq], matrix_freq

def find_in_each_doc(di):
    li = []
    for key in di:
        if di[key].all(0):
            li.append(key)
    return li

def find_person(vectorizer, matrix_freq):
    persons = [
        ["Моника", "Мон"],
        ["Рэйчел", "Рейч"],
        ["Чендлер", "Чэндлер", "Чен"],
        ["Фиби", "Фибс"],
        ["Росс"],
        ["Джоуи", "Джои", "Джо"],
    ]
    freqs = [0]*len(persons)
    for i, person in enumerate(persons):
        for name in person:
            idx = vectorizer.vocabulary_.get(name.lower())
            if idx:
                freqs[i] += matrix_freq[idx]
    most_freq = np.array(freqs).argmax()
    return persons[most_freq][0]

def print_ans(matrix, words, di, vectorizer):
    most_freq, least_freq, matrix_freq = count_freq(matrix, words)
    li = find_in_each_doc(di)
    person = find_person(vectorizer, matrix_freq)

    print(f"a) самое частотное слово: {most_freq}")
    print(f"b) самое редкое слово: {least_freq}")
    print(f"c) слова, которые есть во всех документах: {li}")
    print(f"d) самый популярный герой: {person}")


def main(data_dir):
    corpus = get_corpus(data_dir)
    vectorizer, matrix = get_index_matrix(corpus)
    words = vectorizer.get_feature_names()
    di = get_index_dict(words, matrix)
    print_ans(matrix, words, di, vectorizer)

if __name__ == '__main__':
    args = parse_args()
    main(args.data_dir)