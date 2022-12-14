# InfoSearch

## HW 1

Запустить файл `HW1/index.py`, передав в аргумент `--data_dir` путь к папке с данными. В консоли будет выведены progress bars при формировании корпуса, а в конце -- ответы на задания.

## HW 2

Запустить файл `HW2/search.py`, передав следующие аргументы:
* `--data_dir` путь к папке с данными
* `--query` запрос
* `--build_corpus` если надо собрать корпус / `--no-build_corpus` если корпус уже собран
* `--vectorizer_filename` путь к файлу с векторизатором (опционально, по дефолту "vectorizer.pkl")
* `--matrix_filename` путь к файлу с матрицей корпуса (опционально, по дефолту "matrix.pkl")
* `--names_filename` путь к файлу с именами документов (опционально, по дефолту "names.txt")

В консоли будет выведены progress bars при формировании корпуса, а в конце -- отсортированные по убыванию имена документов коллекции.

## HW 3

Запустить файл `HW3/search.py`, передав следующие аргументы:
* `--data_dir` путь к папке с данными
* `--query` запрос
* `--build_corpus` если надо собрать корпус / `--no-build_corpus` если корпус уже собран
* `--tfidf_vectorizer_filename` путь к файлу с векторизатором (опционально, по дефолту "data_hw3/tfidf_vectorizer.pkl")
* `--count_vectorizer_filename` путь к файлу с векторизатором (опционально, по дефолту "data_hw3/count_vectorizer.pkl")
* `--matrix_filename` путь к файлу с матрицей корпуса (опционально, по дефолту "data_hw3/matrix.pkl")
* `--answers_filename` путь к файлу с именами документов (опционально, по дефолту "data_hw3/answers.txt")
* `--n_answers` число -- сколько ответов вывести (опционально, если не задано, то выводятся все ответы из корпуса)

В консоли будет выведены progress bars при формировании корпуса, а в конце -- отсортированные по убыванию ответы на вопрос.

В качестве документов я использовала вопросы, а в качестве названий ответы на вопросы.

## HW 4

Запустить файл `HW4/search.py`, передав следующие аргументы:
* `--data_dir` путь к папке с данными
* `--query` запрос
* `--build_corpus` если надо собрать корпус / `--no-build_corpus` если корпус уже собран
* `--count_vectorizer_filename` путь к файлу с векторизатором (опционально, по дефолту "bm25/count_vectorizer.pkl")
* `--answers_filename` путь к файлу с именами документов (опционально, по дефолту "bert/answers.txt")
* `--n_answers` число -- сколько ответов вывести (опционально, если не задано, то выводятся все ответы из корпуса)
* `--dir_model` путь к папке с моедлью (опционально, по дефолту "sbert_large_nlu_ru")
* `--questions_matrix_filename_bert` (опционально, по дефолту "bert/questions_matrix.pkl")
* `--answers_matrix_filename_bert` (опционально, по дефолту "bert/answers_matrix.pkl")
* `--questions_matrix_filename_bm25` (опционально, по дефолту "bm25/questions_matrix.pkl")
* `--answers_matrix_filename_bm25` (опционально, по дефолту "bm25/answers_matrix.pkl")
* `--device` (опционально, по дефолту "cpu")
* `--task` номер задания (опционально, по дефолту "1")

В консоли будет выведены progress bars при формировании корпуса, а в конце -- ответы на задания:
1. отсортированные по убыванию ответы на вопрос
2. метрики для каждого способа векторизации

Метрики, которые у меня получились:
```
Metrics for BM25: 0.1125
Metrics for Bert: 0.0697
```

## Project

Intall requirements

```shell
pip install -r requirements.txt
```

To run project

```shell
streamlit run app.py -- [--args]
```

Arguments:
* dir_model
* matrix_filename_bert
* matrix_filename_bm25
* matrix_filename_tfidf
* answers_filename
* device
* count_vectorizer_filename
* tfidf_vectorizer_filename
