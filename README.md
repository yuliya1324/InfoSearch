# InfoSearch

## HW 1

Запустить файл `index.py`, передав в аргумент `--data_dir` путь к папке с данными. В консоли будет выведены progress bars при формировании корпуса, а в конце -- ответы на задания.

## HW 2

Запустить файл `search.py`, передав следующие аргументы:
* `--data_dir` путь к папке с данными
* `--query` запрос
* `--build_corpus` если надо собрать корпус / `--no-build_corpus` если корпус уже собран
* `--vectorizer_filename` путь к файлу с векторизатором (опционально, по дефолту "vectorizer.pkl")
* `--matrix_filename` путь к файлу с матрицей корпуса (опционально, по дефолту "matrix.pkl")
* `--names_filename` путь к файлу с именами документов (опционально, по дефолту "names.txt")

В консоли будет выведены progress bars при формировании корпуса, а в конце -- отсортированные по убыванию имена документов коллекции.
