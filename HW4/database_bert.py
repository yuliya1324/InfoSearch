from tqdm import tqdm
import pickle
from pathlib import Path
import jsonlines
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class DataBaseBert:
    def __init__(self, data_dir: Path, questions_matrix_filename: Path, answers_matrix_filename: Path, build_corpus: bool, answers_filename: Path, dir_model: Path, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(dir_model)
        self.model = AutoModel.from_pretrained(dir_model)
        self.model.to(device)

        if not build_corpus:
            self.matrix_questions = CPU_Unpickler(open(questions_matrix_filename, 'rb')).load()
            self.matrix_answers = CPU_Unpickler(open(answers_matrix_filename, 'rb')).load()
            with open(answers_filename, encoding="utf-8") as file:
                self.answers = file.read().split("\n")
        else:
            self.questions = []
            self.answers = []
            self.matrix_questions = None
            self.matrix_answers = None
            self.get_corpus(data_dir, answers_filename)
            self.get_index(questions_matrix_filename, answers_matrix_filename)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_corpus(self, data_dir: Path, answers_filename: Path):
        with jsonlines.open(data_dir) as reader:
            i = 0
            for item in tqdm(reader, total=50000):
                if i == 50000:
                    break
                q = item["question"]
                ans = item["answers"]
                if q not in self.questions and ans:
                    self.questions.append(q)
                    values = [a["author_rating"]["value"] for a in ans]
                    self.answers.append(ans[np.argmax(values)]["text"])
                    i += 1
        
        with open(answers_filename, "w", encoding="utf-8") as file:
            file.write("\n".join(self.answers))

    def get_index(self, questions_matrix_filename, answers_matrix_filename):
        result = []
        for i in range(500):
            encoded_input = self.tokenizer(self.questions[i*100:(i+1)*100], padding=True, truncation=True, max_length=512, return_tensors='pt')
            for k in encoded_input:
                encoded_input[k] = encoded_input[k].to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            result.append(self.mean_pooling(model_output, encoded_input['attention_mask']))
        self.matrix_questions = torch.stack(result)
        self.matrix_questions = self.matrix_questions.reshape((self.matrix_questions.shape[0]*self.matrix_questions.shape[1], self.matrix_questions.shape[-1]))
        pickle.dump(self.matrix_questions, open(questions_matrix_filename, "wb"))

        result = []
        for i in range(500):
            encoded_input = self.tokenizer(self.answers[i*100:(i+1)*100], padding=True, truncation=True, max_length=512, return_tensors='pt')
            for k in encoded_input:
                encoded_input[k] = encoded_input[k].to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            result.append(self.mean_pooling(model_output, encoded_input['attention_mask']))
        self.matrix_answers = torch.stack(result) #concat
        self.matrix_answers = self.matrix_answers.reshape((self.matrix_answers.shape[0]*self.matrix_answers.shape[1], self.matrix_answers.shape[-1]))
        pickle.dump(self.matrix_answers, open(answers_matrix_filename, "wb"))

    def get_query(self, query: str) -> torch.Tensor:
        encoded_input = self.tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors='pt')
        for k in encoded_input:
            encoded_input[k] = encoded_input[k].to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])
    
    def count_similarity(self, query: torch.Tensor):
         return np.dot(self.matrix_answers, query.T)
    
    def evaluate(self, n: int) -> int:
        matrix = np.dot(self.matrix_answers[:10000], self.matrix_questions[:10000].T)
        docs = np.arange(matrix.shape[0])
        return np.sum(np.argsort(-matrix, axis=1)[:, :n] == np.tile(docs, (n, 1)).T) / matrix.shape[0]
        