import pickle
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import io
from database_base import DataBase

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class DataBaseBert(DataBase):
    def __init__(self, answers_matrix_filename: Path, dir_model: Path, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(dir_model)
        self.model = AutoModel.from_pretrained(dir_model)
        self.model.to(device)
        self.matrix_answers = CPU_Unpickler(open(answers_matrix_filename, 'rb')).load()

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_query(self, query: str) -> torch.Tensor:
        encoded_input = self.tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors='pt')
        for k in encoded_input:
            encoded_input[k] = encoded_input[k].to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input['attention_mask'])
    
    def count_similarity(self, query: str):
        query_vector = self.get_query(query)
        return np.dot(self.matrix_answers, query_vector.T)
         