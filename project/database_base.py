from sklearn.metrics.pairwise import cosine_similarity

class DataBase:
    def get_query(self, query: str):
        raise NotImplementedError()
    
    def count_similarity(self, query: str):
        query_vector = self.get_query(query)
        return cosine_similarity(self.matrix_answers, query_vector).reshape(self.matrix_answers.shape[0])
