import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


class BERTProcessor:
    def __init__(self, model_name='sentence-transformers/bert-base-nli-mean-tokens'):
        """
        Initialize the BERTProcessor with a specified BERT model.

        :param model_name: Name of the BERT model to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode_texts(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings

    @staticmethod
    def calculate_similarity(embedding1, embedding2):
        return cosine_similarity(embedding1, embedding2)
