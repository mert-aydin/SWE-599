import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


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

    def calculate_similarity(self, embedding1, embedding2):
        return cosine_similarity(embedding1, embedding2)

    def preprocess(self, text):
        """
        Preprocess text for BERT analysis.

        :param text: The text to be preprocessed.
        :return: A dictionary containing input ids and attention masks.
        """
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        return encoded_input

    def analyze_text(self, text):
        """
        Analyze the given text using the BERT model.

        :param text: The text to be analyzed.
        :return: The model's output.
        """
        # Preprocess the text
        inputs = self.preprocess(text)

        # Move to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Analyze the text
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process the model output as needed (e.g., extracting the logits)
        logits = outputs.logits
        return logits

    def process_message(self, message):
        if message is None:
            return None

        return self.analyze_text(message)
