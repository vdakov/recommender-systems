import torch
import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModel
from recommendation_algorithms.abstract_recommender import AbstractRecommender

class ContentBasedRecommender(AbstractRecommender):
    """
    Content-based recommender based on embeddings from BERT.
    """
    item_embeddings: np.array
    bert_model: str
    data: pd.DataFrame
    predicted_ratings: np.array
    min_val: float
    max_val: float
    aggregation_method: str
    
    def __init__(self, bert_model:str, data:pd.DataFrame, embedding_dimension:int, batch_size:int, aggregation_method:str, content:List[str]):
        super().__init__()
        self.item_embeddings = []
        self.bert_model = bert_model
        self.data = data
        self.embedding_dimension = embedding_dimension 
        self.batch_size = batch_size
        self.aggregation_method = aggregation_method
        self.content = content
        
    def train(self, train_data):
        self.train_embeddings(self.bert_model, self.content, self.embedding_dimension, self.batch_size,)

        predicted_ratings = train_data.apply(
            lambda row: self.predict_computability_between_user_and_item(
                row["user_id"], row["item_id"]
            ),
            axis=1
            )

        preds = predicted_ratings.to_numpy(dtype=float)
        self.predicted_ratings = preds
        self.min_val = np.min(preds)
        self.max_val = np.max(preds)
        

    def train_embeddings(self, model_name:str, content:pd.DataFrame, batch_size:int, embedding_dimension:int) -> None:
        """
        This method prepares the model for usage, in this case this means loading it 
        with content embeddings.
        """
        if content is None:
            return None

        if isinstance(content, pd.Series):
            content = content.fillna("").astype(str).tolist()
        elif isinstance(content, np.ndarray):
            content = content.astype(str).tolist()


        print(f"Loading BERT model: {model_name}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Set device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using cuda or cpu: {device}")
        model.to(device)
        model.eval()
        emb = []

        for i in tqdm(range(0, len(content), batch_size)):

            batch_texts = content[i:i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=embedding_dimension,
                return_tensors='pt'
            )

            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)

                # Use [CLS] token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                emb.extend(batch_embeddings)

        emb = np.array(emb)
        self.item_embeddings = emb

        print(f"BERT embeddings generated: {emb.shape}")
        print(f"Embedding dimension: {emb.shape[1]}")

        return emb 
    
    def get_item_emb(self, item_id) -> np.array:
        return self.item_embeddings[item_id - 1]
    
    def get_items_per_user_with_rating(self, user_id):
        mask = self.data["user_id"] == user_id
        user_data = self.data.loc[mask, ["item_id", "rating"]]
        ratings = user_data["rating"].to_numpy()
        embeddings = np.stack(user_data["item_id"].apply(lambda x: self.get_item_emb(x)))
        
        return embeddings, ratings
    
    def get_user_emb(self, user_id) -> np.array: 
        embeddings, ratings = self.get_items_per_user_with_rating(user_id)
        user_representation = []
        
        if self.aggregation_method == "average":
            user_representation = np.mean(embeddings, axis=0)
        elif self.aggregation_method == "weighted_average":
            weighted = [e * r for e, r in zip(embeddings, ratings)]
            rating_sum = np.sum(ratings)
            user_representation = np.sum(weighted, axis=0) / rating_sum
        elif self.aggregation_method ==  "avg_pos":
            embeddings_filtered = [emb for (emb, rat) in zip(embeddings, ratings) if rat >= 4]
            if(len(embeddings_filtered) > 0):
                user_representation = np.mean(embeddings_filtered, axis=0)
            else: 
                user_representation = np.zeros(embeddings[0].shape)
            
        
        return user_representation
    
    def get_name(self) -> str:
        """
        Get the name of the model (used for pretty printing).

        :return: The name of the model
        """
        return "Content-Based Recommender Using BERT Embeddings"
    
    def predict_computability_between_user_and_item(self, user_id, item_id):
        user_emb = self.get_user_emb(user_id)
        item_emb = self.get_item_emb(item_id)
        similarity = np.dot(user_emb, item_emb)
        
        
        return similarity
    

    def predict_score(self, user_id: int, item_id: int) -> float:
        """
        Predict a score a user would give a specific item.

        :param user_id: The id of the user
        :param item_id: The id of the item
        :return: Predicted score
        """
        score = self.predict_computability_between_user_and_item(user_id, item_id)
        rating = 1 + (score - self.min_val) * (4 / (self.max_val - self.min_val))
        return rating