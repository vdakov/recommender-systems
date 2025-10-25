import itertools
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List
from recommendation_algorithms.abstract_recommender import AbstractRecommender

class ContentBasedRecommender(AbstractRecommender):
    """
    Content-based recommender based on embeddings from BERT.
    """
    item_embeddings: np.array(np.array(int))
    bert_model_name: str
    data: pd.DataFrame
    predicted_ratings: np.array(int)
    min_val: float
    max_val: float
    aggregation_method: str
    
    def __init__(self, bert_model_name:str, data:pd.DataFrame()):
        super().__init__()
        self.item_embeddings = []
        self.bert_model_name = bert_model_name
        self.data = data
        
    def train(self, content, train_data, batch_size, aggregation_method):
        self.train_embeddings(self.bert_model_name, content, batch_size)
        self.aggregation_method = aggregation_method
        predicted_ratings = train_data.apply(
            lambda row: self.predict_computability_between_user_and_item(
                row["user_id"], row["item_id"], aggregation_method
            ),
            axis=1
            )

        preds = predicted_ratings.to_numpy(dtype=float)
        self.predicted_ratings = preds
        self.min_val = np.min(preds)
        self.max_val = np.max(preds)
        self.aggregation_method = aggregation_method
        

    def train_embeddings(self, model_name:str, content:pd.DataFrame, batch_size:int) -> None:
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
                max_length=512,
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
        ratings = user_data["rating"]
        embeddings = user_data["item_id"].apply(lambda x: self.get_item_emb(x))
        
        return embeddings, ratings
    
    def get_user_emb(self, user_id) -> np.array: 
        embeddings, ratings = self.get_items_per_user_with_rating(user_id)
        user_representation = []
        
        if self.aggregation_method == "average":
            user_representation = np.mean(embeddings)
        elif self.aggregation_method == "weighted_average":
            user_representation = np.dot(ratings, embeddings)
            user_representation = np.divide(user_representation, np.sum(ratings))
        elif self.aggregation_method ==  "avg_pos":
            embeddings_filtered = [emb for (emb, rat) in zip(embeddings, ratings) if rat >= 4]
            user_representation = np.mean(embeddings_filtered)
            
        
        return user_representation
    
    def get_name(self) -> str:
        """
        Get the name of the model (used for pretty printing).

        :return: The name of the model
        """
        return "Content-Based Recommender Using BERT Embeddings"
    
    def predict_computability_between_user_and_item(self, user_id, item_id, aggregation_method):
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
        rating = self.predict_computability_between_user_and_item(user_id, item_id, self.aggregation_method)
        normalized = 1 + (rating - self.min_val) * (4 / (self.max_val - self.min_val))
        return normalized
    

    def get_cached_predicted_score(self, user_id: int, item_id: int) -> float:
        """
        Lookup precomputed score a user is predicted to give an item.
    
        :param user_id: Id of the user
        :param item_id: Id of the item
        :returns: Predicted score
        """
        return self.predictions.loc[((self.predictions['user_id'] == user_id) & (self.predictions['item_id'] == item_id)), 'predicted_score'].values[0]

    def calculate_all_predictions(self, train_data: pd.DataFrame) -> None:
        """
        Calculate and save all rating predictions (each user/item pair) in the training data.

        :param train_data: Training data containing user_ids and item_ids
        """
        user_ids = train_data['user_id'].unique()
        item_ids = train_data['item_id'].unique()
        pairs = list(itertools.product(user_ids, item_ids))
        predictions = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        predictions['predicted_score'] = predictions.apply(lambda x : self.predict_score(x['user_id'], x['item_id']), axis=1)
        self.predictions = predictions

    def calculate_all_rankings(self, k: int, train_data: pd.DataFrame) -> None:
        """
        For each user in the training data, calculate the predicted ranking and save it.

        :param k: Ranking list size
        :param train_data: Training data containing user ids
        """
        user_ids = train_data['user_id'].unique()
        self.rankings = {}
        for user_id in user_ids:
            user_df = self.predictions.loc[
                (self.predictions['user_id'] == user_id),
                ['item_id', 'predicted_score']
            ]

            top_k = (
                user_df.nlargest(k, 'predicted_score')
                .apply(lambda row: (row['item_id'], row['predicted_score']), axis=1)
                .tolist()
            )
            self.rankings[user_id] = top_k

    def get_ranking(self, user_id: int, k: int) -> List[tuple[int, float]]:
        """
        Lookup precomputed ranking for a user.

        :param user_id: Id of the user
        :param k: Maximum size of recommendation list
        :returns: List of pairs of item_ids and scores (ordered descending)
        """
        return self.rankings[user_id][:k]
    