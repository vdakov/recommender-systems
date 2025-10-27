import itertools
from typing import Dict, List
import pandas as pd
import numpy as np
from recommendation_algorithms.abstract_recommender import AbstractRecommender
from tqdm import tqdm 
import os 
import pickle

class BayesianProbabilisticRanking(AbstractRecommender):
    rankings: Dict[int, List[tuple[int, float]]]
    
    """
    BPR implementation of the MF algorithm.
    """
    #def __init__(self, n_factors=20, learning_rate=0.01, regularization=0.02, n_epochs=20, num_samples_per_epoch=1000):
    def __init__(self, n_factors=20, learning_rate=0.01, regularization=0.02, n_epochs=5, num_samples_per_epoch=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.losses = []
        self.rankings = {}
        self.num_samples_per_epoch = num_samples_per_epoch

        # Model parameters
        self.P = None  # User latent factors
        self.Q = None  # Item latent factors
        self.item_bias = None
        self.global_mean = None

    def get_name(self) -> str:
        return "Bayesian Probabilistic Ranking"
    
    def expand_dataframe(self, ratings: pd.DataFrame) -> pd.DataFrame: 
        '''
        Expand dataframe to include the implicit assumption of BPR. Every user 
        needs to have a rating for every item. If a user has not rated an item, we assume the rating is zero. 
        Otherwisee, it is kept. 
        
        Args:
            ratings (pd.DataFrame): dataframe with [user_id, item_id, rating] 
        
        '''
        unique_items = ratings["item_id"].unique()
        unique_users = ratings["user_id"].unique()
        
        full_index = pd.MultiIndex.from_product([unique_items, unique_users], names=["item_id", "user_id"])
        full_df = pd.DataFrame(index=full_index).reset_index()
        expanded = full_df.merge(ratings, on=["user_id", "item_id"], how="left").fillna({"rating":0})

        return expanded

    def train(self, train_data: pd.DataFrame):
        """
        Train the model.

        Args:
            ratings (pd.DataFrame): dataframe with [user_id, item_id, rating] 
        """
        self.train_data = train_data
        ratings = self.expand_dataframe(train_data)
        
        self.user_mapping = {u: i for i, u in enumerate(ratings['user_id'].unique())}
        self.item_mapping = {i: j for j, i in enumerate(ratings['item_id'].unique())}
        self.user_inv = {i: u for u, i in self.user_mapping.items()}
        self.item_inv = {j: i for i, j in self.item_mapping.items()}
        self.user_item_rating = {(u, i): r for u, i, r in zip(ratings['user_id'], ratings['item_id'], ratings['rating'])}

        n_users = len(ratings["user_id"].unique())
        n_items = len(ratings["item_id"].unique())

        # Initialize factors
        self.P = np.random.normal(0, 1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 1, (n_items, self.n_factors))

        # SGD loop
        for _ in tqdm(range(self.n_epochs)):
            # Sampling is done via bootstrapping
            for _ in tqdm(range(self.num_samples_per_epoch)):
                u = np.random.choice(ratings["user_id"].unique())
                i = np.random.choice(ratings["item_id"].unique())
                j = np.random.choice(ratings["item_id"].unique())
                r_ui, r_uj = self.user_item_rating[(u, i)], self.user_item_rating[(u, j)]
                
                x_ui = np.dot(self.P[self.user_mapping[u]], self.Q[self.item_mapping[i]])
                x_uj = np.dot(self.P[self.user_mapping[u]], self.Q[self.item_mapping[j]])
                
                x_uij_hat = x_ui - x_uj
                x_uij = r_ui - r_uj
                
                P_u = self.P[self.user_mapping[u]]
                Q_i = self.Q[self.item_mapping[i]]
                Q_j = self.Q[self.item_mapping[j]]
                
                partial_P_u = Q_i - Q_j
                partial_Q_i = P_u
                partial_Q_j = -1 * P_u
                #differentiated terms - lambda is sigma here
                partial_P_u_prior = -self.regularization * P_u
                partial_Q_i_prior = -self.regularization * Q_i
                partial_Q_j_prior = -self.regularization * Q_j


                self.P[self.user_mapping[u]] += self.learning_rate * (partial_P_u + partial_P_u_prior)
                self.Q[self.item_mapping[i]] += self.learning_rate * (partial_Q_i + partial_Q_i_prior)
                self.Q[self.item_mapping[j]] += self.learning_rate * (partial_Q_j + partial_Q_j_prior)

        return self
    
        
    def calculate_all_predictions(self, train_data: pd.DataFrame) -> None:
        """
        Calculate and save all rating predictions (each user/item pair) in the training data.

        :param train_data: Training data containing user_ids and item_ids
        """
        tqdm.pandas()
        user_ids = train_data['user_id'].unique()
        item_ids = train_data['item_id'].unique()
       
        pairs = list(itertools.product(user_ids, item_ids))
        predictions = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        def predict_score_unnormalized(user_id, item_id):
            u = self.user_mapping[user_id]
            i = self.item_mapping[item_id]


            pred = np.dot(self.P[u], self.Q[i])
            return pred
        
        predictions['predicted_score'] = predictions.apply(lambda x : predict_score_unnormalized(x['user_id'], x['item_id']), axis=1)
        self.min_val = predictions['predicted_score'].min()
        self.max_val = predictions['predicted_score'].max()
        predictions['predicted_score'] = 1 + 4 * (
                    (predictions['predicted_score'] - self.min_val)
                    / (self.max_val - self.min_val)
                )
        self.predictions = predictions

    def predict_score(self, user_id: int, item_id: int) -> float:
        """Predict rating for a single (user, item) pair"""
        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]

        pred = np.dot(self.P[u], self.Q[i])
        pred = 1 + 4 * (pred - self.min_val) / (self.max_val - self.min_val)
        return pred
    
    def predict_order_two_items(self, user, item_i, item_j):
        x_ui = np.dot(self.P[user - 1], self.Q[item_i - 1])
        x_uj = np.dot(self.P[user - 1], self.Q[item_j - 1])
        x_uij_hat = x_ui - x_uj
        return np.signum(x_uij_hat)
    
    def predict(self, test_data):
        """Predict ratings for a test dataframe with [user_id, item_id]"""
        preds = []
        for u, i in zip(test_data['user_id'], test_data['item_id']):
            preds.append(self.predict_score(u, i))
        return np.array(preds)

    def recommend_topk(self, user_id, train_data, n=10, exclude_seen=True):
        """
        Generate Top-K recommendations for a given user.

        Args:
            user_id (int): target user ID (original ID, not index).
            train_data (pd.DataFrame): training ratings [user_id, item_id, rating],
                                       used to exclude already-seen items.
            k (int): number of recommendations.
            exclude_seen (bool): whether to exclude items the user already rated.

        Returns:
            list of (item_id, predicted_score) sorted by score desc.
        """
        if user_id not in self.user_mapping:
            return []

        u = self.user_mapping[user_id]

        # Predict scores for all items
        scores = np.dot(self.P[u], self.Q.T)

        # Exclude seen items
        if exclude_seen:
            seen_items = self.train_data[self.train_data['user_id'] == user_id]['item_id'].values
            seen_idx = [self.item_mapping[i] for i in seen_items if i in self.item_mapping]
            scores[seen_idx] = -np.inf

        # Get top-K items
        
        top_idx = np.argsort(scores)[::-1][:n]
        top_items = [int(self.item_inv[i]) for i in top_idx]
        top_scores = scores[top_idx]

        return list(zip(top_items, top_scores))
    
    # Override as it works differently from the rating prediction rankers
    def calculate_all_rankings(self, k: int, train_data: pd.DataFrame) -> None:
        self.rankings = {}
        for user_id in train_data['user_id'].unique():
            ranking = self.recommend_topk(user_id, train_data, k, exclude_seen=False)
            self.rankings[user_id] = ranking
            
    def save_model(self):
        folder_path = self._get_model_file_path()
        os.makedirs(folder_path, exist_ok=True)

        np.save(os.path.join(folder_path, "P.npy"), self.P)
        np.save(os.path.join(folder_path, "Q.npy"), self.Q)

        # Save mappings and config
        with open(os.path.join(folder_path, "mappings.pkl"), "wb") as f:
            pickle.dump({
                "user_mapping": self.user_mapping,
                "item_mapping": self.item_mapping,
                "user_inv": self.user_inv,
                "item_inv": self.item_inv,
                "n_factors": self.n_factors,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "n_epochs": self.n_epochs,
            }, f)

        print(f"Model saved to {folder_path}")

    def load_model(self):
        folder_path = self._get_model_file_path()

        self.P = np.load(os.path.join(folder_path, "P.npy"))
        self.Q = np.load(os.path.join(folder_path, "Q.npy"))

        with open(os.path.join(folder_path, "mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
            self.user_mapping = mappings["user_mapping"]
            self.item_mapping = mappings["item_mapping"]
            self.user_inv = mappings["user_inv"]
            self.item_inv = mappings["item_inv"]
            self.n_factors = mappings["n_factors"]
            self.learning_rate = mappings["learning_rate"]
            self.regularization = mappings["regularization"]
            self.n_epochs = mappings["n_epochs"]

        print(f"Model loaded from {folder_path}")
