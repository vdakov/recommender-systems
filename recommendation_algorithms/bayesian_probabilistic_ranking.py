from typing import Dict, List
import pandas as pd
import numpy as np
from recommendation_algorithms.abstract_recommender import AbstractRecommender
from tqdm import tqdm 

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
        self.use_bias = False # TODO what is this?

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
        
        full_index = pd.MultiIndex.from_product([unique_items, unique_users], names=["user_id", "item_id"])
        full_df = pd.DataFrame(index=full_index).reset_index()
        expanded = full_df.merge(ratings, on=["user_id", "item_id"], how="left").fillna({"rating":0})

        return expanded

    def train(self, train_data: pd.DataFrame):
        """
        Train the model.

        Args:
            ratings (pd.DataFrame): dataframe with [user_id, item_id, rating] 
        """
        ratings = train_data["rating"]
        ratings = self.expand_dataframe(train_data)
        
        self.user_mapping = {u: i for i, u in enumerate(ratings['user_id'].unique())}
        self.item_mapping = {i: j for j, i in enumerate(ratings['item_id'].unique())}
        self.user_inv = {i: u for u, i in self.user_mapping.items()}
        self.item_inv = {j: i for i, j in self.item_mapping.items()}
        self.user_item_rating = {(u, i): r for u, i, r in zip(ratings['user_id'], ratings['item_id'], ratings['rating'])}

        n_users = max(len(ratings["user_id"].unique()), ratings["user_id"].max())
        n_items = max(len(ratings["item_id"].unique()), ratings["item_id"].max())

        # Initialize factors
        self.P = np.random.normal(0, 1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 1, (n_items, self.n_factors))

        # SGD loop
        for _ in tqdm(range(self.n_epochs)):
            total_error = 0
            
            # Sampling is done via bootstrapping
            for _ in tqdm(range(self.num_samples_per_epoch)):
                u = np.random.choice(ratings["user_id"].unique())
                i = np.random.choice(ratings["item_id"].unique())
                j = np.random.choice(ratings["item_id"].unique())
                r_ui, r_uj = self.user_item_rating[(u, i)], self.user_item_rating[(u, j)]
                
                x_ui = np.dot(self.P[u - 1], self.Q[i - 1])
                x_uj = np.dot(self.P[u - 1], self.Q[j - 1])
                
                x_uij_hat = x_ui - x_uj
                x_uij = r_ui - r_uj
                
                P_u = self.P[u - 1]
                Q_i = self.Q[i - 1]
                Q_j = self.Q[j - 1]
                
                partial_P_u = Q_i - Q_j
                partial_Q_i = P_u
                partial_Q_j = -1 * P_u
                partial_P_u_prior = 2 * np.exp(np.square(P_u)) * P_u * (1 / np.sqrt(2 * np.pi))
                partial_Q_i_prior = 2 * np.exp(np.square(Q_i)) * Q_i * (1 / np.sqrt(2 * np.pi))
                partial_Q_j_prior = 2 * np.exp(np.square(Q_j)) * Q_j * (1 / np.sqrt(2 * np.pi))

                self.P[u - 1] += self.learning_rate * (partial_P_u + partial_P_u_prior)
                self.Q[i - 1] += self.learning_rate * (partial_Q_i + partial_Q_i_prior)
                self.Q[j - 1] += self.learning_rate * (partial_Q_j + partial_Q_j_prior)

        return self

    def predict_score(self, user_id: int, item_id: int) -> float:
        """Predict rating for a single (user, item) pair"""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return np.nan

        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]

        pred = np.dot(self.P[u], self.Q[i])
        if self.use_bias:
            pred += self.global_mean + self.user_bias[u] + self.item_bias[i]
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
            preds.append(self.predict_single(u, i))
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
        if self.use_bias:
            scores += self.global_mean + self.user_bias[u] + self.item_bias

        # Exclude seen items
        if exclude_seen:
            seen_items = train_data[train_data['user_id'] == user_id]['item_id'].values
            seen_idx = [self.item_mapping[i] for i in seen_items if i in self.item_mapping]
            scores[seen_idx] = -np.inf

        # Get top-K items
        top_idx = np.argsort(scores)[::-1][:n]
        top_items = [int(self.item_inv[i]) for i in top_idx]
        top_scores = scores[top_idx]

        return list(zip(top_items, top_scores))
    
    # Override as it works differently from the rating prediction rankers
    def calculate_all_rankings(self, k: int, train_data: pd.DataFrame) -> None:
        for user_id in train_data['user_id'].unique():
            ranking = self.recommend_topk(user_id, train_data, k)
            self.rankings[user_id] = ranking
