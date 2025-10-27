import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools 
import os 
import pickle
from recommendation_algorithms.abstract_recommender import AbstractRecommender

class MatrixFactorizationSGD(AbstractRecommender):
    """
    Matrix Factorization for rating prediction using Stochastic Gradient Descent (SGD).

    This code is copied from Assignment 2 of the DSAIT4335 (Recommender Systems) course at TU Delft.

    Rating matrix R ≈ P × Q^T + biases
    """

    def __init__(self, n_factors=20, learning_rate=0.01, regularization=0.02, n_epochs=20, use_bias=True,):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.use_bias = use_bias

        # Model parameters
        self.P = None  # User latent factors
        self.Q = None  # Item latent factors
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None

    def get_name(self) -> str:
        return "Matrix Factorization"

    def train(self, train_data: pd.DataFrame):
        """
        Train the model.

        Args:
            ratings (pd.DataFrame): dataframe with [user_id, item_id, rating]
        """
        ratings = train_data
        # Map IDs to indices
        self.user_mapping = {u: i for i, u in enumerate(ratings['user_id'].unique())}
        self.item_mapping = {i: j for j, i in enumerate(ratings['item_id'].unique())}
        self.user_inv = {i: u for u, i in self.user_mapping.items()}
        self.item_inv = {j: i for i, j in self.item_mapping.items()}

        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        # Initialize factors
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

        if self.use_bias:
            self.user_bias = np.zeros(n_users)
            self.item_bias = np.zeros(n_items)
            self.global_mean = ratings['rating'].mean()

        # Convert to (user_idx, item_idx, rating) triples
        training_data = [(self.user_mapping[u], self.item_mapping[i], r)
                         for u, i, r in zip(ratings['user_id'], ratings['item_id'], ratings['rating'])]

        # SGD loop
        for _ in range(self.n_epochs):
            np.random.shuffle(training_data)
            total_error = 0

            for u, i, r in training_data:
                pred = np.dot(self.P[u], self.Q[i])
                if self.use_bias:
                    pred += self.global_mean + self.user_bias[u] + self.item_bias[i]

                err = r - pred
                total_error += err ** 2

                # Updates
                P_u = self.P[u]
                Q_i = self.Q[i]

                self.P[u] += self.learning_rate * (err * Q_i - self.regularization * P_u)
                self.Q[i] += self.learning_rate * (err * P_u - self.regularization * Q_i)

                if self.use_bias:
                    self.user_bias[u] += self.learning_rate * (err - self.regularization * self.user_bias[u])
                    self.item_bias[i] += self.learning_rate * (err - self.regularization * self.item_bias[i])

        return self

    def predict_score(self, user_id, item_id):
        """Predict rating for a single (user, item) pair"""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return np.nan

        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]

        pred = np.dot(self.P[u], self.Q[i])
        if self.use_bias:
            pred += self.global_mean + self.user_bias[u] + self.item_bias[i]
        return pred
    
    
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
        predictions['predicted_score'] = predictions.apply(lambda x : self.predict_score(x['user_id'], x['item_id']), axis=1)
        self.predictions = predictions

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
        if self.use_bias:
            scores += self.global_mean + self.user_bias[u] + self.item_bias

        # Exclude seen items
        if exclude_seen:
            seen_items = train_data[train_data['user_id'] == user_id]['item_id'].values
            seen_idx = [self.item_mapping[i] for i in seen_items if i in self.item_mapping]
            scores[seen_idx] = -np.inf

        # Get top-K items
        top_idx = np.argsort(scores)[::-1][:n]
        top_items = [self.item_inv[i] for i in top_idx]
        top_scores = scores[top_idx]

        return list(zip(top_items, top_scores))
    
    # Override as it works differently from the rating prediction rankers
    def calculate_all_rankings(self, k: int, train_data: pd.DataFrame) -> None:
        self.rankings = {}
        for user_id in train_data['user_id'].unique():
            ranking = self.recommend_topk(user_id, train_data, k)
            self.rankings[user_id] = ranking
            
    def save_model(self):
        folder_path = self._get_model_file_path()
        os.makedirs(folder_path, exist_ok=True)

        np.save(os.path.join(folder_path, "P.npy"), self.P)
        np.save(os.path.join(folder_path, "Q.npy"), self.Q)
        if self.use_bias:
            np.save(os.path.join(folder_path, "user_bias.npy"), self.user_bias)
            np.save(os.path.join(folder_path, "item_bias.npy"), self.item_bias)
            with open(os.path.join(folder_path, "global_mean.pkl"), "wb") as f:
                pickle.dump(self.global_mean, f)

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
                "use_bias": self.use_bias
            }, f)

        print(f"Model saved to {folder_path}")

    def load_model(self):
        folder_path = self._get_model_file_path()

        self.P = np.load(os.path.join(folder_path, "P.npy"))
        self.Q = np.load(os.path.join(folder_path, "Q.npy"))
        if self.use_bias:
            self.user_bias = np.load(os.path.join(folder_path, "user_bias.npy"))
            self.item_bias = np.load(os.path.join(folder_path, "item_bias.npy"))
            with open(os.path.join(folder_path, "global_mean.pkl"), "rb") as f:
                self.global_mean = pickle.load(f)

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
            self.use_bias = mappings["use_bias"]

        print(f"Model loaded from {folder_path}")