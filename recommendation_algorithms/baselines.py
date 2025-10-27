from typing import Dict, List
from recommendation_algorithms.abstract_recommender import AbstractRecommender
import pandas as pd
import numpy as np 
import random


class AverageRater(AbstractRecommender):
    train_data: pd.DataFrame

    def train(self, train_data: pd.DataFrame) -> None:
        self.train_data = train_data
        pass 

    def get_name(self) -> str:
        return "Average Item Rating Recommender"

    def predict_score(self, user_id: int, item_id: int) -> float:
        # Calculate the mean score for an item
        return np.mean(self.train_data.loc[(self.train_data['item_id'] == item_id), 'rating'])
    
class AverageHybridRater(AbstractRecommender):
    recommenders: List[AbstractRecommender]
    weights: List[float]  # Each weight corresponds to a recommender
    verbose: bool
    predictions: pd.DataFrame # Precomputed predictions for all user/item pairs

    def __init__(self, training_path: str, recommenders: List[AbstractRecommender], verbose=False):
        self.recommenders = recommenders
        self.predictions = {} # For each recommender, keep a dataframe of precompute 
        self.weights = []
        self.verbose = verbose
        columns_name = ['user_id','item_id','rating','timestamp']
        train_data = pd.read_csv(training_path, sep='\t', names=columns_name)
        self.train(train_data)

    def get_name(self) -> str:
        return "Average Component Rater"

    def train(self, train_data: pd.DataFrame) -> None:
        # Training all models
        for recommender in self.recommenders:
            recommender.train(train_data)
            recommender.calculate_all_predictions(train_data)

        self.weights = np.ones(len(self.recommenders))
        self.weights /= len(self.recommenders)

        # Precompute all predictions
        dfs = []
        for df, w in zip([r.predictions for r in self.recommenders], self.weights):
            temp = df.copy()
            temp['weighted_score'] = temp['predicted_score'] * w
            dfs.append(temp[['user_id', 'item_id', 'weighted_score']])
        combined = pd.concat(dfs, ignore_index=True)
        self.predictions = (
            combined.groupby(['user_id', 'item_id'], as_index=False, sort=False)
                    .agg(predicted_score=('weighted_score', 'sum'))
        )

    def predict_score(self, user_id: int, item_id: int) -> float:
        # Find precomputed prediction
        return self.predictions.loc[((self.predictions['item_id'] == item_id) & (self.predictions['user_id'] == user_id)), 'predicted_score'].values[0]
    
    def predict_ranking(self, user_id: int, k: int) -> List[int]:
        # Predict ranking based on precomputed scores
        user_df = self.predictions.loc[(self.predictions['user_id'] == user_id), ['item_id', 'predicted_score']]
        top_k = user_df.nlargest(k, 'predicted_score')
        return top_k['item_id'].to_list()

class RandomRanker(AbstractRecommender):
    unseen_items: Dict[int, List[int]] # For each user keep track of unseen items

    def __init__(self, train_data: pd.DataFrame):
        self.unseen_items = {}
        self.train(train_data)

    def get_name(self) -> str:
        return "Random Ranker"
    
    def train(self, train_data: pd.DataFrame) -> None:
        # Find unseen items for each user
        user_ids = train_data['user_id'].unique()
        item_ids = train_data['item_id'].unique()
        for user_id in user_ids:
            seen_items = train_data.loc[(train_data['user_id'] == user_id), 'item_id'].unique()
            unseen_items_for_user = [item_id for item_id in item_ids if item_id not in seen_items]
            self.unseen_items[user_id] = unseen_items_for_user

    def predict_score(self, user_id: int, item_id: int) -> float:
        return np.random.uniform(0, 5)
    
    def calculate_all_rankings(self, k: int, train_data: pd.DataFrame) -> None:
        self.rankings = {}
        for user_id in train_data['user_id'].unique():
            unseen_items = self.unseen_items[user_id]
            items_with_scores = [(item_id, self.predict_score(user_id, item_id)) for item_id in unseen_items]
            sorted_items = sorted(items_with_scores, key= lambda x : x[1], reverse=True)[:k]
            self.rankings[user_id] = sorted_items

class PopularRanker(AbstractRecommender):
    unseen_items: Dict[int, List[int]] # For each user keep track of unseen items
    popularities: Dict[int, int] # For each item keep track of amount of ratings 

    def __init__(self, train_data: pd.DataFrame):
        self.unseen_items = {}
        self.popularities = {}
        self.train(train_data)

    def get_name(self) -> str:
        return "Popularity Based Ranker"

    def train(self, train_data: pd.DataFrame) -> None:
        # Find unseen items for each user
        user_ids = train_data['user_id'].unique()
        item_ids = train_data['item_id'].unique()
        for user_id in user_ids:
            seen_items = train_data.loc[(train_data['user_id'] == user_id), 'item_id'].unique()
            unseen_items_for_user = [item_id for item_id in item_ids if item_id not in seen_items]
            self.unseen_items[user_id] = unseen_items_for_user
        
        # Find popularity of each item (amount of ratings)
        for item_id in item_ids:
            user_ratings = train_data.loc[
                (train_data['item_id'] == item_id),
                'user_id'
            ].unique()
            self.popularities[item_id] = len(user_ratings)

    def predict_score(self, user_id: int, item_id: int) -> float:
        raise ValueError("Predicting score not implemented for ranker")

    def predict_ranking(self, user_id: int, k: int) -> List[tuple[int, float]]:
        # Recommend most popular items that are not yet interacted by the target user. Most popular items are the ones that are rated by majority of users in the training data.
        unseen_items = self.unseen_items[user_id]
        def normalize_popularity(popularity: int) -> float:
            return popularity / max(self.popularities.values()) * 5.0  # Scale to rating range (1-5)
        items_with_popularity = [(item_id, normalize_popularity(self.popularities[item_id])) for item_id in unseen_items]
        sorted_items = sorted(items_with_popularity, key= lambda x : x[1], reverse=True)
        return sorted_items[:k]
    
    def calculate_all_rankings(self, k: int, train_data: pd.DataFrame) -> None:
        self.rankings = {}
        for user_id in train_data['user_id'].unique():
            ranking = self.predict_ranking(user_id, k)
            self.rankings[user_id] = ranking