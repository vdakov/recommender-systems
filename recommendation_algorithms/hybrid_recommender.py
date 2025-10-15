from typing import List
from recommendation_algorithms.abstract_recommender import AbstractRecommender
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import minimize


class HybridRecommender:
    recommenders: List[AbstractRecommender]
    weights: List[float]  # Each weight corresponds to a recommender
    predicted_scores = pd.DataFrame

    def __init__(self):
        # TODO fill list of recommenders
        self.recommenders = []
        self.weights = []
        pass 

    def train(self, training_path: str) -> None:
        # Loading training data
        columns_name = ['user_id','item_id','rating','timestamp']
        train_data = pd.read_csv(training_path, sep='\t', names=columns_name)

        # Training all models
        for recommender in self.recommenders:
            recommender.train(train_data)

        # Find weights which minimize MSE
        self.linear_regression(train_data)

        # Predict score for each u/i pair for each model, so aggregated dataframe can be created for ranking prediction
        user_ids = train_data['user_id'].unique()
        item_ids = train_data['item_id'].unique()
        pairs = list(itertools.product(user_ids, item_ids))
        self.predicted_scores = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        self.predicted_scores['predicted_score'] = self.predicted_scores.apply(lambda x : self.predict_score(x['user_id'], x['item_id'], self.weights), axis=1)

    def linear_regression(self, train_data: pd.DataFrame): 
        # Initial Weights
        ws = np.zeros(len(self.recommenders))

        # Define objective function (MSE)
        def rmse(weights: List[float]):
            y_pred = train_data.apply(lambda x : self.predict_score(x['user_id'], x['item_id'], weights)).to_list()
            y_true = train_data['rating'].to_list()
            errors = y_true - y_pred 

            # Calculate RMSE
            return np.dot(errors, errors) / len(errors)
        
        # Find weights that minimize objective function
        optimization_result = minimize(rmse, ws, method='L-BFGS-B')
        self.weights = optimization_result.x

    def predict_score(self, user_id: int, item_id: int, weights) -> float:
        return np.sum([p[0] * p[1].predict_score(user_id, item_id) for p in zip(self.recommenders, weights)])
     
    def predict_top_k(self, user_id: int, k: int) -> List[int]:
        user_df = self.predict_scores[self.predicted_scores['user_id'] == user_id]
        top_items = user_df.nlargest(k, 'predicted_score')['item_id']
        return top_items.tolist()
