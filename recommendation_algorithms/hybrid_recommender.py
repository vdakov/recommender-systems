from typing import List
from recommendation_algorithms.abstract_recommender import AbstractRecommender
from recommendation_algorithms.matrix_factorization import MatrixFactorizationSGD
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import minimize


class HybridRecommender:
    recommenders: List[AbstractRecommender]
    weights: List[float]  # Each weight corresponds to a recommender
    predicted_scores = pd.DataFrame
    verbose: bool

    def __init__(self, training_path: str, verbose=False):
        # TODO fill list of recommenders
        matrix_factorization = MatrixFactorizationSGD()
        self.recommenders = [matrix_factorization]
        self.weights = []
        self.verbose = verbose
        self.train(training_path)

    def train(self, training_path: str) -> None:
        # Loading training data
        columns_name = ['user_id','item_id','rating','timestamp']
        train_data = pd.read_csv(training_path, sep='\t', names=columns_name)
        
        # Training all models
        if self.verbose:
            print(f'Started training hybrid recommender on {len(train_data['user_id'].unique())} users and {len(train_data['item_id'].unique())} items...')
            print(f'Training individual models...')
        for recommender in self.recommenders:
            recommender.train(train_data)
        if self.verbose:
            print(f'Finished training individual models.')
            print('Started linear regression...')

        # Find weights which minimize MSE
        self.linear_regression(train_data)
        if self.verbose:
            print(f'Finished linear regression, weights are:')
            for i in range(len(self.recommenders)):
                print(f'  {self.recommenders[i].get_name()}: {self.weights[i]}')
            print(f'Filling prediction dataframe...')
        # Predict score for each u/i pair for each model, so aggregated dataframe can be created for ranking prediction
        user_ids = train_data['user_id'].unique()
        item_ids = train_data['item_id'].unique()
        pairs = list(itertools.product(user_ids, item_ids))
        self.predicted_scores = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        
        self.predicted_scores['predicted_score'] = self.predicted_scores.apply(lambda x : self._predict_score_with_weights(x['user_id'], x['item_id'], self.weights), axis=1)
        if self.verbose:
            print('Finished prediction dataframe, hybrid model is ready to use!')

    def linear_regression(self, train_data: pd.DataFrame): 
        # Initial Weights
        ws = np.zeros(len(self.recommenders))

        # Define objective function (MSE)
        def mse(weights: List[float]):
            y_pred = train_data.apply(lambda x : self._predict_score_with_weights(x['user_id'], x['item_id'], weights), axis=1).to_numpy()
            y_true = train_data['rating'].to_numpy()
            errors = y_true - y_pred 

            # Calculate RMSE
            return np.dot(errors, errors) / len(errors)
        
        # Find weights that minimize objective function
        optimization_result = minimize(mse, ws, method='L-BFGS-B')
        self.weights = optimization_result.x

    def predict_score(self, user_id: int, item_id: int) -> float:
        df = self.predicted_scores
        return df.loc[((df['user_id'] == user_id) & (df['item_id'] == item_id)), 'predicted_score'].values[0]
    
    def _predict_score_with_weights(self, user_id: int, item_id: int, weights: List[float]) -> float:
        return np.sum([p[0] * p[1].predict_score(user_id, item_id) for p in zip(weights, self.recommenders)])
        
    def predict_top_k(self, user_id: int, k: int) -> List[int]:
        user_df = self.predict_scores[self.predicted_scores['user_id'] == user_id]
        top_items = user_df.nlargest(k, 'predicted_score')['item_id']
        return top_items.tolist()
