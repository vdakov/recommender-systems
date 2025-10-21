from abc import ABC, abstractmethod
import itertools
from typing import Dict, List
import pandas as pd


class AbstractRecommender(ABC):
    """
    Abstract class to represent a recommendation model.
    All models in the hybrid recommender must extend this class and implement its methods.
    """
    predictions: pd.DataFrame

    @abstractmethod
    def train(self, train_data: pd.DataFrame) -> None:
        """
        This method prepares the model for usage, e.g. computing similarity matrices and training models.

        :param train_data: Dataframe containing training data, relevant keys: user_id, item_id, rating.
        """
        pass 

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the model (used for pretty printing).

        :return: The name of the model
        """
        pass

    @abstractmethod
    def predict_score(self, user_id: int, item_id: int) -> float:
        """
        Predict a score a user would give a specific item.

        :param user_id: The id of the user
        :param item_id: The id of the item
        :return: Predicted score
        """
        pass

    def get_cached_predicted_score(self, user_id: int, item_id: int) -> float:
        return self.predictions.loc[((self.predictions['user_id'] == user_id) & (self.predictions['item_id'] == item_id)), 'predicted_score'].values[0]

    def calculate_all_predictions(self, train_data: pd.DataFrame) -> None:
        user_ids = train_data['user_id'].unique()
        item_ids = train_data['item_id'].unique()
        pairs = list(itertools.product(user_ids, item_ids))
        predictions = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        predictions['predicted_score'] = predictions.apply(lambda x : self.predict_score(x['user_id'], x['item_id']), axis=1)
        self.predictions = predictions
