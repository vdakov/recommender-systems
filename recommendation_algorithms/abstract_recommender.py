from abc import ABC, abstractmethod
import itertools
import os
from typing import Dict, List
import pandas as pd
import json


class AbstractRecommender(ABC):
    """
    Abstract class to represent a recommendation model.
    All models in the hybrid recommender must extend this class and implement its methods.
    """
    predictions: pd.DataFrame
    rankings: Dict[int, List[tuple[int, float]]]

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
    
    def _get_predictions_file_path(self) -> str:
        """
        Get the file path for storing/loading precomputed predictions.

        :return: File path as string
        """
        folder_path = os.path.join('model_checkpoints', self.get_name().replace(" ", "_").lower())
        filepath = os.path.join(folder_path, 'predictions.csv')
        os.makedirs(folder_path, exist_ok=True)
        return filepath

    def _get_ranking_predictions_file_path(self) -> str:
        """
        Get the folder path to which rankings can be saved.

        :return: File path as string
        """
        folder_path = os.path.join('model_checkpoints', self.get_name().replace(" ", "_").lower())
        filepath = os.path.join(folder_path, 'rankings')
        os.makedirs(filepath, exist_ok=True)
        return filepath
    
    def checkpoint_exists(self) -> bool:
        """
        Check if a checkpoint file for predictions exists.

        :return: True if the checkpoint file exists, False otherwise
        """
        return os.path.isfile(self._get_predictions_file_path())

    def load_predictions_from_file(self) -> None:
        """
        Load precomputed predictions from a CSV file.

        :param filepath: Path to the CSV file containing predictions
        """
        self.predictions = pd.read_csv(self._get_predictions_file_path())

    def save_predictions_to_file(self) -> None:
        """
        Save precomputed predictions to a CSV file.

        :param filepath: Path to the CSV file to save predictions
        """
        self.predictions.to_csv(self._get_predictions_file_path(), index=False)


    def save_rankings_to_file(self) -> None:
        """
        Save precomputed rankings to a set of CSV files and a JSON mapping file.
        """
        folder_path = self._get_ranking_predictions_file_path()
        user_dict = {}

        for user_id, ranking in self.rankings.items():
            ranking_df = pd.DataFrame(ranking, columns=['item_id', 'predicted_score'])
            filepath = os.path.join(folder_path, f'user_{user_id}_ranking.csv')
            ranking_df.to_csv(filepath, index=False)
            user_dict[user_id] = filepath

        user_dict_json = json.dumps(user_dict, indent=4)
        with open(os.path.join(folder_path, 'user_ranking_file_map.json'), 'w') as f:
            f.write(user_dict_json)