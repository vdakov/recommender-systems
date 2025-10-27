from abc import ABC, abstractmethod
import itertools
import os
from typing import Dict, List
import pandas as pd
import json
from tqdm import tqdm, tqdm_pandas


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
        tqdm.pandas()
        user_ids = train_data['user_id'].unique()
        item_ids = train_data['item_id'].unique()
        pairs = list(itertools.product(user_ids, item_ids))
        predictions = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        predictions['predicted_score'] = predictions.apply(lambda x : self.predict_score(x['user_id'], x['item_id']), axis=1)
        self.predictions = predictions
        self.normalize_predictions()

    def calculate_all_rankings(self, k: int, train_data: pd.DataFrame) -> None:
        """
        For each user in the training data, calculate the predicted ranking and save it.

        :param k: Ranking list size
        :param train_data: Training data containing user ids
        """
        tqdm.pandas()
        user_ids = train_data['user_id'].unique()
        self.rankings = {}
        for user_id in tqdm(user_ids):
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

    def _get_predictions_file_path(self, is_test: bool = False) -> str:
        """
        Get the file path for storing/loading precomputed predictions.

        :param is_test: Whether to use the test or train folder
        :return: File path as string
        """
        folder_path = os.path.join(f'model_checkpoints/{"test" if is_test else "train"}', self.get_name().replace(" ", "_").lower())
        filepath = os.path.join(folder_path, 'predictions.csv')
        os.makedirs(folder_path, exist_ok=True)
        return filepath

    def _get_ranking_predictions_file_path(self, is_test: bool = False) -> str:
        """
        Get the folder path to which rankings can be saved.

        :param is_test: Whether to use the test or train folder
        :return: File path as string
        """
        folder_path = os.path.join(f'model_checkpoints/{"test" if is_test else "train"}', self.get_name().replace(" ", "_").lower())
        filepath = os.path.join(folder_path, 'rankings')
        os.makedirs(filepath, exist_ok=True)
        return filepath

    def checkpoint_exists(self, is_test: bool = False) -> bool:
        """
        Check if a checkpoint file for predictions exists.

        :param is_test: Whether to check in the test or train folder
        :return: True if the checkpoint file exists, False otherwise
        """
        return os.path.isfile(self._get_predictions_file_path(is_test=is_test))

    def load_predictions_from_file(self, is_test: bool = False) -> None:
        """
        Load precomputed predictions from a CSV file.

        :param is_test: Whether to load from the test or train folder
        """
        if is_test:
            self.test_predictions = pd.read_csv(self._get_predictions_file_path(is_test=True))
            
        else:
            self.predictions = pd.read_csv(self._get_predictions_file_path(is_test=False))
        self.normalize_predictions()

    def load_ranking_from_file(self, user_id:int) -> None:
        """
        Load precomputed rankings from a CSV file.

        :param user_id: Id of the user
        """
        file_path = os.path.join(self._get_ranking_predictions_file_path(), f'user_{user_id}_ranking.csv')
        if not hasattr(self, "rankings") or self.rankings is None:
            self.rankings = {}
        self.rankings[user_id] = pd.read_csv(file_path)

    def load_all_rankings_from_file(self, train_data: pd.DataFrame):
        """
        Load precomputed rankings for all users in the training data from CSV files.
        :param train_data:
        :return:
        """
        for user in tqdm(train_data["user_id"].unique()):
            self.load_ranking_from_file(user)

    def save_predictions_to_file(self, is_test: bool = False) -> None:
        """
        Save precomputed predictions to a CSV file.

        :param is_test: Whether to save to the test or train folder
        """
        self.predictions.to_csv(self._get_predictions_file_path(is_test=is_test), index=False)


    def save_rankings_to_file(self, is_test: bool = False) -> None:
        """
        Save precomputed rankings to a set of CSV files and a JSON mapping file.

        :param is_test: Whether to save to the test or train folder
        """
        folder_path = self._get_ranking_predictions_file_path(is_test=is_test)
        user_dict = {}

        for user_id, ranking in self.rankings.items():
            ranking_df = pd.DataFrame(ranking, columns=['item_id', 'predicted_score'])
            filepath = os.path.join(folder_path, f'user_{user_id}_ranking.csv')
            ranking_df.to_csv(filepath, index=False)
            user_dict[int(user_id)] = filepath

        with open(os.path.join(folder_path, 'user_ranking_file_map.json'), 'w') as f:
            json.dump(user_dict, f, indent=4)

    def calculate_rating_predictions_test_data(self, test_data: pd.DataFrame) -> None:
        """
        Calculate all predictions and rankings for the test data.

        :param test_data: Test data containing user_ids and item_ids
        """
        self.calculate_all_predictions(test_data)
        self.save_predictions_to_file(is_test=True)
        print("Calculated and saved predictions for test data.")
        
    def normalize_predictions(self, min_rating: float = 1.0, max_rating: float = 5.0) -> None:
        """
        Linearly normalize all predicted scores to a valid range [min_rating, max_rating].

        :param min_rating: Minimum rating value (default=1.0)
        :param max_rating: Maximum rating value (default=5.0)
        """
        if not hasattr(self, "predictions") or self.predictions is None or self.predictions.empty:
            raise ValueError("No predictions available to normalize. Please run calculate_all_predictions() first.")

        preds = self.predictions["predicted_score"]
        min_pred, max_pred = preds.min(), preds.max()

        if min_pred == max_pred:
            # avoid division by zero if all predictions are identical
            self.predictions["predicted_score"] = min_rating + (max_rating - min_rating) / 2
            print("All predictions had the same value. Set to midpoint of normalization range.")
            return

        self.predictions["predicted_score"] = (
            min_rating + (preds - min_pred) * (max_rating - min_rating) / (max_pred - min_pred)
        )

        print(f"Predictions normalized linearly to range [{min_rating}, {max_rating}].")

    def calculate_ranking_predictions_test_data(self, test_data: pd.DataFrame, k: int) -> None:
        """
        Calculate all predictions and rankings for the test data.

        :param test_data: Test data containing user_ids and item_ids
        :param k: Ranking list size
        """
        self.calculate_all_rankings(k, test_data)
        self.save_rankings_to_file(is_test=True)
        print("Calculated and saved predictions and rankings for test data.")
        
    def _get_model_file_path(self, is_test: bool = False) -> str:
        """
        Get the file path for storing/loading precomputed predictions.

        :param is_test: Whether to use the test or train folder
        :return: File path as string
        """
        folder_path = os.path.join(f'model_checkpoints/{"test" if is_test else "train"}', self.get_name().replace(" ", "_").lower())
        filepath = os.path.join(folder_path, 'model')
        os.makedirs(filepath, exist_ok=True)
        return filepath
