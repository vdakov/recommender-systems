from abc import ABC, abstractmethod
import pandas as pd


class AbstractRecommender(ABC):
    """
    Abstract class to represent a recommendation model.
    All models in the hybrid recommender must extend this class and implement its methods.
    """

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
