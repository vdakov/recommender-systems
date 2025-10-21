from recommendation_algorithms.abstract_recommender import AbstractRecommender
import pandas as pd


# This is a recommender that always gives a score as 5 for testing purposes
class FiveRecommender(AbstractRecommender):
    def train(self, train_data: pd.DataFrame) -> None:
        pass

    def get_name(self) -> str:
        """
        Get the name of the model (used for pretty printing).

        :return: The name of the model
        """
        return "5 stars"

    def predict_score(self, user_id: int, item_id: int) -> float:
        """
        Predict a score a user would give a specific item.

        :param user_id: The id of the user
        :param item_id: The id of the item
        :return: Predicted score
        """
        return 5.0
    