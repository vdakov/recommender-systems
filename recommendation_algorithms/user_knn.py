import pandas as pd
import numpy as np

def user_similarity(user1_ratings, user2_ratings) -> float:
    pass

class UserKNN:

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, k: int = 5):
        """
        Initialize the UserKNN model with training and testing data.

        Parameters
        :param train_data: pd.DataFrame with columns ['user_id', 'item_id', 'rating']
        :param test_data: pd.DataFrame with columns ['user_id', 'item_id', 'rating']
        :param k: k value for number of nearest neighbors
        """
        self.train_data = train_data
        self.test_data = test_data
        self.user_ids = self.train_data['user_id'].unique()
        self.k = k
        self.user_means = self.train_data.groupby('user_id')['rating'].mean().to_dict()
        self.fit = False

    def set_k(self, k):
        self.k = k

    def compute_user_similarity_matrix(self):
        """
        Compute the user similarity matrix and store it in self.similarity_matrix.
        :return:
        """
        pass

    def user_similarity_train(self, user1, user2) -> float:
        """
        Compute user similarity based on training data. If matrix has already been computed, retrieve the value there.
        :param user1: user id
        :param user2: user id
        :return: similarity score as a float
        """
        pass

    def user_similarity_test(self, user1, user2) -> float:
        """
        Compute user similarity based on test data.
        :param user1:
        :param user2:
        :return:
        """
        pass

    def predict_rating(self, user_id, item_id) -> float:
        """
        Predict rating for a given user and item using UserKNN approach.
        :param user_id: target user id
        :param item_id: target item id
        """
        pass

    def predict_ranking(self, user_id, n: int = 10) -> list:
        """
        Predict the top-n recommended items for a given user using UserKNN approach.
        :param user_id: target user id
        :param n: number of recommendations to return
        :return: list of (item_id, predicted_score) sorted by score desc.
        """
        pass
