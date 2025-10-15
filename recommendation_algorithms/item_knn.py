import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
from torchgen.native_function_generation import self_to_out_signature

# TODO: what if user in test not in train?

def item_similarity(item1_ratings, item2_ratings) -> float:
    """
    Coompute cosine similarity between two items based on shared user ratings.

    :param item1_ratings:
    :param item2_ratings:
    :return:
    """
    merged = pd.merge(item1_ratings, item2_ratings, on='user_id', suffixes=('_1', '_2'), how='inner')
    if merged.empty:
        return 0.0
    sim = dist.cosine(merged['rating_1'], merged['rating_2'])  # TODO: check
    return sim


class ItemKNN:

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, k: int = 5):
        """
        Initialize the ItemKNN model with training and testing data.

        Parameters
        :param train_data: pd.DataFrame with columns ['user_id', 'item_id', 'rating']
        :param test_data: pd.DataFrame with columns ['user_id', 'item_id', 'rating']
        :param k: k value for number of nearest neighbors
        """
        self.train_data = train_data
        self.test_data = test_data
        self.item_ids = self.train_data['item_id'].unique()
        self.k = k
        self.similarity_matrix = pd.DataFrame()  # Empty so that it can be filled later
        self.fit = False

    def compute_item_similarity_matrix(self):
        self.similarity_matrix = pd.DataFrame(np.zeros((len(self.item_ids), len(self.item_ids))), index=self.item_ids, columns=self.item_ids)
        for i in range(len(self.item_ids)):
            for j in range(i+1, len(self.item_ids)):
                sim = self.item_similarity_train(self.item_ids[i], self.item_ids[j])
                self.similarity_matrix.at[self.item_ids[i], self.item_ids[j]] = sim
                self.similarity_matrix.at[self.item_ids[j], self.item_ids[i]] = sim
        self.fit = True

    def item_similarity_train(self, item1, item2) -> float:
        """
        Compute item similarity based on training data. If matrix has already been computed, retrieve the value there.
        """
        if self.fit:
            return self.similarity_matrix.at[item1, item2]

        i1_ratings = self.train_data[self.train_data['item_id'] == item1][['user_id', 'rating']]
        i2_ratings = self.train_data[self.train_data['item_id'] == item2][['user_id', 'rating']]
        return item_similarity(i1_ratings, i2_ratings)

    def item_similarity_test(self, item1, item2) -> float:
        """
        Compute item similarity based on test data.
        :param item1:
        :param item2:
        :return: similarity score as a float
        """
        i1_ratings = self.test_data[self.train_data['item_id'] == item1][['user_id', 'rating']]
        i2_ratings = self.test_data[self.train_data['item_id'] == item2][['user_id', 'rating']]
        return item_similarity(i1_ratings, i2_ratings)


    def set_k(self, k):
        self.k = k

    def get_k_neighbors(self, item_id) -> list:  # TODO: may add k as parameter
        """
        Get top-k most similar items to target item.

        Parameters:
        :param item_id: target item id

        Returns:
        :return: List of tuples: [(neighbor_item_id, similarity), ...]
        """
        if item_id not in self.similarity_matrix.index:
            return []
        sims = self.similarity_matrix.loc[item_id].sort_values(ascending=False)[:self.k]
        top_k_items = sims.index.tolist()
        return list(zip(top_k_items, sims))

    def predict_rating(self, target_user, target_item) -> float:
        """
        Predict rating for a given user and item using k-NN.

        Parameters:
        :param target_user: user id
        :param target_item: item id

        Returns:
        :return: predicted rating (float)
        """
        # TODO

    def predict_ranking(self, target_user, n: int = 10) -> list:
        """
        Predict top-n item recommendations for a given user.

        Parameters:
        :param target_user: user id
        :param n: number of recommendations

        Returns:
        :return: List of tuples: [(item_id, predicted_rating), ...]
        """
        # TODO
