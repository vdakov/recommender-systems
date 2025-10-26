import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine

from recommendation_algorithms.abstract_recommender import AbstractRecommender


# TODO: what if user in test not in train?

def item_similarity(item1_ratings, item2_ratings) -> float:
    """
    Compute cosine similarity between two items based on shared user ratings.

    :param item1_ratings: Series of ratings with index = user_id, name = 'rating'
    :param item2_ratings: Series of ratings with index = user_id, name = 'rating'
    :return: similarity score as a float. 1 means identical, 0 means no similarity.
    """
    merged = pd.merge(item1_ratings, item2_ratings, left_index=True, right_index=True, suffixes=('_1', '_2'), how='inner')
    if merged.empty:
        return 0.0
    sim = 1 - cosine(merged['rating_1'], merged['rating_2'])
    return sim


class ItemKNN(AbstractRecommender):

    def __init__(self, k: int = 5):
        """
        Initialize the ItemKNN model.

        Parameters
        :param k: k value for number of nearest neighbors
        """
        self.train_data = pd.DataFrame() # train_data
        self.test_data = pd.DataFrame() # test_data
        self.item_ids = []
        self.k = k
        self.similarity_matrix = pd.DataFrame()  # Empty so that it can be filled later
        self.fit = False

    def get_name(self) -> str:
        return 'Item KNN'

    def train(self, train_data: pd.DataFrame) -> None:  # TODO: may add another field containing sorted list for each item? should give speed-ups w/ continued use
        self.train_data = train_data
        self.item_ids = list(self.train_data['item_id'].unique())  # items in training data
        
        # Get similarity matrix
        self.similarity_matrix = pd.DataFrame(np.zeros((len(self.item_ids), len(self.item_ids))), index=self.item_ids,
                                              columns=self.item_ids)
        by_item = {iid: grp.set_index('user_id')['rating'] for iid, grp in train_data.groupby('item_id')}
        # Fill only the upper triangle and mirror it (half the work)
        for i, item1 in tqdm(enumerate(self.item_ids)):
            r1 = by_item[item1]
            for j in range(i + 1, len(self.item_ids)):
                item2 = self.item_ids[j]
                r2 = by_item[item2]
                s = item_similarity(r1, r2)
                self.similarity_matrix.loc[item1, item2] = s
                self.similarity_matrix.loc[item2, item1] = s
        self.fit = True
        
    def restore_training(self, train_data, similarity_matrix):
        self.train_data = train_data
        self.item_ids = list(self.train_data['item_id'].unique())  # items in training data
        self.similarity_matrix = similarity_matrix
        self.fit = True 

    def get_k_neighbors(self, target_item, similarity_matrix: pd.DataFrame = None) -> pd.Series:  # TODO: may add a cache for speed (discuss)
        """
        Get top-k most similar items to target item.

        Parameters:
        :param target_item: target item id
        :param similarity_matrix: masked similarity matrix to use. If None, use self.similarity_matrix.

        Returns:
        :return: List of tuples: [(neighbor_item_id, similarity), ...]
        """
        if similarity_matrix is None:
            similarity_matrix = self.similarity_matrix
        if target_item not in self.similarity_matrix.index:
            return pd.Series()

        sims = similarity_matrix[target_item].drop(target_item, errors='ignore').sort_values(ascending=False).head(self.k)
        return sims

    def predict_score(self, target_user, target_item) -> float:
        """
        Predict rating for a given user and item using k-NN.

        r*_u,i = (Sum_(j in N_i) (s_i,j * r_u,j)) / (Sum_(j in N_i) |s_i,j|)

        N_i = set of top-k most similar items to target item i that target user u has rated
        s_i,j = similarity between target item i and item j ([sims])
        r_u,j = rating of target user u for item j

        Parameters:
        :param target_user: user id
        :param target_item: item id

        Returns:
        :return: predicted rating (float)
        """
        if not self.fit:
            raise Exception("Model not trained yet. Call train() before predict_score().")

        user_u = self.train_data[self.train_data['user_id'] == target_user].set_index('item_id')
        items_rated_by_u = user_u.index.values.tolist()

        item_sim_matrix_masked = self.similarity_matrix.loc[items_rated_by_u]  # [items_rated_by_u]
        sims = self.get_k_neighbors(target_item,
                                    item_sim_matrix_masked)  # Series indexed by item_id, values are similarities
        if sims.empty:
            return 3.0  # TODO: return mean score for item / of user

        ni = sims.index.values  # N_i
        ruj = pd.Series([user_u.at[n, 'rating'] for n in ni], index=ni)  # ratings of target user for items in ni

        numerator = (sims * ruj).sum()
        denominator = sims.abs().sum()

        return numerator / denominator if denominator != 0 else 0.0
