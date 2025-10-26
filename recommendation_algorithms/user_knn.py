import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import correlation

from recommendation_algorithms.abstract_recommender import AbstractRecommender


def user_similarity(user1_ratings, user2_ratings) -> float:
    """
    Compute Pearson correlation similarity between two users based on shared item ratings.
    :param user1_ratings: Series of ratings with index = item_id
    :param user2_ratings: Series of ratings with index = item_id
    :return:
    """
    merged = pd.merge(user1_ratings, user2_ratings, left_index=True, right_index=True, suffixes=('_1', '_2'),
                      how='inner')
    if merged.empty:
        return 0.0

    u1 = merged['rating_1']
    u2 = merged['rating_2']
    mx = np.mean(u1)
    my = np.mean(u2)
    numerator = np.sum((u1 - mx) * (u2 - my))
    denominator = np.sqrt(np.sum((u1 - mx) ** 2)) * np.sqrt(np.sum((u2 - my) ** 2))

    return numerator / denominator if denominator != 0 else 0.0

class UserKNN(AbstractRecommender):

    def __init__(self, k: int = 5):
        """
        Initialize the UserKNN model with training and testing data.

        Parameters
        :param k: k value for number of nearest neighbors
        """
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.user_ids = []
        self.k = k
        self.similarity_matrix = pd.DataFrame()
        self.user_means = pd.Series()
        self.fit = False

    def get_name(self) -> str:
        return 'User KNN'

    def train(self, train_data: pd.DataFrame) -> None:
        self.train_data = train_data
        self.user_ids = list(self.train_data['user_id'].unique())
        self.user_means = self.train_data.groupby('user_id')['rating'].mean()

        self.similarity_matrix = pd.DataFrame(np.zeros((len(self.user_ids), len(self.user_ids))), index=self.user_ids, columns=self.user_ids)
        by_user = {uid: grp.set_index('item_id')['rating'] for uid, grp in train_data.groupby('user_id')}

        for i, user1 in tqdm(enumerate(self.user_ids)):
            r1 = by_user[user1]
            for j in range(i + 1, len(self.user_ids)):
                user2 = self.user_ids[j]
                r2 = by_user[user2]
                s = user_similarity(r1, r2)
                self.similarity_matrix.loc[user1, user2] = s
                self.similarity_matrix.loc[user2, user1] = s  # mirror
        self.fit = True
        
    def restore_training(self, train_data, similarity_matrix):
        self.train_data = train_data
        self.user_ids = list(self.train_data['user_id'].unique())
        self.user_means = self.train_data.groupby('user_id')['rating'].mean()

        self.similarity_matrix = similarity_matrix
        self.fit = True

    def get_k_neighbors(self, target_user, similarity_matrix: pd.DataFrame = None) -> pd.Series:
        if similarity_matrix is None:
            similarity_matrix = self.similarity_matrix
        if target_user not in self.similarity_matrix.index:
            return pd.Series()

        sims = similarity_matrix[target_user].drop(target_user, errors='ignore').sort_values(ascending=False).head(
            self.k)
        return sims

    def predict_score(self, target_user, target_item) -> float:
        """
        Predict rating for a given user and item using UserKNN approach.

        r*_u,i = r_u + Σ (sim(u,v) * (r_v,i - r_v)) / Σ |sim(u,v)|

        where:
        - r*_u,i is the predicted rating for user u on item i
        - r_u - mean rating of target user u
        - sim(u,v) - similarity between target user u and neighbor user v
        - r_v,i - rating of neighbor user v on item i
        - r_v - mean rating of neighbor user v

        :param target_user: target user id
        :param target_item: target item id
        """
        if not self.fit:
            raise Exception("Model not trained yet. Call train() before predict_score().")

        item_i = self.train_data[self.train_data['item_id'] == target_item].set_index('user_id')
        users_that_rated_i = item_i.index.values.tolist()
        # print(f'users_that_rated_i: \n{users_that_rated_i}')

        sim_matrix_masked = self.similarity_matrix.loc[users_that_rated_i]
        # print(f'sim_matrix_masked: \n{sim_matrix_masked}')
        sims = self.get_k_neighbors(target_item, sim_matrix_masked)
        # print(f'sims: \n{sims}')

        if sims.empty:
            return 0.0 # TODO: return mean score for item / of user - design choice

        ni = sims.index.values
        # print(f'ni \n{ni}')
        rvi = np.array([item_i.at[n, 'rating'] for n in ni])
        # print(f'rvi \n{rvi}')
        rv = np.array([self.user_means.loc[n] for n in ni])
        # print(f'rv \n{rv}')

        numerator = np.sum(sims.values * (rvi - rv))
        denominator = np.sum(np.abs(sims.values))

        return self.user_means.loc[target_user] + (numerator / denominator) \
            if denominator != 0 else self.user_means.loc[target_user]

    def predict_ranking(self, user_id, n: int = 10) -> list:
        """
        Predict the top-n recommended items for a given user using UserKNN approach.
        :param user_id: target user id
        :param n: number of recommendations to return
        :return: list of (item_id, predicted_score) sorted by score desc.
        """
        pass
