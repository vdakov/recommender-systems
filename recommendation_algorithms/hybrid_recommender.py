from datetime import datetime
from typing import Dict, List, Optional, Sequence
from recommendation_algorithms.abstract_recommender import AbstractRecommender
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, ndcg_score
import os


class HybridRecommender:
    rating_recommenders: List[AbstractRecommender]
    ranking_recommenders: List[AbstractRecommender]
    rating_weights: List[float]  # Each weight corresponds to a recommender
    ranking_weights: List[float]  # Each weight corresponds to a recommender
    verbose: bool
    predictions: pd.DataFrame # Precomputed rating predictions for all user/item pairs
    precomputed_rankings: Dict[int, List[tuple[int, float]]] # List of precomputed ranking for all users
    max_k: int # Max length for ranking list

    def __init__(self, train_data: pd.DataFrame, rating_recommenders: List[AbstractRecommender], ranking_recommenders: List[AbstractRecommender], max_k: int, verbose=False):
        self.rating_recommenders = rating_recommenders
        self.ranking_recommenders = ranking_recommenders
        self.predictions = {} 
        self.rating_weights = []
        self.ranking_weights = []
        self.verbose = verbose
        self.max_k = max_k
        self.train(train_data)

    def train(self, train_data: pd.DataFrame) -> None: 
        # Training all individual models
        if self.verbose:
            print(f'Started training hybrid recommender on {len(train_data['user_id'].unique())} users and {len(train_data['item_id'].unique())} items...')
            print(f'Training individual models...')
        trained_recommenders = []
        for recommender in self.rating_recommenders:
            recommender.train(train_data)
            recommender.calculate_all_predictions(train_data) # Precomputing rating predictions
            trained_recommenders.append(recommender.get_name())
        for recommender in self.ranking_recommenders:
            if not recommender.get_name() in trained_recommenders:
                recommender.train(train_data)
                trained_recommenders.append(recommender.get_name())
            recommender.calculate_all_rankings(self.max_k, train_data) # Precomputing ranking predictions
        if self.verbose:
            print(f'Finished training individual models.')
            print('Started linear regression...')

        # Find weights which minimize objective function for both rating and ranking task
        self._linear_regression_rating(train_data, visualize=self.verbose)
        if self.verbose:
            print(f'Finished rating linear regression, weights are:')
            for i in range(len(self.rating_recommenders)):
                print(f'  {self.rating_recommenders[i].get_name()}: {self.rating_weights[i]}')
        self._linear_regression_ranking(self.max_k, train_data, visualize=self.verbose)
        if self.verbose:
            print(f'Finished ranking linear regression, weights are:')
            for i in range(len(self.ranking_recommenders)):
                print(f'  {self.ranking_recommenders[i].get_name()}: {self.ranking_weights[i]}')

        # Precompute all predictions for hybrid rater/ranker
        # Rating
        print(f"Precomputing predictions...")
        dfs = []
        for df, w in zip([r.predictions for r in self.rating_recommenders], self.rating_weights):
            temp = df.copy()
            temp['weighted_score'] = temp['predicted_score'] * w
            dfs.append(temp[['user_id', 'item_id', 'weighted_score']])
        combined = pd.concat(dfs, ignore_index=True)
        self.predictions = (
            combined.groupby(['user_id', 'item_id'], as_index=False, sort=False)
                    .agg(predicted_score=('weighted_score', 'sum'))
        )
        self.precomputed_rankings = {}
        # Ranking
        user_ids = train_data['user_id'].unique()
        for user_id in user_ids: 
            self.precomputed_rankings[user_id] = self._predict_ranking_with_weights(user_id, self.max_k, self.ranking_weights)

        print("Finished computing predictions, model is ready to use.")

    # Minimize objective function (MSE) to find weights for rating prediction
    def _linear_regression_rating(
        self,
        train_data: pd.DataFrame,
        # All params for visualization
        visualize: bool = False,
        save_dir: str = "plots",
        track_every: int = 1,              # record every k iterations
        max_snapshots: int = 250,          # cap history length
        track_dims: Optional[Sequence[int]] = None  # which weight indices to plot (None = all)
    ):
        ws0 = np.zeros(len(self.rating_recommenders), dtype=float)

        mse_history: List[float] = []
        weights_history: List[np.ndarray] = []

        # Store the latest f(w) from the most recent evaluation to avoid recomputation
        last_f = {'val': None}  # mutable enclosure for reuse in callback
        iter_count = {'k': 0}

        # Objective function to optimize (MSE)
        def objective(weights: List[float]):
            # Find weighted sum of predictions
            dfs = []
            for df, w in zip([r.predictions for r in self.rating_recommenders], weights):
                temp = df.copy()
                temp['weighted_score'] = temp['predicted_score'] * w
                dfs.append(temp[['user_id', 'item_id', 'weighted_score']])
            combined = pd.concat(dfs, ignore_index=True)
            result = (
                combined.groupby(['user_id', 'item_id'], as_index=False, sort=False)
                        .agg(predicted_score=('weighted_score', 'sum'))
            )
            aligned = ( # Align with GT so that predictions are same length
                train_data[['user_id', 'item_id']]
                .merge(result, on=['user_id', 'item_id'], how='left')
            )
            y_pred = aligned['predicted_score'].to_numpy()

            # Ground truth
            y_true = train_data['rating'].to_numpy()

            # Calculate MSE
            mse = mean_squared_error(y_true, y_pred)
            last_f['val'] = mse
            return mse

        # Callback used to keep track of MSE for visualization purposes
        def cb(wk: np.ndarray):
            if not visualize:
                return
            k = iter_count['k']
            if (k % track_every) == 0 and len(mse_history) < max_snapshots:
                # record the last objective value computed by SciPy
                mse_history.append(float(last_f['val']) if last_f['val'] is not None else np.nan)
                # optionally track only some dimensions to reduce memory/plot time
                if track_dims is None:
                    weights_history.append(wk.copy())
                else:
                    weights_history.append(wk[np.array(track_dims, dtype=int)].copy())
            iter_count['k'] = k + 1

        # Optimize
        res = minimize(objective, ws0, method='L-BFGS-B', callback=cb)
        self.rating_weights = res.x

        # --- Visualization & save (plot once) ---
        if visualize and len(mse_history) > 0:
            # Prepare weights matrix
            W = np.vstack(weights_history)  # shape: (T, D_tracked)
            # Labels for weight lines
            if track_dims is None:
                labels = [f"w{i}" for i in range(W.shape[1])]
            else:
                labels = [f"w{i}" for i in track_dims]

            # Plot once, then save
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            ax[0].plot(mse_history, marker='o')
            ax[0].set_title("MSE per Iteration (downsampled)")
            ax[0].set_xlabel("Logged iteration")
            ax[0].set_ylabel("MSE")

            for j in range(W.shape[1]):
                ax[1].plot(W[:, j], label=labels[j])
            ax[1].set_title("Tracked Weights over Iterations")
            ax[1].set_xlabel("Logged iteration")
            ax[1].set_ylabel("Weight value")
            if W.shape[1] <= 15:  # keep legend small; adjust as you like
                ax[1].legend(loc="best")

            plt.tight_layout()

            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = os.path.join(save_dir, f"linear_regression_rating_{timestamp}.png")
            plt.savefig(file_path, dpi=300)
            plt.close(fig)

            print(f"[INFO] Visualization saved to: {file_path}")

    # Minimize objective function (NDCG) to find weights for ranking prediction
    def _linear_regression_ranking(
        self,
        k: int,
        train_data: pd.DataFrame,
        # All params for visualization
        visualize: bool = False,
        save_dir: str = "plots",
        track_every: int = 1,              # record every k iterations
        max_snapshots: int = 250,          # cap history length
        track_dims: Optional[Sequence[int]] = None  # which weight indices to plot (None = all)
    ):
        ws0 = np.ones(len(self.ranking_recommenders), dtype=float) / len(self.ranking_recommenders)
        ndcg_history: List[float] = []
        weights_history: List[np.ndarray] = []

        # Store the latest f(w) from the most recent evaluation to avoid recomputation
        last_f = {'val': None}  # mutable enclosure for reuse in callback
        iter_count = {'k': 0}

        # Mean NDCG across users
        def NDCG(ground_truth, rec_list):
            result = 0.0
            
            # For each user calculate NDCG and take mean
            for gt_items, rec_items in zip(ground_truth, rec_list):
                K = len(rec_items)
                # Calculate DCG
                DCG = 0
                for i in range(K):
                    numerator = 1 if rec_items[i] in gt_items else 0 
                    denominator = np.log2(i + 2)
                    DCG += numerator / denominator
                # IDCG (ideal is all relevant items up to K)
                ideal_len = min(K, len(gt_items))
                if ideal_len == 0:
                    ndcg = 0.0
                else:
                    IDCG = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
                    ndcg = DCG / IDCG
                # Calculate NDCG
                result += ndcg
          
            result /= float(len(ground_truth))
            return result
        
        # Objective function to optimize (NDCG)
        def objective(weights: List[float]):
            user_ids = train_data['user_id'].unique()

            # Calculate mean NDCG across users
            preds = []
            gts = []
            for user_id in user_ids:
                # Find predicted ranking for each user    
                predicted_ranking = self._predict_ranking_with_weights(user_id, k, weights)
                item_ids = [item_id for item_id, _ in predicted_ranking]
                # Find ground truth ranking
                user_df = train_data[train_data['user_id'] == user_id]
                gt = (
                    user_df.nlargest(k, 'rating')
                )['item_id'].to_list()
                size = min(len(item_ids), len(gt))
                gt = gt[:size]
                item_ids = item_ids[:size]
                preds.append(item_ids)
                gts.append(gt)
            
            # Take mean ndcg
            ndcg = NDCG(gts, preds)
            last_f['val'] = ndcg
            return 1 - ndcg

        # Callback used to keep track of NDCG for visualization purposes
        def cb(wk: np.ndarray):
            if not visualize:
                return
            k = iter_count['k']
            if (k % track_every) == 0 and len(ndcg_history) < max_snapshots:
                # record the last objective value computed by SciPy
                ndcg_history.append(float(last_f['val']) if last_f['val'] is not None else np.nan)
                # optionally track only some dimensions to reduce memory/plot time
                if track_dims is None:
                    weights_history.append(wk.copy())
                else:
                    weights_history.append(wk[np.array(track_dims, dtype=int)].copy())
            iter_count['k'] = k + 1

        # Optimize
        res = minimize(objective, ws0, method='L-BFGS-B', callback=cb)
        self.ranking_weights = res.x

        # --- Visualization & save (plot once) ---
        if visualize and len(ndcg_history) > 0:
            # Prepare weights matrix
            W = np.vstack(weights_history)  # shape: (T, D_tracked)
            # Labels for weight lines
            if track_dims is None:
                labels = [f"w{i}" for i in range(W.shape[1])]
            else:
                labels = [f"w{i}" for i in track_dims]

            # Plot once, then save
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            ax[0].plot(ndcg_history, marker='o')
            ax[0].set_title("NDCG per Iteration (downsampled)")
            ax[0].set_xlabel("Logged iteration")
            ax[0].set_ylabel("NDCG")

            for j in range(W.shape[1]):
                ax[1].plot(W[:, j], label=labels[j])
            ax[1].set_title("Tracked Weights over Iterations")
            ax[1].set_xlabel("Logged iteration")
            ax[1].set_ylabel("Weight value")
            if W.shape[1] <= 15:  # keep legend small; adjust as you like
                ax[1].legend(loc="best")

            plt.tight_layout()

            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = os.path.join(save_dir, f"linear_regression_ranking_{timestamp}.png")
            plt.savefig(file_path, dpi=300)
            plt.close(fig)

            print(f"[INFO] Visualization saved to: {file_path}")

    # Predict a rating for an item with specific weights (used in weight optimization)
    def _predict_score_with_weights(self, user_id: int, item_id: int, weights: List[float]) -> float:
        return np.sum([p[0] * p[1].get_cached_predicted_score(user_id, item_id) for p in zip(weights, self.rating_recommenders)])

    # Predict a ranking for a user with specific weights (used in weight optimization)
    def _predict_ranking_with_weights(self, user_id: int, k: int, weights: List[float]) -> List[tuple[int, float]]:
        item_scores = {}
        for recommender, w in zip(self.ranking_recommenders, weights):
            ranking = recommender.get_ranking(user_id, k)
            for item_id, score in ranking:
                weighted_score = w * score 
                if item_id in item_scores:
                    item_scores[item_id] += weighted_score
                else:
                    item_scores[item_id] = weighted_score
        # sorted_ranking = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_ranking = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_ranking[:k]

    def predict_score(self, user_id: int, item_id: int) -> float:
        # Find precomputed prediction
        return self.predictions.loc[((self.predictions['item_id'] == item_id) & (self.predictions['user_id'] == user_id)), 'predicted_score'].values[0]
    
    def predict_ranking(self, user_id: int, k: int) -> List[int]:
        # Predict ranking based on precomputed scores
        if k > self.max_k:
            raise ValueError(f"Requested ranking length {k} is larger than the max length {self.max_k}")
        return self.precomputed_rankings[user_id][:k]
