from datetime import datetime
from typing import List, Optional, Sequence
from recommendation_algorithms.abstract_recommender import AbstractRecommender
from recommendation_algorithms.matrix_factorization import MatrixFactorizationSGD
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

class HybridRecommender:
    recommenders: List[AbstractRecommender]
    weights: List[float]  # Each weight corresponds to a recommender
    predicted_scores = pd.DataFrame
    verbose: bool

    def __init__(self, training_path: str, verbose=False):
        # TODO fill list of recommenders
        matrix_factorization = MatrixFactorizationSGD()
        self.recommenders = [matrix_factorization]
        self.weights = []
        self.verbose = verbose
        self.train(training_path)

    def train(self, training_path: str) -> None:
        # Loading training data
        columns_name = ['user_id','item_id','rating','timestamp']
        train_data = pd.read_csv(training_path, sep='\t', names=columns_name)
        
        # Training all models
        if self.verbose:
            print(f'Started training hybrid recommender on {len(train_data['user_id'].unique())} users and {len(train_data['item_id'].unique())} items...')
            print(f'Training individual models...')
        for recommender in self.recommenders:
            recommender.train(train_data)
        if self.verbose:
            print(f'Finished training individual models.')
            print('Started linear regression...')

        # Find weights which minimize MSE
        self.linear_regression(train_data, visualize=self.verbose)
        if self.verbose:
            print(f'Finished linear regression, weights are:')
            for i in range(len(self.recommenders)):
                print(f'  {self.recommenders[i].get_name()}: {self.weights[i]}')
            print(f'Filling prediction dataframe...')
        # Predict score for each u/i pair for each model, so aggregated dataframe can be created for ranking prediction
        user_ids = train_data['user_id'].unique()
        item_ids = train_data['item_id'].unique()
        pairs = list(itertools.product(user_ids, item_ids))
        self.predicted_scores = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        
        self.predicted_scores['predicted_score'] = self.predicted_scores.apply(lambda x : self._predict_score_with_weights(x['user_id'], x['item_id'], self.weights), axis=1)
        if self.verbose:
            print('Finished prediction dataframe, hybrid model is ready to use!')

    def linear_regression(
        self,
        train_data: pd.DataFrame,
        visualize: bool = False,
        save_dir: str = "plots",
        track_every: int = 1,              # record every k iterations
        max_snapshots: int = 250,          # cap history length
        track_dims: Optional[Sequence[int]] = None  # which weight indices to plot (None = all)
    ):
        ws0 = np.zeros(len(self.recommenders), dtype=float)

        mse_history: List[float] = []
        weights_history: List[np.ndarray] = []

        # Store the latest f(w) from the most recent evaluation to avoid recomputation
        last_f = {'val': None}  # mutable enclosure for reuse in callback
        iter_count = {'k': 0}

        def mse(weights: List[float]):
            y_pred = train_data.apply(
                lambda x: self._predict_score_with_weights(x['user_id'], x['item_id'], weights),
                axis=1
            ).to_numpy()
            y_true = train_data['rating'].to_numpy()
            errors = y_true - y_pred
            val = np.dot(errors, errors) / len(errors)
            last_f['val'] = val  # cache for callback
            return val

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
        res = minimize(mse, ws0, method='L-BFGS-B', callback=cb)
        self.weights = res.x

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
            file_path = os.path.join(save_dir, f"linear_regression_{timestamp}.png")
            plt.savefig(file_path, dpi=300)
            plt.close(fig)

            print(f"[INFO] Visualization saved to: {file_path}")

    def predict_score(self, user_id: int, item_id: int) -> float:
        df = self.predicted_scores
        return df.loc[((df['user_id'] == user_id) & (df['item_id'] == item_id)), 'predicted_score'].values[0]
    
    def _predict_score_with_weights(self, user_id: int, item_id: int, weights: List[float]) -> float:
        return np.sum([p[0] * p[1].predict_score(user_id, item_id) for p in zip(weights, self.recommenders)])
        
    def predict_top_k(self, user_id: int, k: int) -> List[int]:
        user_df = self.predict_scores[self.predicted_scores['user_id'] == user_id]
        top_items = user_df.nlargest(k, 'predicted_score')['item_id']
        return top_items.tolist()
