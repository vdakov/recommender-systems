from datetime import datetime
from typing import Dict, List, Optional, Sequence
from recommendation_algorithms.abstract_recommender import AbstractRecommender
from recommendation_algorithms.five_recommender import FiveRecommender
from recommendation_algorithms.matrix_factorization import MatrixFactorizationSGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import os


class HybridRecommender:
    recommenders: List[AbstractRecommender]
    weights: List[float]  # Each weight corresponds to a recommender
    verbose: bool
    predictions: pd.DataFrame # Precomputed predictions for all user/item pairs

    def __init__(self, training_path: str, verbose=False):
        # TODO fill list of recommenders
        matrix_factorization = MatrixFactorizationSGD()
        five_recommender = FiveRecommender() # TODO remove
        self.recommenders = [matrix_factorization, five_recommender]
        self.predictions = {} # For each recommender, keep a dataframe of precompute 
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
            recommender.calculate_all_predictions(train_data)
        if self.verbose:
            print(f'Finished training individual models.')
            print('Started linear regression...')

        # Find weights which minimize MSE
        self.linear_regression(train_data, visualize=self.verbose)
        if self.verbose:
            print(f'Finished linear regression, weights are:')
            for i in range(len(self.recommenders)):
                print(f'  {self.recommenders[i].get_name()}: {self.weights[i]}')

        # Precompute all predictions
        print(f"Precomputing predictions...")
        dfs = []
        for df, w in zip([r.predictions for r in self.recommenders], self.weights):
            temp = df.copy()
            temp['weighted_score'] = temp['predicted_score'] * w
            dfs.append(temp[['user_id', 'item_id', 'weighted_score']])
        combined = pd.concat(dfs, ignore_index=True)
        self.predictions = (
            combined.groupby(['user_id', 'item_id'], as_index=False, sort=False)
                    .agg(predicted_score=('weighted_score', 'sum'))
        )
        print("Finished computing predictions, model is ready to use.")

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

        # Objective function to optimize (MSE)
        def objective(weights: List[float]):
            # Find weighted sum of predictions
            dfs = []
            for df, w in zip([r.predictions for r in self.recommenders], weights):
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

    def _predict_score_with_weights(self, user_id: int, item_id: int, weights: List[float]) -> float:
        return np.sum([p[0] * p[1].get_cached_predicted_score(user_id, item_id) for p in zip(weights, self.recommenders)])

    def predict_score(self, user_id: int, item_id: int) -> float:
        # Find precomputed prediction
        return self.predictions.loc[((self.predictions['item_id'] == item_id) & (self.predictions['user_id'] == user_id)), 'predicted_score'].values[0]
    
    def predict_ranking(self, user_id: int, k: int) -> List[int]:
        # Predict ranking based on precomputed scores
        user_df = self.predictions.loc[(self.predictions['user_id'] == user_id), ['item_id', 'predicted_score']]
        top_k = user_df.nlargest(k, 'predicted_score')
        return top_k['item_id'].to_list()
