import pandas as pd
import itertools
from recommendation_algorithms.abstract_recommender import AbstractRecommender
from collections.abc import Callable
from typing import List
from tqdm import tqdm
from recommendation_algorithms.user_knn import UserKNN 
from recommendation_algorithms.item_knn import ItemKNN 



def grid_search(hyperparameter_dict: dict, recommendation_algorithm:AbstractRecommender, train_data:pd.DataFrame,  metric: Callable[[List[int], List[int]], dict], similarity_matrix=None) -> dict: 

        
    best_config = {}
    hyperparams = hyperparameter_dict.keys()
    combinations = itertools.product(*hyperparameter_dict.values())
    gridsearch = [dict(zip(hyperparams, cc)) for cc in combinations]
    params = []

    best_params = None 
    best_params_score = float('inf')
    for grid in tqdm(gridsearch): 
        recommendation_algorithm_curr = recommendation_algorithm(**grid)
        if isinstance(recommendation_algorithm_curr, UserKNN) or isinstance(recommendation_algorithm_curr, ItemKNN) :
            recommendation_algorithm_curr.restore_training(train_data, similarity_matrix)
        else:
            recommendation_algorithm_curr.train(train_data)
        recommendation_algorithm_curr.calculate_all_predictions(train_data)
        score = metric(recommendation_algorithm_curr.predictions["predicted_score"], train_data["rating"])
        params.append((score, grid))
        print("Parameters", [(k, params[k]) for k in best_params.keys() if (k != "data" and k!= "content")], "with metric:", score)
        if score < best_params_score: 
            best_params = grid
            best_params_score = score
            
    print("-----------------------------------")
    print("Best params metric", best_params_score)
    print("Best params:", [(k, best_params[k]) for k in best_params.keys() if (k != "data" and k!= "content")])

            
    return best_config, params
