import pandas as pd
import itertools
from recommendation_algorithms.abstract_recommender import AbstractRecommender
from collections.abc import Callable
from typing import List 



def grid_search(hyperparameter_dict: dict, recommendation_algorithm:AbstractRecommender, train_data:pd.DataFrame,  metric: Callable[[List[int], List[int]], dict]) -> dict: 

        
    best_config = {}
    hyperparams = hyperparameter_dict.keys()
    combinations = itertools.product(*hyperparameter_dict.values())
    gridsearch = [dict(zip(hyperparams, cc)) for cc in combinations]
    params = []

    best_params = None 
    best_params_score = float('inf')
    for grid in gridsearch: 
        recommendation_algorithm_curr = recommendation_algorithm(**grid)
        recommendation_algorithm_curr.train(train_data)
        recommendation_algorithm_curr.calculate_all_predictions(train_data)
        score = metric(recommendation_algorithm_curr.predictions["predicted_score"], train_data["rating"])
        params.append((score, grid))
        print("Parameters", grid, "with metric:", score)
        if score < best_params_score: 
            best_params = grid
            
    print("-----------------------------------")
    print("Best params metric", best_params_score)
    print("Best params:", best_params)

            
    return best_config
