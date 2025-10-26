import numpy as np
from typing import List

def MAE(actual_ratings:List[float], pred_ratings:List[float]):
    absolute_errors = [np.abs(a - p) for a, p in zip(actual_ratings, pred_ratings)]
    result = np.sum(absolute_errors)/len(actual_ratings)

    return result

def MSE(actual_rating:List[float], pred_rating:List[float]):
    squared_errors = [(a - p) ** 2 for a, p in zip(actual_rating, pred_rating)]
    result = np.sum(squared_errors)/len(actual_rating)
  
    return result

def RMSE(actual_rating:List[float], pred_rating:List[float]):
    result = np.sqrt(MSE(actual_rating, pred_rating))

    return result