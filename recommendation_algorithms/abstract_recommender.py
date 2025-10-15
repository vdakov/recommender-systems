from abc import ABC, abstractmethod
from typing import List
import pandas as pd

class AbstractRecommender(ABC):
    @abstractmethod
    def train(self, train_data: pd.DataFrame) -> None:
        pass 

    @abstractmethod
    def predict_score(self, user_id: int, item_id: int) -> float:
        # TODO
        pass 

    @abstractmethod
    def predict_top_k(self, user_id: int, k: int) -> List[int]:
        # TODO, has to know all scores for all algorithms for all user/item pairs
        pass
