from typing import Dict, Any, List, Callable, Tuple, Optional
import numpy as np

PreprocessPipeline = List[Callable[[np.ndarray, np.ndarray, Optional[bool]], Tuple[np.ndarray, np.ndarray]]]

AllPossibleModelParameters = Dict[str, List[Any]]

CrossValidationMean = Dict[str, Any]


class LearningModel:

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def get_params(self) -> Dict[str, Any]:
        pass

    def set_params(self, new_params: Dict[str, Any]) -> bool:
        pass
