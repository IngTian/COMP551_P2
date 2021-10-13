from typing import TypedDict, Dict, Any, Union, List, Callable, Tuple, Optional
import logging
import numpy as np


class ModelPerformanceReport(TypedDict):
    validation_accuracy: float
    validation_macro_f1: float
    validation_weighted_f1: float
    training_accuracy: float
    training_macro_f1: float
    training_weighted_f1: float


class ParameterEffectsCurve(TypedDict):
    validation_accuracy: List[float]
    validation_macro_f1: List[float]
    validation_weighted_f1: List[float]
    training_accuracy: List[float]
    training_macro_f1: List[float]
    training_weighted_f1: List[float]
    hyper_parameter_name: str
    hyper_parameter_value: List[Any]


PreprocessPipeline = List[Callable[[np.ndarray, np.ndarray, Optional[bool]], Tuple[np.ndarray, np.ndarray]]]

AllPossibleModelParameters = Dict[str, List[Any]]


class LearningModel:

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def get_params(self) -> Dict[str, Any]:
        pass

    def set_params(self, new_params: Dict[str, Any]) -> bool:
        pass
