# flake8: noqa
from typing import Any, Dict

from evals.predictors import build_predictor, PredictorRegistry
from evals.predictors.multimodal.api import MultiModalPredictor


_MULTIMODAL_PREDICTOR_CONFIG_MAP = {
    "maestro_ob2_judge": "MaestroOB2JudgeHuggingFacePredictorConfig",
    "maestro_ob2_qwen": "MaestroOB2QwenPredictorConfig",
}


class MultimodalPredictorRegistry(PredictorRegistry):
    _REGISTRY: Dict[str, Any] = {}


for name, config_cls_name in _MULTIMODAL_PREDICTOR_CONFIG_MAP.items():
    name_in = name.split("-")[0]
    MultimodalPredictorRegistry.register(
        name, f"evals.predictors.multimodal.{name_in}.{config_cls_name}"
    )
