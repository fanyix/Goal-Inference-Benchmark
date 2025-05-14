import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dataclasses import dataclass
from evals.predictors.multimodal.api import MultiModalPredictor
from typing import Union


@dataclass(frozen=True)
class HuggingFacePredictorConfig:
    model_path: str
    device: Union[str, torch.device] = "cuda"

    def __post_init__(self) -> None:
        """Ensure that the model path exists and that CUDA is available if using a GPU."""
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")


class HuggingFacePredictor(MultiModalPredictor):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dtype: Union[torch.dtype, str],
        device: str = "cuda",
        batch_size: int = 1,
    ) -> None:
        """
        A wrapper around a HuggingFace model.
        This class is partially abstract and users will need to implement other methods defined in MultiModalPredictor.

        Args:
            model: The model to use. This should be a HuggingFace model, and will search within ~/.cache/huggingface/hub for the model.
            tokenizer: The tokenizer to use.
            dtype: The torch datatype to use for the model.
            device: The device to use for the model.
            batch_size: The batch size to use for the model.

        Returns:
            None
        """
        self.dtype: torch.dtype = dtype
        self.model: AutoModelForCausalLM = model
        self.tokenizer: AutoTokenizer = tokenizer
        self.device: Union[str, torch.device] = device
        self.batch_size: int = batch_size

    @staticmethod
    def get_tokenizer_and_model(
        model_path: str,
        dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Get a HuggingFace Tokenizer and Model.

        Args:
            model_path: The path to the checkpoint containing both the model and tokenizer.
            dtype: The torch datatype to use for the model.

        Returns:
            The tokenizer and model to use.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )
        return tokenizer, model

    @staticmethod
    def from_config(
        config: HuggingFacePredictorConfig,
    ) -> "HuggingFacePredictor":
        """
        Construct a HuggingFacePredictor from a config.

        Args:
            config: The config to use for construction.
        Returns:
            The constructed HuggingFacePredictor.
        """
        tokenizer, model = HuggingFacePredictor.get_tokenizer_and_model(
            model_path=config.model_path,
            dtype=torch.float16,
        )

        return HuggingFacePredictor(
            model=model,
            tokenizer=tokenizer,
            dtype=torch.float16,
            device=config.device,
        )
