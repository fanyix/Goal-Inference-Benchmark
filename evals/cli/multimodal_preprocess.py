import inspect
import os
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import cached_property
from typing import Any, Dict, List, Optional

import evals.tasks.multimodal  # noqa

import torch
from evals.api import PredictorConfig, TaskConfig
from evals.params import from_cli, parse_args, to_cli
from evals.predictors.multimodal import build_predictor, MultimodalPredictorRegistry
from evals.tasks.multimodal import (
    build_multimodal_task,
    MultimodalTaskRegistry,
    VideoReasoningTaskConfig,
)
from evals.utils import (
    filter_by_pattern,
    flatten_dict,
    format_dict,
    get_git_info,
    get_global_rank,
    get_random_state,
    initialize_logger,
    rank_zero_info,
    setup_env,
    unroll_configs,
    write_to_json,
)


@dataclass
class MultimodalEvalConfig:
    dataset_dir: str
    tasks: str
    task_args: Optional[Dict[str, Dict[str, Any]]] = None
    seed: int = 42
    manual_seed: bool = True
    dump_dir: Optional[str] = None
    max_samples: Optional[int] = None
    show_progress: bool = True

    predictor_name: str = "ugen"
    predictor_config: Optional[PredictorConfig] = None

    def __post_init__(self) -> None:
        available = list(MultimodalPredictorRegistry.names())
        assert self.predictor_name in available, (self.predictor_name, available)

        # TODO: after adding more tasks, revisit here about the name format
        # For multimodal models, each `task_name` should be in the format {task}-{template}.
        # If they miss the {template}, we will here add their default template type,
        # which is named by the predictor name.
        self.tasks = ",".join(
            [
                name if "-" in name else f"{name}-{self.predictor_name}"
                for name in self.tasks.split(",")
            ]
        )

    @cached_property
    def task_configs(self) -> Dict[str, TaskConfig]:
        configs: Dict[str, TaskConfig] = {}
        for name in filter_by_pattern(MultimodalTaskRegistry.names(), self.tasks):
            defaults = inspect.signature(
                MultimodalTaskRegistry._REGISTRY[name]
            ).parameters
            params = (self.task_args or {}).get(name, {})
            for fname, kwargs in unroll_configs(defaults, params, prefix=name).items():
                if "dataset_dir" in defaults:
                    kwargs["dataset_dir"] = self.dataset_dir
                configs[fname] = MultimodalTaskRegistry.get_config(name, **kwargs)
        return configs


def main(cfg: MultimodalEvalConfig) -> None:
    if cfg.manual_seed:
        torch.manual_seed(cfg.seed)
    _ckpt_file = getattr(cfg.predictor_config, "checkpoint_dir", "").split("/")[-1]
    now = datetime.now()
    _date = now.strftime("%y_%m_%d_%H_%M_%S")
    folder = f"{cfg.predictor_name}_{_date}"
    if _ckpt_file != "":
        folder += f"_{_ckpt_file}"

    if cfg.dump_dir is not None and get_global_rank() == 0:
        serializable_task_configs = {}
        for task_name, task_config in cfg.task_configs.items():
            if isinstance(task_config, VideoReasoningTaskConfig):
                serializable_task_configs[task_name] = task_config.to_json_dict()
            else:
                serializable_task_configs[task_name] = task_config  # type: ignore[assignment]

        metadata = {
            "parameters": asdict(cfg),
            "git_info": get_git_info(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": " ".join(map(shlex.quote, sys.argv)),
            "task_configs": serializable_task_configs,
        }
        metadata_file = os.path.join(cfg.dump_dir, folder, "metadata.json")
        write_to_json(metadata, metadata_file)

    names: List[str] = filter_by_pattern(MultimodalTaskRegistry.names(), cfg.tasks)
    assert len(names) > 0, f"No tasks were found given the pattern '{cfg.tasks}'"
    rank_zero_info(f"Selected tasks for execution: {names}")

    task_configs = cfg.task_configs
    assert cfg.predictor_config is not None
    predictor = build_predictor(cfg.predictor_config)

    for name, task_config in task_configs.items():
        task = build_multimodal_task(task_config)
        start = time.monotonic()

        name = name.split(".")[0]  # only use the readable part of the name for logging
        rank_zero_info(f"Running preprocessing on task {name}")

        # For MP rank 0 and DP rank 0, result contains:
        #   - "metrics": average results
        #   - "raw_results": per-data, includng model input and output
        # Otherwise, result is empty.
        result = task.run(  # type: ignore
            predictor=predictor,
            random_state=get_random_state(cfg.seed),
            max_samples=cfg.max_samples,
            show_progress=cfg.show_progress,
            # TODO: kwargs (e.g. temperature) are decided by the predictor for different
            # supported arguments and default values. It is now initialized in predictor config
            # but not used in predictor initialization but used here as inputs of task.run()
            generate_kwargs=getattr(cfg.predictor_config, "generate_kwargs", {}),
        )
        log = format_dict(flatten_dict(result.metrics), delimiter=" | ", decimal=6)
        rank_zero_info(f"Evaluation results on task {name}: {log}")
        rank_zero_info(f"Task {name} took {time.monotonic() - start:.2f} seconds")


def cfg_from_cli() -> MultimodalEvalConfig:
    known, unknown = to_cli(MultimodalEvalConfig).parse_known_args()  # type: ignore
    cfg: MultimodalEvalConfig = from_cli(
        MultimodalEvalConfig, vars(known), allow_incomplete=True
    )

    cfg.predictor_config = parse_args(
        MultimodalPredictorRegistry.get_config_cls(cfg.predictor_name), unknown
    )
    return cfg


if __name__ == "__main__":
    cfg: MultimodalEvalConfig = cfg_from_cli()
    initialize_logger()
    setup_env()
    main(cfg)
