"""Entrypoint for training runs."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from optimizer_comparison.artifacts.hf_hub import (
    persist_training_artifacts_to_hf,
    validate_hf_hub_before_training,
)
from optimizer_comparison.artifacts.local_store import (
    create_run_dir,
    save_json,
    save_resolved_config,
)
from optimizer_comparison.tracking.mlflow_logger import log_training_run
from optimizer_comparison.training.mock_trainer import run_mock_training
from optimizer_comparison.training.result_contract import set_local_artifact_paths
from optimizer_comparison.training.trainer import run_training


# /**
#  * Запускает training-пайплайн с конфигурацией Hydra.
#  *
#  * @param config Полная конфигурация запуска, собранная Hydra.
#  * @return None. Результаты сохраняются в локальную директорию запуска.
#  */
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    validate_hf_hub_before_training(config)

    experiment_name = str(config.experiment.name)
    artifacts_root_dir = str(config.artifacts.root_dir)
    config_filename = str(config.artifacts.config_filename)
    result_filename = str(config.artifacts.result_filename)

    run_dir = create_run_dir(
        artifacts_root_dir=artifacts_root_dir,
        experiment_name=experiment_name,
    )
    config_path = save_resolved_config(config=config, run_dir=run_dir, filename=config_filename)
    result_path = run_dir / result_filename

    if bool(config.mode.use_mock_trainer):
        result = run_mock_training(config)
    else:
        result = run_training(config=config, run_dir=run_dir)

    result = set_local_artifact_paths(
        result=result,
        run_dir=run_dir,
        config_path=config_path,
        result_path=result_path,
    )
    save_json(data=result, output_path=result_path)
    result = persist_training_artifacts_to_hf(config=config, result=result)
    save_json(data=result, output_path=result_path)
    log_training_run(config=config, training_result=result)


if __name__ == "__main__":
    main()
