from __future__ import annotations

import os
import re
import hashlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .config import WorkerConfig
from .dataset import normalize_dataset, split_parameters
from .errors import PermanentJobError, RetryableJobError
from .process import ProcessRunner
from .protocol import ExecutionSpec
from .storage import ObjectStorage, verify_sha256


ProgressCallback = Callable[[str, int, str], None]
CancelCheck = Callable[[], bool]


PARAM_FLAGS = {
    "num_topics": "--num_topics",
    "vocab_size": "--vocab_size",
    "epochs": "--epochs",
    "batch_size": "--batch_size",
    "hidden_dim": "--hidden_dim",
    "hidden_sizes": "--hidden_sizes",
    "learning_rate": "--learning_rate",
    "patience": "--patience",
    "skip_eval": "--skip-eval",
    "skip_viz": "--skip-viz",
    "no_early_stopping": "--no_early_stopping",
    "language": "--language",
    "model_size": "--model_size",
    "mode": "--mode",
    "kl_start": "--kl_start",
    "kl_end": "--kl_end",
    "kl_warmup": "--kl_warmup",
    "embedding_provider": "--embedding-provider",
    "max_iter": "--max_iter",
    "max_topics": "--max_topics",
    "n_iter": "--n_iter",
    "alpha": "--alpha",
    "beta": "--beta",
    "inference_type": "--inference_type",
    "dropout": "--dropout",
    "num_layers": "--num_layers",
    "embedding_dim": "--embedding_dim",
    "n_neighbors": "--n_neighbors",
    "n_components": "--n_components",
    "min_cluster_size": "--min_cluster_size",
    "min_samples": "--min_samples",
    "top_n_words": "--top_n_words",
    "random_state": "--random_state",
}

PREPARE_PARAM_NAMES = {
    "vocab_size",
    "batch_size",
    "model_size",
    "mode",
    "embedding_provider",
    "language",
}


@dataclass(frozen=True)
class JobPaths:
    root: Path
    data_root: Path
    workspace: Path
    result_root: Path
    source_file: Path
    dataset_file: Path
    log_file: Path

    @classmethod
    def create(cls, job_root: Path, spec: ExecutionSpec) -> "JobPaths":
        root_base = job_root.resolve()
        root_base.mkdir(parents=True, exist_ok=True)
        root = (root_base / f"task-{spec.task_id}-attempt-{spec.attempt}").resolve()
        try:
            root.relative_to(root_base)
        except ValueError as exc:
            raise PermanentJobError("unsafe job directory") from exc
        if root.exists():
            shutil.rmtree(root)
        data_root = root / "data"
        dataset_dir = data_root / f"dataset_{spec.task_id}"
        workspace = root / "workspace"
        result_root = root / "result"
        for directory in (dataset_dir, workspace, result_root):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            data_root=data_root,
            workspace=workspace,
            result_root=result_root,
            source_file=dataset_dir / spec.dataset.filename,
            dataset_file=dataset_dir / "data.csv",
            log_file=root / "worker.log",
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@dataclass(frozen=True)
class PipelineResult:
    result_dir: Path
    paths: JobPaths
    elapsed_seconds: float


class ThetaPipeline:
    def __init__(
        self,
        config: WorkerConfig,
        storage: ObjectStorage,
        process_runner: ProcessRunner | None = None,
    ):
        self.config = config
        self.storage = storage
        self.process_runner = process_runner or ProcessRunner()
        self.models_dir = config.project_root / "src" / "models"
        self.prepare_script = self.models_dir / "prepare_data.py"
        self.pipeline_script = self.models_dir / "run_pipeline.py"

    def validate_installation(self) -> None:
        if not self.prepare_script.is_file() or not self.pipeline_script.is_file():
            raise RuntimeError(
                f"THETA training scripts were not found under {self.models_dir}"
            )

    def execute(
        self,
        spec: ExecutionSpec,
        is_cancelled: CancelCheck,
        progress: ProgressCallback,
    ) -> PipelineResult:
        self.validate_installation()
        paths = JobPaths.create(self.config.job_root, spec)
        started = time.monotonic()
        deadline = started + spec.resources.timeout_seconds

        progress("downloading", 5, "正在读取数据集")
        self.storage.download(spec.dataset.object_key, paths.source_file)
        verify_sha256(paths.source_file, spec.dataset.sha256)

        input_options, engine_params = split_parameters(spec.params)
        progress("preparing_data", 8, "正在解析数据格式与所选列")
        normalize_dataset(paths.source_file, paths.dataset_file, spec.dataset.format, input_options)
        if spec.model.name == "theta":
            # THETA's native main.py still resolves the normalized text table by
            # this legacy dataset-specific name, while every baseline consumes
            # data.csv through the prepared workspace. Keep both names inside
            # the isolated job directory; they contain identical verified data.
            shutil.copy2(
                paths.dataset_file,
                paths.dataset_file.with_name(f"{self._dataset_alias(spec)}_text_only.csv"),
            )
        spec = self._with_params(spec, engine_params)

        environment = self._child_environment(paths)
        prepare_command = self.build_prepare_command(spec, paths)
        progress("preparing_data", 10, "正在预处理数据")
        self._run_command(
            prepare_command, paths, environment, deadline, is_cancelled, progress, "preparing_data", 20
        )

        data_exp = self._find_theta_experiment(spec, paths) if spec.model.name == "theta" else None
        train_command = self.build_train_command(spec, paths, data_exp)
        progress("training", 30, "正在启动模型训练")
        self._run_command(
            train_command, paths, environment, deadline, is_cancelled, progress, "training", 60
        )

        result_dir = self._find_result_directory(spec, paths, data_exp)
        self._validate_result(spec, result_dir)
        progress("uploading", 90, "训练结束，正在整理模型产物")
        return PipelineResult(result_dir, paths, time.monotonic() - started)

    @staticmethod
    def _with_params(spec: ExecutionSpec, params: dict[str, object]) -> ExecutionSpec:
        from dataclasses import replace

        return replace(spec, params=params)

    def build_prepare_command(self, spec: ExecutionSpec, paths: JobPaths) -> list[str]:
        dataset_alias = self._dataset_alias(spec)
        task_alias = self._task_alias(spec)
        prepare_model = (
            "theta"
            if spec.model.name == "theta"
            else "dtm"
            if spec.model.name == "dtm"
            else "baseline"
        )
        command = [
            self.config.python_executable,
            str(self.prepare_script),
            "--dataset",
            dataset_alias,
            "--model",
            prepare_model,
            "--user_id",
            self._user_alias(spec),
            "--exp_name",
            task_alias,
            "--force",
        ]
        if spec.model.name != "theta":
            command.extend(["--output_dir", str(paths.workspace)])
            if spec.model.name not in {"ctm", "bertopic"}:
                command.append("--skip-sbert")
            if spec.model.name in {"lda", "hdp", "btm", "stm", "nvdm", "gsm", "prodlda"}:
                # These models consume BOW only. Unused Word2Vec training wastes
                # time and introduces an unrelated dependency/failure boundary.
                command.append("--bow-only")
            if spec.model.name == "dtm":
                command.append("--with-time")
        command.extend(self._parameter_arguments(spec.params, PREPARE_PARAM_NAMES))
        return command

    def build_train_command(
        self, spec: ExecutionSpec, paths: JobPaths, data_exp: Path | None
    ) -> list[str]:
        command = [
            self.config.python_executable,
            str(self.pipeline_script),
            "--dataset",
            self._dataset_alias(spec),
            "--models",
            spec.model.name,
            "--user_id",
            self._user_alias(spec),
            "--task_name",
            self._task_alias(spec),
        ]
        if spec.model.name == "theta":
            if data_exp is None:
                raise PermanentJobError("THETA preprocessing did not produce an experiment")
            command.extend(["--data_exp", data_exp.name, "--exp_name", self._task_alias(spec)])
        else:
            command.extend(["--workspace_dir", str(paths.workspace)])
        if self.config.resource_class == "gpu" and self.config.gpu_id is not None:
            command.extend(["--gpu", str(self.config.gpu_id)])
        command.extend(self._parameter_arguments(spec.params, set(PARAM_FLAGS)))
        return command

    @staticmethod
    def _parameter_arguments(params: Mapping[str, object], allowed: set[str]) -> list[str]:
        result: list[str] = []
        for name in sorted(params):
            if name not in PARAM_FLAGS:
                raise PermanentJobError(f"unsupported training parameter: {name}")
            if name not in allowed:
                continue
            value = params[name]
            flag = PARAM_FLAGS[name]
            if isinstance(value, bool):
                if value:
                    result.append(flag)
            elif isinstance(value, (list, tuple)):
                result.append(flag)
                result.extend(str(item) for item in value)
            elif value is not None:
                result.extend([flag, str(value)])
        return result

    def _run_command(
        self,
        command: list[str],
        paths: JobPaths,
        environment: Mapping[str, str],
        deadline: float,
        is_cancelled: CancelCheck,
        progress: ProgressCallback,
        phase: str,
        percentage: int,
    ) -> None:
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise RetryableJobError("task timed out before command execution")

        last_progress = 0.0

        def heartbeat() -> None:
            nonlocal last_progress
            now = time.monotonic()
            if now - last_progress >= self.config.heartbeat_interval_seconds:
                progress(phase, percentage, "模型训练中" if phase == "training" else "数据准备中")
                last_progress = now

        self.process_runner.run(
            command=command,
            cwd=self.models_dir,
            env=environment,
            log_path=paths.log_file,
            timeout_seconds=remaining,
            is_cancelled=is_cancelled,
            heartbeat=heartbeat,
            shutdown_grace_seconds=self.config.shutdown_grace_seconds,
        )

    def _child_environment(self, paths: JobPaths) -> dict[str, str]:
        environment = dict(os.environ)
        for name in list(environment):
            upper = name.upper()
            if upper.startswith("REDIS_") or upper in {
                "OBJECT_STORAGE_ACCESS_KEY",
                "OBJECT_STORAGE_SECRET_KEY",
                "OBJECT_STORAGE_SESSION_TOKEN",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
            }:
                environment.pop(name, None)
        environment.update(
            {
                "PROJECT_ROOT": str(self.config.project_root),
                "DATA_DIR": str(paths.data_root),
                "WORKSPACE_DIR": str(paths.workspace),
                "RESULT_DIR": str(paths.result_root),
                "PYTHONUNBUFFERED": "1",
                # Windows commonly defaults redirected child-process output to
                # GBK.  The training scripts print Unicode status symbols, so
                # force UTF-8 for both stdout/stderr and Python's text mode.
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            }
        )
        if self.config.resource_class == "gpu" and self.config.gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
        elif self.config.resource_class == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        return environment

    @staticmethod
    def _dataset_alias(spec: ExecutionSpec) -> str:
        return f"dataset_{spec.task_id}"

    @staticmethod
    def _user_alias(spec: ExecutionSpec) -> str:
        digest = hashlib.sha256(spec.user_id.encode("utf-8")).hexdigest()[:16]
        return f"user_{digest}"

    @staticmethod
    def _task_alias(spec: ExecutionSpec) -> str:
        return f"job_{spec.task_id}_a{spec.attempt}"

    def _find_theta_experiment(self, spec: ExecutionSpec, paths: JobPaths) -> Path:
        base = (
            paths.result_root
            / self._dataset_alias(spec)
            / str(spec.params.get("model_size", "0.6B"))
            / "theta"
        )
        matches = sorted(base.glob(f"exp_*_{self._task_alias(spec)}")) if base.exists() else []
        if not matches:
            raise PermanentJobError("THETA preprocessing did not create the expected experiment")
        return matches[-1]

    def _find_result_directory(
        self, spec: ExecutionSpec, paths: JobPaths, data_exp: Path | None
    ) -> Path:
        if spec.model.name == "theta":
            assert data_exp is not None
            return data_exp
        return (
            paths.result_root
            / self._user_alias(spec)
            / self._dataset_alias(spec)
            / spec.model.name
            / self._task_alias(spec)
        )

    @staticmethod
    def _validate_result(spec: ExecutionSpec, result_dir: Path) -> None:
        if not result_dir.is_dir():
            raise RetryableJobError(f"training did not create result directory: {result_dir}")
        trained = any(
            re.fullmatch(r'theta(?:_k\d+)?\.npy', path.name)
            and path.with_name(path.name.replace('theta', 'beta', 1)).is_file()
            for path in result_dir.rglob('theta*.npy') if path.is_file()
        )
        if spec.model.name == 'bertopic' and not (
            any(result_dir.rglob('document_topics.npy')) and any(result_dir.rglob('result_manifest.json'))
        ):
            raise RetryableJobError(
                f"BERTopic export incomplete: document assignments or result manifest missing; partial results: {result_dir}"
            )
        if not trained:
            raise RetryableJobError(
                "训练进程已结束，但未生成模型文件；请检查模型所需的数据列和执行日志"
            )
