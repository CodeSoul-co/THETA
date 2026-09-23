"""Bridge the distributed worker to the Agent's full model/parameter contract.

The numerical implementations remain in src/models.  This adapter only selects
the same validated argument routing used by the local Agent worker.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .dataset import split_parameters
from .errors import PermanentJobError
from .process import ProcessRunner
from .protocol import ExecutionSpec


def create_agent_pipeline(config, storage):
    agent_root = config.project_root / "agent"
    if not (agent_root / "workers" / "pipeline_adapter.py").is_file():
        raise RuntimeError(f"Agent worker adapters were not found under {agent_root}")
    if str(agent_root) not in sys.path:
        sys.path.insert(0, str(agent_root))

    from workers.model_contract import validate_parameters
    from workers.pipeline_adapter import AgentThetaPipeline

    class AgentExecutionRunner(ProcessRunner):
        plan: dict | None = None

        def run(self, **kwargs):
            if self.plan is None:
                raise RuntimeError("Agent execution plan was not bound to the worker command")
            command = list(kwargs["command"])
            plan_file = Path(kwargs["log_path"]).parent / "agent-execution-plan.json"
            plan_file.parent.mkdir(parents=True, exist_ok=True)
            plan_file.write_text(json.dumps(self.plan, ensure_ascii=False), encoding="utf-8")
            plan_file.chmod(0o600)
            environment = dict(kwargs["env"])
            environment["THETA_AGENT_PLAN_FILE"] = str(plan_file)
            environment["THETA_PROJECT_ROOT"] = str(config.project_root)
            kwargs["env"] = environment
            kwargs["command"] = [
                command[0], str(agent_root / "workers" / "engine_entry.py"), *command[1:]
            ]
            return super().run(**kwargs)

    runner = AgentExecutionRunner()

    class DistributedAgentPipeline(AgentThetaPipeline):
        def execute(self, spec: ExecutionSpec, is_cancelled, progress):
            input_options, engine_params = split_parameters(spec.params)
            if "embedding.model_path" in engine_params:
                raise PermanentJobError(
                    "embedding.model_path is host-managed and cannot be supplied by a remote task"
                )
            plan = {
                "modelId": spec.model.name,
                "textColumn": input_options.get("text_column", "text"),
                "params": engine_params,
                "device": f"cuda:{config.gpu_id}" if config.resource_class == "gpu" and config.gpu_id is not None else "cpu",
            }
            if input_options.get("time_column"):
                plan["timeColumn"] = input_options["time_column"]
            if input_options.get("label_column"):
                plan["labelColumn"] = input_options["label_column"]
            if input_options.get("covariates"):
                plan["covariates"] = input_options["covariates"]
            validate_parameters(config.project_root, plan)
            self.plan = plan
            runner.plan = plan
            return super().execute(spec, is_cancelled, progress)

    return DistributedAgentPipeline(config, storage, runner)
