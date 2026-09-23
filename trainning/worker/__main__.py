from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

from .artifacts import ArtifactPublisher
from .config import WorkerConfig
from .agent_pipeline import create_agent_pipeline
from .redis_queue import RedisQueue
from .service import WorkerService
from .storage import create_storage


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
        load_dotenv(Path.cwd() / ".env", override=False)
    except ImportError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        config = WorkerConfig.load()
        storage = create_storage(config.storage)
        queue = RedisQueue(config.redis)
        service = WorkerService(
            config=config,
            queue=queue,
            pipeline=create_agent_pipeline(config, storage),
            artifacts=ArtifactPublisher(storage),
        )
    except Exception:
        logging.exception("worker initialization failed")
        return 1

    def stop(_signum: int, _frame: object) -> None:
        service.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        service.run()
    except Exception:
        logging.exception("worker terminated unexpectedly")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
