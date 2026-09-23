class WorkerError(Exception):
    """Base exception for controlled Worker failures."""


class PermanentJobError(WorkerError):
    """The task is malformed or unsupported and should not be retried as-is."""


class RetryableJobError(WorkerError):
    """The task may succeed when retried on another attempt or Worker."""


class JobCancelled(WorkerError):
    """The control plane requested cancellation."""


class WorkerStopping(WorkerError):
    """The Worker is shutting down; leave the message pending for recovery."""


class LeaseLost(WorkerError):
    """Another Worker may own the task; stop without publishing a terminal event."""
