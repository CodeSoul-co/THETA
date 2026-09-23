from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Protocol

from .config import StorageConfig
from .errors import PermanentJobError, RetryableJobError
from .protocol import validate_object_key


class ObjectStorage(Protocol):
    def download(self, object_key: str, destination: Path) -> None: ...

    def upload(self, source: Path, object_key: str) -> None: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    expected = expected.strip().lower()
    if not expected:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise PermanentJobError(
            f"dataset checksum mismatch: expected {expected}, got {actual}"
        )


def _safe_local_path(root: Path, object_key: str) -> Path:
    key = validate_object_key(object_key, "object_key")
    root = root.resolve()
    path = (root / Path(*key.split("/"))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermanentJobError("object key escapes storage root") from exc
    return path


class FilesystemObjectStorage:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def download(self, object_key: str, destination: Path) -> None:
        source = _safe_local_path(self.root, object_key)
        if not source.is_file():
            raise PermanentJobError(f"object does not exist: {object_key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def upload(self, source: Path, object_key: str) -> None:
        if not source.is_file():
            raise PermanentJobError(f"upload source does not exist: {source}")
        destination = _safe_local_path(self.root, object_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


class S3ObjectStorage:
    def __init__(self, config: StorageConfig):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("S3 storage requires boto3; install worker/requirements.txt") from exc

        endpoint = config.endpoint or None
        if endpoint and "://" not in endpoint:
            scheme = "https" if config.use_ssl else "http"
            endpoint = f"{scheme}://{endpoint}"

        kwargs: dict[str, object] = {
            "service_name": "s3",
            "endpoint_url": endpoint,
            "region_name": config.region,
            "config": Config(signature_version="s3v4"),
        }
        if config.access_key:
            kwargs["aws_access_key_id"] = config.access_key
        if config.secret_key:
            kwargs["aws_secret_access_key"] = config.secret_key
        if config.session_token:
            kwargs["aws_session_token"] = config.session_token
        self.client = boto3.client(**kwargs)
        self.bucket = config.bucket

    def download(self, object_key: str, destination: Path) -> None:
        key = validate_object_key(object_key, "object_key")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, key, str(destination))
        except Exception as exc:
            raise RetryableJobError(f"download object {key} failed: {exc}") from exc

    def upload(self, source: Path, object_key: str) -> None:
        if not source.is_file():
            raise PermanentJobError(f"upload source does not exist: {source}")
        key = validate_object_key(object_key, "object_key")
        try:
            self.client.upload_file(str(source), self.bucket, key)
        except Exception as exc:
            raise RetryableJobError(f"upload object {key} failed: {exc}") from exc


def create_storage(config: StorageConfig) -> ObjectStorage:
    if config.driver == "filesystem":
        return FilesystemObjectStorage(config.filesystem_root)
    if config.driver == "s3":
        return S3ObjectStorage(config)
    raise ValueError(f"unsupported object storage driver: {config.driver}")
