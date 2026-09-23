from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worker.errors import PermanentJobError
from worker.storage import FilesystemObjectStorage, sha256_file, verify_sha256


class FilesystemStorageTests(unittest.TestCase):
    def test_upload_download_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = FilesystemObjectStorage(root / "objects")
            source = root / "source.txt"
            source.write_text("hello", encoding="utf-8")
            storage.upload(source, "results/hello.txt")
            destination = root / "download.txt"
            storage.download("results/hello.txt", destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "hello")
            verify_sha256(destination, sha256_file(source))

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = FilesystemObjectStorage(Path(directory))
            with self.assertRaises(PermanentJobError):
                storage.download("../secret.txt", Path(directory) / "out")


if __name__ == "__main__":
    unittest.main()
