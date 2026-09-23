import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


MODELS_DIR = Path(__file__).resolve().parents[1]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from model.embedding_providers import EmbeddingProviderSettings
from config import config_from_args, create_parser


class EmbeddingProviderSettingsTests(unittest.TestCase):
    def test_zero_shot_preserves_zero_controls_and_freezes_prepared_embeddings(self):
        args = create_parser().parse_args([
            "pipeline", "--dataset", "fixture", "--mode", "zero_shot",
            "--patience", "0", "--kl_warmup", "0",
        ])
        config = config_from_args(args)
        self.assertEqual(config.model.patience, 0)
        self.assertEqual(config.model.kl_warmup_epochs, 0)
        self.assertFalse(config.model.train_word_embeddings)

    def test_task_selected_key_environment_takes_precedence(self):
        settings = EmbeddingProviderSettings(
            provider="openai_compatible",
            cloud_provider="openai_compatible",
            api_base="https://embedding.example/v1",
            api_key_env="TENANT_EMBEDDING_KEY",
            model="example-model",
        )
        with patch.dict(
            os.environ,
            {"EMBEDDING_API_KEY": "generic", "TENANT_EMBEDDING_KEY": "selected"},
            clear=False,
        ):
            self.assertEqual(settings.api_key, "selected")

    def test_generic_key_is_a_fallback(self):
        settings = EmbeddingProviderSettings(
            provider="openai_compatible",
            cloud_provider="openai_compatible",
            api_base="https://embedding.example/v1",
            api_key_env="MISSING_EMBEDDING_KEY",
            model="example-model",
        )
        with patch.dict(os.environ, {"EMBEDDING_API_KEY": "generic"}, clear=False):
            os.environ.pop("MISSING_EMBEDDING_KEY", None)
            self.assertEqual(settings.api_key, "generic")


if __name__ == "__main__":
    unittest.main()
