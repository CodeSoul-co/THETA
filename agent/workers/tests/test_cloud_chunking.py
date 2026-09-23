import os
import http.client
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/models'))
from model.embedding_providers import EmbeddingProviderSettings, OpenAICompatibleEmbeddingProvider, split_utf8_text


class CloudChunkingTests(unittest.TestCase):
    def test_truncated_http_response_retries_without_returning_partial_vectors(self):
        class BrokenResponse(io.BytesIO):
            def read(self, *args):
                raise http.client.IncompleteRead(b'partial')
        with patch.dict(os.environ, {'EMBEDDING_API_KEY': 'unit-test-not-a-secret'}):
            provider = OpenAICompatibleEmbeddingProvider(EmbeddingProviderSettings(provider='cloud', cloud_provider='zhipu', api_base='https://unit.invalid', api_key_env='EMBEDDING_API_KEY', model='embedding-3', normalize=False, max_retries=2))
            with patch('urllib.request.urlopen', side_effect=[BrokenResponse(), io.BytesIO(b'{"data":[{"index":0,"embedding":[1,2]}]}')]) as transport, patch('time.sleep'):
                np.testing.assert_allclose(provider._embed_batch(['test']), [[1, 2]])
                self.assertEqual(transport.call_count, 2)
            with patch('urllib.request.urlopen', side_effect=http.client.IncompleteRead(b'partial')) as transport, patch('time.sleep'):
                with self.assertRaises(RuntimeError):
                    provider._embed_batch(['test'])
                self.assertEqual(transport.call_count, 3)
            with patch('urllib.request.urlopen', side_effect=PermissionError('budget exhausted')) as transport:
                with self.assertRaises(PermissionError):
                    provider._embed_batch(['test'])
                self.assertEqual(transport.call_count, 1)

    def test_unicode_windows_reconstruct_every_character(self):
        text = '中文 English 한국어 🚀' * 500
        chunks = split_utf8_text(text, 3070)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len(chunk.encode('utf-8')) <= 3070 for chunk in chunks))

    def test_pooling_preserves_document_order_and_caps_batch(self):
        with patch.dict(os.environ, {'EMBEDDING_API_KEY': 'unit-test-not-a-secret'}):
            provider = OpenAICompatibleEmbeddingProvider(EmbeddingProviderSettings(provider='cloud', cloud_provider='zhipu', api_base='https://unit.invalid', api_key_env='EMBEDDING_API_KEY', model='embedding-3', normalize=False))
            batches = []
            def fake(texts):
                batches.append(texts)
                return [np.array([1., 0.]) if text[0] == '中' else np.array([0., 1.]) for text in texts]
            with patch.object(provider, '_embed_batch', side_effect=fake):
                actual = provider.embed(['中' * 70_000, 'English'], batch_size=128, show_progress=False)
        self.assertTrue(all(len(batch) <= 64 for batch in batches))
        np.testing.assert_allclose(actual, [[1, 0], [0, 1]])

    def test_rejects_missing_duplicate_or_invalid_vectors(self):
        for data in [[], [{'index': 0, 'embedding': [1]}], [{'index': 0, 'embedding': [1]}, {'index': 0, 'embedding': [2]}], [{'index': 0, 'embedding': [float('nan')]}, {'index': 1, 'embedding': [1]}]]:
            with self.assertRaises(ValueError):
                OpenAICompatibleEmbeddingProvider._parse_embeddings({'data': data}, 2)
