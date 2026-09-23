import unittest
from types import SimpleNamespace
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/models'))
from utils.local_embedding import encode_documents, last_token_pool


class Encoded(dict):
    def to(self, device):
        return Encoded({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False): return 0
    def encode(self, text, **kwargs): return list(range(1, int(text) + 1))
    def prepare_for_model(self, tokens, **kwargs): return {'input_ids': tokens, 'attention_mask': [1] * len(tokens)}
    def pad(self, rows, **kwargs):
        length = max(len(row['input_ids']) for row in rows)
        return Encoded({key: torch.tensor([row[key] + [0] * (length - len(row[key])) for row in rows]) for key in rows[0]})


class Model:
    def __init__(self): self.batch_sizes = []
    def __call__(self, input_ids, attention_mask):
        self.batch_sizes.append(input_ids.shape[0])
        return SimpleNamespace(last_hidden_state=input_ids.unsqueeze(-1).float())


class LocalEmbeddingTests(unittest.TestCase):
    def test_last_token_uses_mask_for_left_and_right_padding(self):
        hidden = torch.tensor([[[0.], [1.], [2.]], [[1.], [2.], [0.]]])
        np.testing.assert_allclose(last_token_pool(hidden, torch.tensor([[0, 1, 1], [1, 1, 0]])), [[2.], [2.]])

    def test_batches_preserve_all_windows_and_document_order(self):
        model = Model()
        actual = encode_documents(['29', '3', '17'], Tokenizer(), model, 'cpu', 16, 2)
        np.testing.assert_allclose(actual[:, 0], [(16 + 24 + 29) / 3, 3, (16 + 17) / 2])
        self.assertEqual(model.batch_sizes, [2, 2, 2])

    def test_batch_size_does_not_change_embeddings(self):
        first = encode_documents(['29', '3', '17'], Tokenizer(), Model(), 'cpu', 16, 1)
        second = encode_documents(['29', '3', '17'], Tokenizer(), Model(), 'cpu', 16, 4)
        np.testing.assert_allclose(first, second)
