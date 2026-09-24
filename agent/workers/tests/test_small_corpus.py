"""The default BOW preprocessing must not filter every word in a two-document corpus."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

class SmallCorpusTest(unittest.TestCase):
    def test_two_documents_keep_vocabulary_and_can_train_prodlda(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as temporary:
            code = f'''
import sys
from pathlib import Path
sys.path.insert(0, {str(root / 'src/models')!r})
from prepare_data import generate_bow
from model.baseline.prodlda import ProdLDA
import torch
matrix, vocabulary = generate_bow(['robot machine sensor technology', 'fruit apple orange garden'], 100, Path({temporary!r}))
assert matrix.shape[0] == 2 and len(vocabulary) > 0
model = ProdLDA(vocab_size=len(vocabulary), num_topics=2, hidden_dim=16)
output = model(torch.tensor(matrix.toarray(), dtype=torch.float32))
loss = output['recon_loss'].mean() + output['kl_loss'].mean()
assert torch.isfinite(loss)
loss.backward()
'''
            result = subprocess.run([sys.executable, '-c', code], cwd=root, text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
