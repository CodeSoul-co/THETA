"""Serialization regressions, not a substitute for end-to-end training acceptance."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/models'))
from model_delivery import save_trained_model, load_trained_model


class ModelDeliveryTests(unittest.TestCase):
    def test_twelve_model_classes_share_one_verified_loader(self):
        import torch
        models = {'theta': ('model.theta.etm', 'ETM'), 'lda': ('model.baseline.lda', 'SklearnLDA'),
                  'hdp': ('model.baseline.hdp', 'HDP'), 'btm': ('model.baseline.btm', 'BTM'),
                  'stm': ('model.baseline.stm', 'STM'), 'dtm': ('model.baseline.dtm', 'DTM'),
                  'ctm': ('model.baseline.ctm', 'CTM'), 'etm': ('model.baseline.etm', 'OriginalETM'),
                  'bertopic': ('model.baseline.bertopic', 'BERTopicModel'),
                  'nvdm': ('model.baseline.nvdm', 'NVDM'), 'gsm': ('model.baseline.gsm', 'GSM'),
                  'prodlda': ('model.baseline.prodlda', 'ProdLDA')}
        for name, (module, symbol) in models.items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as folder:
                cls = getattr(importlib.import_module(module), symbol)
                kwargs = {'vocab_size': 8, 'num_topics': 2}
                if name == 'theta': kwargs.update(doc_embedding_dim=4, word_embedding_dim=4, hidden_dim=8)
                if name == 'hdp': kwargs = {'vocab_size': 8, 'max_topics': 2}
                model = cls(**kwargs)
                save_trained_model(model, folder, name, [str(i) for i in range(8)])
                with self.assertRaises(ValueError): load_trained_model(folder)
                restored = load_trained_model(folder, trusted=True)
                self.assertIs(type(restored), type(model))
                if isinstance(model, torch.nn.Module):
                    for key, tensor in model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key], tensor.cpu())
                manifest = json.loads((Path(folder) / 'model_delivery.json').read_text())
                self.assertEqual(manifest['modelId'], name)
                self.assertIn('get_beta', manifest['methods'])
                (Path(folder) / 'trained_model.joblib').write_bytes(b'corrupted')
                with self.assertRaisesRegex(ValueError, '校验失败'): load_trained_model(folder, trusted=True)

    def test_fitted_lda_predictions_survive_loading(self):
        import numpy as np
        from model.baseline.lda import SklearnLDA
        bow = np.array([[5, 3, 0, 0], [2, 4, 0, 0], [0, 0, 5, 3], [0, 0, 3, 4]])
        model = SklearnLDA(vocab_size=4, num_topics=2, max_iter=2)
        model.fit(bow)
        with tempfile.TemporaryDirectory() as folder:
            save_trained_model(model, folder, 'lda', ['苹果', '水果', '代码', '算法'])
            restored = load_trained_model(folder, trusted=True)
            np.testing.assert_allclose(restored.get_beta(), model.get_beta())
            np.testing.assert_allclose(restored.transform(bow), model.transform(bow))


if __name__ == '__main__': unittest.main()
