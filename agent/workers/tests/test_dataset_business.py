import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from workers.dataset.catalog import discover, use
from workers.dataset.business import understand


class DatasetBusinessTests(unittest.TestCase):
    def test_discovery_selection_and_content_understanding_preserve_source(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {'THETA_DATA_DIR': home}):
            root = Path(home)
            source = root / '客服反馈.csv'
            original = 'text,channel\n物流延迟三天希望改善配送，联系13812345678,售后\n退款处理等待时间长，foo@example.com,售后\n咨询如何申请退款,咨询\n'
            source.write_text(original)
            (root / 'README.txt').write_text('this is also a supported text dataset')
            (root / '.private.csv').write_text('hidden')
            (root / 'models').mkdir()
            (root / 'models/ignored.csv').write_text('generated')
            (root / 'outside.csv').symlink_to('/etc/passwd')
            found = discover({})
            self.assertEqual(len(found['datasets']), 1)
            entry = next(item for item in found['datasets'] if item['name'] == source.name)
            self.assertNotIn('物流', str(found))
            dataset = use({'catalogId': entry['catalogId'], 'uploadDir': str(root / '.uploads')})
            context = understand({'dataset': dataset})
            self.assertEqual(context['documentCount'], 3)
            self.assertEqual(context['textColumn'], 'text')
            self.assertIn('物流延迟', str(context['excerpts']))
            self.assertNotIn('13812345678', str(context))
            self.assertNotIn('foo@example.com', str(context))
            self.assertEqual(context['segments'][0]['column'], 'channel')
            self.assertEqual(source.read_text(), original)
            source.write_text(original + '新反馈,售后\n')
            with self.assertRaisesRegex(ValueError, '变化'):
                use({'catalogId': entry['catalogId'], 'uploadDir': str(root / '.uploads')})
            with self.assertRaises(ValueError):
                use({'catalogId': '../../etc/passwd', 'uploadDir': str(root / '.uploads')})

    def test_empty_catalog_is_explicit(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {'THETA_DATA_DIR': home}):
            self.assertEqual(discover({})['datasets'], [])
