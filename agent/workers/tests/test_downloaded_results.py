import json
import tempfile
import unittest
from pathlib import Path
from workers.downloaded_results import receive
from workers.local_compute import results
from workers.capabilities import dataset_import
from workers.analysis_report import evidence
from workers.results_reader import file_hash


class DownloadedResultsTests(unittest.TestCase):
    def test_remote_delivery_supports_local_reading_reports_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            job_id = 'job-' + 'a' * 64
            root = home / 'compute' / job_id / 'delivery'
            (root / 'native').mkdir(parents=True)
            source = home / 'source.csv'
            source.write_text('text,score\nresearch alpha,1\nresearch beta,2\nresearch gamma,3\nresearch delta,4')
            dataset = dataset_import({'filePath': str(source), 'uploadDir': str(home / 'uploads')})
            splits = {'groups': {role: {'count': count} for role, count in [('all', 4), ('train', 2), ('validation', 1), ('test', 1)]}}
            (root / 'split_results.json').write_text(json.dumps(splits))
            (root / 'index.html').write_text('<!doctype html><title>report</title>')
            (root / 'native' / 'topic_proportions.csv').write_text('topic_id,proportion\n1,0.25\n2,0.75\n')
            files = [{'name': f.relative_to(root).as_posix(), 'path': str(f), 'sha256': file_hash(f), 'sizeBytes': f.stat().st_size, 'kind': 'table' if f.suffix == '.csv' else 'artifact'} for f in root.rglob('*') if f.is_file()]
            request = {'jobId': job_id, 'dataset': dataset, 'plan': {'modelId': 'lda', 'textColumn': 'text', 'params': {}}}
            report = receive({'home': folder, 'request': request, 'report': {'resultDir': str(root), 'reportPath': str(root / 'index.html'), 'files': files, 'evidence': {'tables': [], 'figures': [], 'matrices': [], 'evidence': []}}})
            self.assertEqual(results({'home': folder, 'jobId': job_id, 'view': 'report'}), report)
            directory = home / 'analysis'; directory.mkdir()
            details = evidence({'home': folder, 'jobId': job_id, 'directory': str(directory)})
            self.assertEqual(details['splits']['groups']['all']['count'], 4)
            self.assertIn('原始记录数 | 4', details['description'])
            self.assertEqual(len(details['catalog']), 1)
            (root / 'native' / 'topic_proportions.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError, '已变化'):
                results({'home': folder, 'jobId': job_id, 'view': 'summary'})
