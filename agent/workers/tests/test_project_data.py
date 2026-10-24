"""Filesystem preuploads use the same project boundary as managed uploads."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from workers.dataset.catalog import discover, use

class ProjectDataTests(unittest.TestCase):
    def test_catalog_ids_and_imports_are_project_scoped(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {'THETA_DATA_DIR': home}):
            root = Path(home).resolve()
            (root / 'global.txt').write_text('global private data')
            for project in ['a', 'b']:
                (root / 'projects' / project).mkdir(parents=True)
            a = root / 'projects/a/source.txt'
            a.write_text('first document\nsecond document')
            shutil.copy2(a, root / 'projects/b/source.txt')
            first = discover({'projectId': 'a'})['datasets']
            second = discover({'projectId': 'b'})['datasets']
            self.assertEqual([f['name'] for f in first], ['source.txt'])
            self.assertNotEqual(first[0]['catalogId'], second[0]['catalogId'])
            with self.assertRaises(ValueError):
                use({'projectId': 'a', 'catalogId': second[0]['catalogId'], 'uploadDir': str(root / 'managed/a')})
            imported = use({'projectId': 'a', 'catalogId': first[0]['catalogId'], 'uploadDir': str(root / 'managed/a')})
            self.assertTrue(Path(imported['managedPath']).is_relative_to(root / 'managed/a'))
            with self.assertRaises(ValueError):
                discover({'projectId': '../b'})
            (root / 'projects/alias').symlink_to(root / 'projects/b', target_is_directory=True)
            with self.assertRaises(ValueError):
                discover({'projectId': 'alias'})
