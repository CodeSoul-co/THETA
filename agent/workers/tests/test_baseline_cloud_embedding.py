import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src/models'))
from workers.execution_policy import execution_policy, training_environment
from workers.capabilities import runtime_check

class BaselineCloudEmbeddingTest(unittest.TestCase):
    def test_baselines_do_not_require_local_weights_when_cloud_is_selected(self):
        with patch.dict(os.environ,{'EMBEDDING_API_BASE':'https://example.invalid/v1','EMBEDDING_API_KEY':'fixture','EMBEDDING_API_KEY_ENV':'EMBEDDING_API_KEY','EMBEDDING_MODEL':'test'},clear=True):
            for model in ['ctm','bertopic']:
                plan={'modelId':model,'params':{'embedding_provider':'cloud','embedding.model_path':'/missing/unused-local-weight'},'timeoutSeconds':60,'externalRequestLimit':2}
                policy=execution_policy(plan)
                self.assertEqual(policy['embedding']['mode'],'cloud')
                self.assertEqual(training_environment(policy)['EMBEDDING_PROVIDER'],'cloud')
                result=runtime_check({'modelId':model,'embeddingProvider':'cloud'})
                self.assertTrue(result['ready'],result['issues'])
                self.assertIsNone(result['requiredAssetVariable'])

    def test_cloud_baseline_preparation_writes_real_provider_matrix(self):
        import prepare_data
        from model import embedding_providers
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ,{'EMBEDDING_PROVIDER':'cloud','EMBEDDING_API_KEY':'fixture'}):
            matrix=np.array([[1,2,3],[4,5,6]],dtype=np.float32)
            with patch.object(embedding_providers.OpenAICompatibleEmbeddingProvider,'embed',return_value=matrix) as embed:
                result=prepare_data.generate_sbert_embeddings(['first text','second text'],Path(home),batch_size=2)
            embed.assert_called_once_with(['first text','second text'],batch_size=2)
            np.testing.assert_array_equal(result,matrix)
            np.testing.assert_array_equal(np.load(Path(home)/'sbert_embeddings.npy'),matrix)
