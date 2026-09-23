"""Read-only numerical/artifact audit of reports produced through Agent conversation.

Usage: agent/.venv/bin/python agent/scripts/audit-research-results.py manifest.json ...
Outputs JSON; does not train, generate plots, or alter saved evidence.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def digest(file):
    with Path(file).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(manifest):
    report = json.loads(Path(manifest).read_text())
    assert report['reportStatus'] == 'complete', manifest
    files = report['files']
    for entry in files:
        assert digest(entry['path']) == entry['sha256'], entry['name']
    def matrix(prefix):
        return np.load(next(entry['path'] for entry in files if Path(entry['name']).name.startswith(prefix) and entry['name'].endswith('.npy')), allow_pickle=False)
    theta, beta = matrix('theta_k'), matrix('beta_k')
    assert np.isfinite(theta).all() and np.isfinite(beta).all()
    assert (theta >= 0).all() and (beta >= 0).all()
    assert theta.shape[1] == beta.shape[0] == report['summary']['topicCount']
    assert theta.shape[0] == report['summary']['documentCount']
    np.testing.assert_allclose(beta.sum(axis=1), 1, atol=1e-5)
    assert report['sourceData']['matrixRowsAligned']
    model = report['modelId']
    result = {'model': model, 'jobId': report['jobId'], 'manifest': str(Path(manifest).resolve()),
              'report': report['reportPath'], 'filesVerified': len(files), 'thetaShape': list(theta.shape),
              'betaShape': list(beta.shape), 'meanRawTheta': theta.mean(axis=0).tolist(),
              'thetaStd': theta.std(axis=0).tolist(), 'betaMaxPairwiseDifference': float(np.max(np.ptp(beta, axis=0))),
              'sourceRowsAligned': True, 'trainingPlan': report.get('trainingPlan'),
              'quality': report['quality'], 'missing': report['missingEvidence'],
              'tables': {table['relativePath']: table for table in report['evidence']['tables']},
              'figureFamilies': sorted({str(Path(item['name']).with_suffix('')) for item in files if item['kind'] == 'figure'}),
              'metrics': [item['content'] for item in report['evidence']['evidence'] if item['kind'] == 'metrics']}
    if model == 'bertopic':
        labels = matrix('document_topics')
        assert len(labels) == len(theta)
        unique, counts = np.unique(labels, return_counts=True)
        result['assignedCounts'] = {str(int(k)): int(n) for k, n in zip(unique, counts)}
        keep = (labels >= 0) & (theta.sum(axis=1) > 0)
        result['conditionalMeanTheta'] = (theta[keep] / theta[keep].sum(axis=1, keepdims=True)).mean(axis=0).tolist()
    else:
        np.testing.assert_allclose(theta.sum(axis=1), 1, atol=1e-5)
    if model == 'dtm':
        temporal_beta = matrix('beta_over_time')
        np.testing.assert_allclose(temporal_beta[-1], beta)
        np.testing.assert_allclose(temporal_beta.sum(axis=2), 1, atol=1e-5)
        workspace = next(Path(p).parent for p in report['inputSignatures'] if p.endswith('/time_indices.npy'))
        indices = np.load(workspace / 'time_indices.npy', allow_pickle=False)
        source_rows = np.load(workspace / 'source_rows.npy', allow_pickle=False)
        source = next(p for p in report['inputSignatures'] if '/uploads/' in p and p.endswith('/data.csv'))
        with open(source, newline='') as handle:
            rows = list(csv.DictReader(handle))
        labels = sorted({int(row['year']) for row in rows})
        np.testing.assert_array_equal([labels[int(t)] for t in indices], [int(rows[int(i)]['year']) for i in source_rows])
        result['temporalBetaShape'] = list(temporal_beta.shape)
        result['allTemporalRowsVerified'] = len(indices)
        result['periods'] = labels
    if model == 'stm':
        result['covariateEncoding'] = report['evidence'].get('covariateEncoding')
        result['coefficients'] = [item['content'] for item in report['evidence']['evidence'] if 'covariate_effects' in item['relativePath']]
        assert not any('anova' in item['name'].lower() for item in files), 'Do not deliver post-fit ANOVA as effect evidence'
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(json.dumps([audit(manifest) for manifest in sys.argv[1:]], ensure_ascii=False, indent=2))
