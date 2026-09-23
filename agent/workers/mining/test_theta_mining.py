import unittest
from theta_mining import contrast, evidence_spans, partition, topic_probe, audit_sample, audit_disagreements


class MiningTests(unittest.TestCase):
    def test_semantic_audit_samples_all_states_without_order_bias_or_invented_truth(self):
        rows=[{'doc_id':str(i),'text':f'Raw example {i}'} for i in range(15)]
        labels={str(i):('positive','negative','unknown')[i//5] for i in range(15)}
        panel=audit_sample(rows,labels,per_state=2)
        self.assertEqual(panel,audit_sample(rows[::-1],labels,per_state=2))
        self.assertEqual(len(panel['cases']),6)
        judgments={case['doc_id']:case['assigned_state'] for case in panel['cases']}
        first=panel['cases'][0]['doc_id'];judgments[first]='negative'
        report=audit_disagreements(panel,judgments)
        self.assertEqual(report['by_assigned_state']['positive']['disagreement_doc_ids'],[first])
        with self.assertRaises(ValueError):audit_disagreements(panel,{})
        with self.assertRaises(ValueError):audit_sample(rows,labels,per_state=True)

    def test_exhaustive_partition(self):
        rows = [{'doc_id': str(i)} for i in range(3)]
        self.assertEqual(partition(rows, {'0': 'positive', '1': 'negative', '2': 'unknown'})['unknown'], ['2'])
        for labels in ({'0': 'positive'}, {'0': 'positive', '1': 'negative', '2': False}):
            with self.assertRaises(ValueError):
                partition(rows, labels)

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):
            partition([{'doc_id': 'a'}, {'doc_id': 'a'}], {'a': 'positive'})

    def test_known_includes_negatives_and_unknown_bounds(self):
        rows = [{'doc_id': str(i), 'arm': i // 10, 'source': 'x'} for i in range(20)]
        labels = {str(i): 'positive' if i in (0, 10) else 'negative' for i in range(20)}
        labels['11'] = 'unknown'
        result = contrast(rows, labels, group_field='arm', arms=[[0], [1]], stratify=['source'])
        self.assertEqual(result['arm1']['known'], 9)
        self.assertEqual(result['arm1']['rate_bounds_all'], [0.1, 0.2])
        self.assertTrue(result['stratified']['source']['strata'][0]['eligible'])
        self.assertAlmostEqual(result['difference_unknown_bounds'][1], 0.1)

    def test_null_rate_and_no_comparable_strata(self):
        rows = [{'doc_id': 'a', 'arm': 0}, {'doc_id': 'b', 'arm': 1}]
        result = contrast(rows, {'a':'unknown', 'b':'unknown'}, group_field='arm', arms=[[0],[1]], stratify=['source'])
        self.assertIsNone(result['difference'])
        self.assertIsNone(result['stratified']['source']['weighted_difference'])
        self.assertEqual(result['stratified']['source']['missing_metadata_n'], 2)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            contrast([], {}, group_field='arm', arms=[[0], [0]])

    def test_simpson_reversal_and_removal(self):
        rows, labels = [], {}
        for source, arm, n, p in [('easy',0,10,9),('easy',1,100,80),('hard',0,100,30),('hard',1,10,2)]:
            for j in range(n):
                identifier = str(len(rows))
                rows.append({'doc_id':identifier, 'arm':arm, 'source':source})
                labels[identifier] = 'positive' if j < p else 'negative'
        result = contrast(rows, labels, group_field='arm', arms=[[0],[1]], stratify=['source'])
        self.assertGreater(result['difference'], 0)
        self.assertTrue(result['stratified']['source']['raw_stratified_sign_reversal'])
        self.assertTrue(result['stratified']['source']['largest_stratum_removal']['eligible'])

    def test_exact_evidence(self):
        rows = [{'doc_id':'a', 'text':'Not a real refund.'}]
        self.assertEqual(evidence_spans(rows, [{'doc_id':'a','quote':'real refund'}])[0]['start'], 6)
        with self.assertRaises(ValueError):
            evidence_spans(rows, [{'doc_id':'a','quote':'received refund'}])

    def test_topic_rows_and_boundary(self):
        rows = [{'doc_id':str(i), 'text':str(i)} for i in range(4)]
        result = topic_probe(rows, [[.9,.1],[.49,.51],[.1,.9]], ['2','0','1'], 0, k=1)
        self.assertEqual(result['representative'][0]['doc_id'], '2')
        self.assertEqual(result['boundary'][0]['doc_id'], '0')
        self.assertEqual(result['unmodeled_doc_ids'], ['3'])
        with self.assertRaises(ValueError):
            topic_probe(rows, [[.9,.1]], ['0','1'], 0)

    def test_latent_scores_preserved(self):
        result = topic_probe([{'doc_id':'a','text':'a'}], [[-2,3]], ['a'], 0)
        self.assertEqual(result['representative'][0]['score'], -2)


if __name__ == '__main__':
    unittest.main()
