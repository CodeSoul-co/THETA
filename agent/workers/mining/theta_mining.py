"""General post-model research primitives. No task IDs, reference labels or judge.

Topic weights are discovery evidence; semantic assignments must be supplied by
the researcher and reviewed against original text. All results are descriptive.
"""
from collections import Counter
import hashlib
import math


def _index(rows, id_key):
    ids = [str(row[id_key]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Document IDs must be unique')
    return dict(zip(ids, rows))


def topic_probe(rows, weights, row_ids, topic, *, id_key='doc_id', text_key='text', k=5):
    """Return high-score, ambiguous and low-score originals for one model axis.

    row_ids must come from verified preprocessing provenance, NOT a guessed join.
    Structural checks here cannot independently establish that provenance.
    Raw scores are not interpreted as probabilities (latent axes are supported).
    """
    import numpy as np
    index = _index(rows, id_key)
    matrix = np.asarray(weights, dtype=float)
    ids = [str(value) for value in row_ids]
    if matrix.ndim != 2 or len(matrix) != len(ids) or len(set(ids)) != len(ids) or not set(ids) <= set(index):
        raise ValueError('Matrix rows must map one-to-one to existing document IDs')
    if not np.isfinite(matrix).all() or not 0 <= topic < matrix.shape[1] or not 1 <= k <= 30:
        raise ValueError('Invalid finite matrix, topic index or sample size')
    scores = matrix[:, topic]
    others = np.max(np.delete(matrix, topic, axis=1), axis=1) if matrix.shape[1] > 1 else scores
    margin = scores - others
    def select(order):
        return [{'doc_id': ids[int(i)], 'model_row': int(i), 'score': float(scores[i]),
                 'margin_to_other_axis': float(margin[i]), 'text': index[ids[int(i)]][text_key]}
                for i in order[:k]]
    return {'topic_index': topic, 'model_rows': len(ids), 'corpus_rows': len(rows),
            'unmodeled_doc_ids': sorted(set(index) - set(ids)),
            'representative': select(np.argsort(-scores, kind='stable')),
            'boundary': select(np.argsort(abs(margin), kind='stable')),
            'contrast': select(np.argsort(scores, kind='stable')),
            'limitations': ['Purposive samples, not prevalence estimates or semantic labels.',
                           'Row-ID provenance is caller-supplied; structural consistency alone does not verify semantic alignment.',
                           'Axis scores/margins are model-dependent, not confidence in a finding.']}


def partition(rows, labels, *, id_key='doc_id'):
    """Validate exhaustive positive/negative/unknown labels. Never fill omissions."""
    index = _index(rows, id_key)
    if set(labels) != set(index):
        raise ValueError('Provide exactly one label per selected document; omissions are not negatives')
    result = {label: [] for label in ('positive', 'negative', 'unknown')}
    for identifier, label in labels.items():
        if label not in result:
            raise ValueError('Labels must be positive, negative or unknown')
        result[label].append(identifier)
    return {label: sorted(ids) for label, ids in result.items()}


def audit_sample(rows, labels, *, per_state=12, seed='semantic-boundary-audit', id_key='doc_id', text_key='text'):
    """Reproducible stratified raw-text panel for checking a proposed definition.

    This supplies NO semantic truth. The analyst must check raw text against the
    definition, including false positives and missed positives. Do not repeatedly
    change the seed to select agreeable cases or call this independent validation.
    """
    groups=partition(rows,labels,id_key=id_key)
    if isinstance(per_state,bool) or not isinstance(per_state,int) or not 1<=per_state<=100:
        raise ValueError('per_state must be an integer from 1 to 100')
    index=_index(rows,id_key)
    sampled=[]
    for state, ids in groups.items():
        ordered=sorted(ids,key=lambda i:(hashlib.sha256((str(seed)+'\0'+i).encode()).hexdigest(),i))
        sampled.extend({'doc_id':i,'assigned_state':state,'text':index[i][text_key]} for i in ordered[:per_state])
    return {'seed':str(seed),'per_state':per_state,'population_n':len(rows),
            'state_population':{state:len(ids) for state,ids in groups.items()},'cases':sampled,
            'limitations':['Stratified purposive-size sample; unweighted pooled agreement is not population accuracy.',
                'Assignments and subsequent analyst judgments are not ground truth or independent certification.',
                'Freeze the definition and panel before inspection; disclose any revision and use fresh cases for another check.']}


def audit_disagreements(panel, judgments):
    """Summarize explicit analyst judgments without silently accepting omissions."""
    cases=panel['cases']
    partition(cases,judgments)
    summary={}
    for state in ('positive','negative','unknown'):
        selected=[case for case in cases if case['assigned_state']==state]
        summary[state]={'checked':len(selected),'judged_states':dict(Counter(judgments[case['doc_id']] for case in selected)),
            'disagreement_doc_ids':[case['doc_id'] for case in selected if judgments[case['doc_id']]!=state]}
    return {'by_assigned_state':summary,'population_n':panel['population_n'],
            'interpretation':'Disagreement with supplied analyst judgments, not certified semantic accuracy. Revisit definitions and bound unresolved cases before making a population claim.'}


def _counts(ids, labels):
    counts = Counter(labels[i] for i in ids)
    total = len(ids)
    positive, negative, unknown = (counts[name] for name in ('positive', 'negative', 'unknown'))
    known = positive + negative
    return {'total': total, 'positive': positive, 'negative': negative, 'unknown': unknown, 'known': known,
            'rate_known': positive / known if known else None,
            'rate_bounds_all': [positive / total, (positive + unknown) / total] if total else [None, None]}


def _comparison(left, right, labels):
    a, b = _counts(left, labels), _counts(right, labels)
    difference = b['rate_known'] - a['rate_known'] if a['known'] and b['known'] else None
    bounds = [b['rate_bounds_all'][0] - a['rate_bounds_all'][1],
              b['rate_bounds_all'][1] - a['rate_bounds_all'][0]] if a['total'] and b['total'] else [None, None]
    return {'arm0': a, 'arm1': b, 'difference': difference, 'difference_unknown_bounds': bounds}


def contrast(rows, labels, *, group_field, arms, stratify=(), min_known=5, id_key='doc_id'):
    """Two disjoint metadata arms; known means positive PLUS negative.

    Returns raw contrasts, comparable strata, largest-stratum removal and missing
    metadata. Effects are descriptive; the population and concepts are chosen by
    the caller. Does not optimize definitions, select a winning finding or assert
    causal identification. Missing strata are never silently imputed.
    """
    partition(rows, labels, id_key=id_key)
    if len(arms) != 2 or not all(arms) or any(value is None for arm in arms for value in arm) or set(arms[0]) & set(arms[1]):
        raise ValueError('Provide two nonempty, disjoint metadata arms without null')
    if not isinstance(min_known, int) or isinstance(min_known, bool) or min_known < 1:
        raise ValueError('min_known must be a positive integer')
    index = _index(rows, id_key)
    groups = [{i for i, r in index.items() if r.get(group_field) in arm} for arm in arms]
    selected = groups[0] | groups[1]
    result = _comparison(*groups, labels)
    result.update({'group_field': group_field, 'arms': arms, 'population_n': len(rows),
                   'excluded_group_metadata_n': len(rows) - len(selected), 'stratified': {}})
    for field in stratify:
        buckets = {}
        missing = set()
        for i in sorted(selected):
            value = index[i].get(field)
            if value is None or value == '':
                missing.add(i)
            else:
                # Preserve metadata value types (the integer 1 is not the string "1").
                key = (type(value).__name__, str(value))
                buckets.setdefault(key, set()).add(i)
        details = []
        covered = set()
        for key, ids in sorted(buckets.items()):
            left, right = ids & groups[0], ids & groups[1]
            comparison = _comparison(left, right, labels)
            eligible = min(comparison['arm0']['known'], comparison['arm1']['known']) >= min_known
            if eligible:
                covered |= ids
            details.append({'value_type': key[0], 'value': key[1], 'eligible': eligible, **comparison})
        eligible = [d for d in details if d['eligible']]
        denominator = sum(d['arm0']['known'] + d['arm1']['known'] for d in eligible)
        weighted = sum(d['difference'] * (d['arm0']['known'] + d['arm1']['known']) for d in eligible) / denominator if denominator else None
        largest = max(sorted(buckets), key=lambda key: len(buckets[key]), default=None)
        # Missing strata are excluded from this sensitivity contrast and counted.
        retained = selected - missing - (buckets[largest] if largest is not None else set())
        removal = _comparison(groups[0] & retained, groups[1] & retained, labels)
        removal_eligible = largest is not None and min(removal['arm0']['known'], removal['arm1']['known']) >= min_known
        positive_known_metadata = {i for i in selected - missing if labels[i] == 'positive'}
        concentration = max((len(ids & positive_known_metadata) for ids in buckets.values()), default=0)
        result['stratified'][field] = {'strata': details, 'missing_metadata_n': len(missing),
            'comparable_n': len(eligible), 'covered_documents_n': len(covered),
            'weighted_difference': weighted,
            'raw_stratified_sign_reversal': (result['difference'] * weighted < 0) if weighted is not None and result['difference'] is not None else None,
            'largest_stratum': list(largest) if largest else None,
            'largest_stratum_removal': {**removal, 'eligible': removal_eligible},
            'max_positive_support_share': concentration / len(positive_known_metadata) if positive_known_metadata else None}
    result['limitations'] = ['Exploratory descriptive association, not independent confirmation or causality.',
        'Uncertainty bounds address unknown labels, not all sources of label error or sampling uncertainty.',
        'Stratified average uses known-document weights only over comparable strata; compare its coverage before interpreting changes.']
    return result


def evidence_spans(rows, selections, *, id_key='doc_id', text_key='text'):
    """Verify exact source quotations supplied by the analyst; no invented spans."""
    index = _index(rows, id_key)
    verified = []
    for item in selections:
        identifier, quote = str(item['doc_id']), item['quote']
        if identifier not in index or not isinstance(quote, str) or not quote:
            raise ValueError('Evidence requires an existing ID and nonempty exact quote')
        text = index[identifier][text_key]
        start = text.find(quote)
        if start < 0:
            raise ValueError('Quotation does not occur verbatim in the original document')
        verified.append({**item, 'doc_id': identifier, 'start': start, 'end': start + len(quote)})
    return verified
