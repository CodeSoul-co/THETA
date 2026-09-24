"""Numeric-only progress shared by embedding workers and the local workbench."""
import json


def report_embedding(source, scope, current, total, batches, chunks=None, chunk_total=None, batch_total=None):
    event = dict(source=source, scope=scope, current=current, total=total, completedBatches=batches)
    if chunks is not None:
        event['chunks'] = chunks
    if chunk_total is not None:
        event['chunkTotal'] = chunk_total
    if batch_total is not None:
        event['totalBatches'] = batch_total
    print('THETA_EMBEDDING ' + json.dumps(event), flush=True)
