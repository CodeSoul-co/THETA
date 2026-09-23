"""Bounded, batched long-document encoding for decoder embedding models."""
import numpy as np
import torch


def last_token_pool(hidden, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = (positions * attention_mask).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def encode_documents(texts, tokenizer, model, device, max_length=512, batch_size=32):
    """Keep every long-document window, then average its last-token embeddings.

    batch_size bounds the number of windows in one forward pass, not documents.
    No document tail is silently discarded. Progress is emitted for real batches.
    """
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity < 10 or batch_size < 1:
        raise ValueError('编码长度或批大小无效')
    stride = max(1, max_length // 2)
    sums, counts = [None] * len(texts), np.zeros(len(texts), dtype=np.int64)
    pending, owners = [], []
    completed_windows = 0

    def flush():
        nonlocal completed_windows
        if not pending:
            return
        inputs = tokenizer.pad(pending, padding=True, return_tensors='pt').to(device)
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        vectors = last_token_pool(hidden, inputs['attention_mask']).float().cpu().numpy()
        for owner, vector in zip(owners, vectors):
            sums[owner] = vector.copy() if sums[owner] is None else sums[owner] + vector
            counts[owner] += 1
        completed_windows += len(pending)
        print(f'  本地嵌入：已编码 {completed_windows} 个文本块，当前文档 {owners[-1] + 1}/{len(texts)}', flush=True)
        pending.clear()
        owners.clear()

    with torch.inference_mode():
        for owner, text in enumerate(texts):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, max(1, len(tokens)), stride):
                end = min(start + capacity, len(tokens))
                chunk = tokens[start:end]
                pending.append(tokenizer.prepare_for_model(chunk, add_special_tokens=True, return_attention_mask=True))
                owners.append(owner)
                if len(pending) >= batch_size:
                    flush()
                if end >= len(tokens):
                    break
        flush()
    return np.vstack([total / count for total, count in zip(sums, counts)])
