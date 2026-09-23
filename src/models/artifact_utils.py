"""Dependency-light helpers for atomic artifact publication and signatures."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Tuple


BOW_MANIFEST_NAME = "bow_manifest.json"
BOW_SCHEMA = "theta.bow-artifacts"
BOW_VERSION = 1
BOW_ARTIFACT_FILENAMES = {
    "matrix": "bow_matrix.npz",
    "vocab_embeddings": "vocab_embeddings.npy",
    "vocab": "vocab.txt",
}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(data) -> bytes:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def manifest_signature(manifest: Mapping) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "signature"
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validate_metadata(role: str, entry: Mapping) -> None:
    if role in {"matrix", "vocab_embeddings"}:
        shape = entry.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or not all(isinstance(value, int) and value >= 0 for value in shape)
        ):
            raise ValueError(f"invalid {role} shape metadata")
    if role == "matrix" and entry.get("format") != "scipy_csr_npz":
        raise ValueError("invalid matrix format metadata")
    if role == "vocab_embeddings" and not isinstance(
        entry.get("dtype"), str
    ):
        raise ValueError("invalid vocab embedding dtype metadata")
    if role == "vocab":
        if not isinstance(entry.get("count"), int) or entry["count"] < 0:
            raise ValueError("invalid vocab count metadata")
        if entry.get("encoding") != "utf-8":
            raise ValueError("invalid vocab encoding metadata")


def _validate_cross_component_metadata(artifacts: Mapping) -> None:
    vocab_count = artifacts["vocab"]["count"]
    matrix_shape = artifacts["matrix"]["shape"]
    embeddings_shape = artifacts["vocab_embeddings"]["shape"]
    if matrix_shape[1] != vocab_count or embeddings_shape[0] != vocab_count:
        raise ValueError(
            "BOW artifact vocab dimensions do not match"
        )


def commit_bow_artifacts(
    directory,
    temporary_paths: Mapping[str, Path],
    metadata: Mapping[str, Mapping],
) -> Dict:
    """Publish all BOW components and commit them with a manifest last."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    required_roles = set(BOW_ARTIFACT_FILENAMES)
    if set(temporary_paths) != required_roles or set(metadata) != required_roles:
        raise ValueError("BOW artifact roles are incomplete")

    artifacts = {}
    for role, filename in BOW_ARTIFACT_FILENAMES.items():
        temp_path = Path(temporary_paths[role])
        if not temp_path.is_file():
            raise ValueError(f"temporary {role} artifact is missing")
        if temp_path.parent.resolve() != directory.resolve():
            raise ValueError("temporary artifacts must share the target directory")
        with open(temp_path, "rb") as file_obj:
            os.fsync(file_obj.fileno())
        entry = {
            "filename": filename,
            "sha256": sha256_file(temp_path),
            **dict(metadata[role]),
        }
        _validate_metadata(role, entry)
        artifacts[role] = entry
    _validate_cross_component_metadata(artifacts)

    manifest = {
        "schema": BOW_SCHEMA,
        "version": BOW_VERSION,
        "artifacts": artifacts,
    }
    manifest["signature"] = manifest_signature(manifest)
    manifest_path = directory / BOW_MANIFEST_NAME

    if manifest_path.exists():
        os.remove(manifest_path)

    for role, filename in BOW_ARTIFACT_FILENAMES.items():
        os.replace(temporary_paths[role], directory / filename)

    legacy_dense_path = directory / "bow_matrix.npy"
    if legacy_dense_path.exists():
        os.remove(legacy_dense_path)

    descriptor, temp_manifest_name = tempfile.mkstemp(
        prefix=".bow_manifest.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(manifest, file_obj, sort_keys=True, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_manifest_name, manifest_path)
    finally:
        if os.path.exists(temp_manifest_name):
            os.remove(temp_manifest_name)

    return manifest


def validate_bow_manifest(directory) -> Tuple[Dict, str]:
    """Validate the committed BOW set and return its manifest/signature."""
    directory = Path(directory)
    manifest_path = directory / BOW_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError("BOW manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid BOW manifest: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("invalid BOW manifest root")
    if manifest.get("schema") != BOW_SCHEMA:
        raise ValueError("invalid BOW manifest schema")
    if manifest.get("version") != BOW_VERSION:
        raise ValueError("invalid BOW manifest version")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != set(BOW_ARTIFACT_FILENAMES)
    ):
        raise ValueError("BOW manifest artifact set is incomplete")

    expected_signature = manifest_signature(manifest)
    if manifest.get("signature") != expected_signature:
        raise ValueError("BOW manifest signature mismatch")

    for role, expected_filename in BOW_ARTIFACT_FILENAMES.items():
        entry = artifacts[role]
        if not isinstance(entry, dict):
            raise ValueError(f"invalid {role} manifest entry")
        if entry.get("filename") != expected_filename:
            raise ValueError(f"invalid {role} filename")
        _validate_metadata(role, entry)
        artifact_path = directory / expected_filename
        if not artifact_path.is_file():
            raise ValueError(f"{role} artifact is missing")
        if sha256_file(artifact_path) != entry.get("sha256"):
            raise ValueError(f"{role} checksum mismatch")
    _validate_cross_component_metadata(artifacts)

    return manifest, expected_signature


def has_valid_bow_manifest(directory) -> bool:
    try:
        validate_bow_manifest(directory)
        return True
    except ValueError:
        return False


def file_set_signature(directory, filenames, schema: str) -> Dict:
    directory = Path(directory)
    files = {}
    for filename in sorted(filenames):
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"required file is missing: {filename}")
        files[filename] = sha256_file(path)
    payload = {
        "schema": schema,
        "version": 1,
        "files": files,
    }
    payload["signature"] = manifest_signature(payload)
    return payload


def adapter_signature(directory) -> Dict:
    directory = Path(directory)
    config_filename = "adapter_config.json"
    weight_candidates = (
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    weight_filename = next(
        (
            filename
            for filename in weight_candidates
            if (directory / filename).is_file()
        ),
        None,
    )
    if weight_filename is None:
        raise ValueError("LoRA adapter weights are missing")
    return file_set_signature(
        directory,
        (config_filename, weight_filename),
        "theta.lora-adapter",
    )


def find_topic_matrix_pair(directory, num_topics=None):
    """Resolve one experiment's original pair, including inferred K and nested exports."""
    import re
    root = Path(directory)
    pairs = []
    for theta in root.rglob('theta*.npy'):
        if theta.name != 'theta.npy' and not re.fullmatch(r'theta_k\d+\.npy', theta.name):
            continue
        beta = theta.with_name(theta.name.replace('theta', 'beta', 1))
        if beta.is_file(): pairs.append((theta, beta))
    exact = [pair for pair in pairs if pair[0].name == f'theta_k{num_topics}.npy'] if num_topics is not None else []
    selected = exact or pairs
    if len(selected) != 1:
        raise ValueError(f'Expected one theta/beta pair in {root}; found {len(selected)}. Select a single experiment.')
    return selected[0]


def validate_topic_matrices(theta, beta, vocab, model):
    import numpy as np
    if theta.ndim != 2 or beta.ndim != 2 or not theta.size or not beta.size:
        raise ValueError('Empty or non-matrix topic results')
    if theta.shape[1] != beta.shape[0] or beta.shape[1] != len(vocab):
        raise ValueError('theta/beta/vocabulary axes do not match')
    if not np.isfinite(theta).all() or not np.isfinite(beta).all() or (beta < 0).any():
        raise ValueError('Non-finite or negative topic-word results')
    if model != 'nvdm' and (theta < 0).any(): raise ValueError('Negative topic probabilities')
    if (beta.sum(axis=1) <= 0).any(): raise ValueError('Topics have empty word distributions')


def export_baseline_metadata(directory, model_name, result, trainer):
    """Complete existing native exports, without recomputing theta or beta or fitting."""
    import numpy as np
    from scipy import sparse
    root = Path(directory)
    theta_file, beta_file = find_topic_matrix_pair(root)
    output = theta_file.parent
    model = result['model']
    vocab = list(model._beta_vocab) if model_name == 'bertopic' else list(trainer.vocab)
    validate_topic_matrices(result['theta'], result['beta'], vocab, model_name)
    (output / 'vocab.json').write_text(json.dumps(vocab, ensure_ascii=False), encoding='utf-8')
    if model_name == 'bertopic':
        # The already-fitted BERTopic vectorizer defines its vocabulary; never use the BOW worker's other vocabulary.
        vectorizer = model.model.vectorizer_model
        counts = vectorizer.transform(trainer.texts)
        columns = [vectorizer.vocabulary_[word] for word in vocab]
        sparse.save_npz(output / 'bow_matrix.npz', counts[:, columns].tocsr())
        np.save(output / 'document_topics.npy', np.asarray(model.get_topics(), dtype=np.int64))
    history = result.get('training_history')
    if history and not list(output.glob('training_history*.json')):
        keys = list(dict.fromkeys(key for row in history for key in row if key != 'epoch'))
        history = {('train_loss' if key == 'loss' else key): [row.get(key) for row in history] for key in keys}
        (output / 'training_history.json').write_text(json.dumps(history, allow_nan=False), encoding='utf-8')
    if not list(output.glob('topic_words*.json')):
        words = {str(k): [vocab[i] for i in np.argsort(-result['beta'][k])[:20]] for k in range(result['beta'].shape[0])}
        (output / 'topic_words.json').write_text(json.dumps(words, ensure_ascii=False), encoding='utf-8')
    # Keep an actual reusable model checkpoint for every model, including traditional baselines.
    if not any(output.glob('*.pt')) and not any(output.glob('*.pkl')) and not any(output.glob('*.joblib')):
        if hasattr(model, 'save_model'): model.save_model(str(output / 'model.pt'))
        else:
            import joblib
            joblib.dump(model, output / 'model.joblib')
    return write_topic_result_manifest(root, model_name, result['theta'], result['beta'], vocab)


def write_topic_result_manifest(root, model_name, theta, beta, vocab):
    root = Path(root)
    validate_topic_matrices(theta, beta, vocab, model_name)
    theta_file, beta_file = find_topic_matrix_pair(root)
    metadata = {'schemaVersion': 'theta.model-result.v1', 'modelId': model_name,
                'documentCount': int(theta.shape[0]), 'topicCount': int(beta.shape[0]),
                'vocabularySize': len(vocab), 'theta': str(theta_file.relative_to(root)), 'beta': str(beta_file.relative_to(root)),
                'thetaSemantics': 'latent_coordinates_not_probabilities' if model_name == 'nvdm' else 'membership_mass_excluding_outliers' if model_name == 'bertopic' else 'topic_distribution',
                'betaSemantics': 'normalized_selected_ctfidf_weights' if model_name == 'bertopic' else 'last_time_slice_topic_word_distribution' if model_name == 'dtm' else 'topic_word_distribution',
                'matrixHashes': {'theta': sha256_file(theta_file), 'beta': sha256_file(beta_file)}}
    (root / 'result_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata
