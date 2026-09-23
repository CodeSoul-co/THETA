# Python Training Worker

The Worker consumes immutable `job.ready` messages from Redis Streams, downloads
the dataset from object storage, runs the existing THETA training scripts in an
isolated task directory, uploads a result bundle, and publishes lifecycle events
back to the Go control plane. It never connects to MySQL.

## Execution flow

```text
theta:jobs:cpu / theta:jobs:gpu
              |
       Consumer Group + lease
              |
      download and checksum data
              |
 prepare_data.py -> run_pipeline.py
              |
 archive + manifest + log upload
              |
        theta:jobs:events
```

One process handles one task at a time. Scale by running more processes or
containers with unique `WORKER_ID` values. CPU and GPU Workers use different
streams and consumer groups.

## Dataset and Agent contract

The Worker accepts CSV, TSV, TXT, Markdown, JSON, JSONL/NDJSON, XLSX, XLS,
Parquet, PDF, and DOCX objects. It verifies the uploaded checksum, converts the
selected text/label/time/covariate columns to the engine's UTF-8 CSV contract,
and only then starts preprocessing. Column mappings arrive as `input.*` values
inside `params` and are removed before engine argument routing.

The distributed image includes the same Agent adapter used by local execution.
All twelve registered models and the parameter namespaces returned by
`models.inspect` (`model.*`, `trainer.*`, `fit.*`, `prepare.*`, `main.*`,
`config.*`, `pipeline.*`, `word2vec.*`, and `embedding.*`) are validated against
the checked-out engine source. Unknown parameters fail the job instead of being
ignored. Rebuild and redeploy the image for these capabilities to exist on a
remote installation; changing this repository alone does not update a running
Worker.

Neural models accept `hidden_sizes` (one to ten layer widths, each 4-8192),
`dropout`, `learning_rate`, `patience`, and `no_early_stopping`. The legacy
`hidden_dim` plus `num_layers` pair remains supported; explicit
`hidden_sizes` takes precedence. THETA zero-shot jobs may use an
OpenAI-compatible remote embedding service through `embedding_provider`,
`embedding_cloud_provider`, `embedding_model`, `embedding_api_base`,
`embedding_api_key_env`, and `embedding_dimensions`. Only the environment
variable name travels in the task; its secret value must be installed on the
Worker host.

## Local setup

Create a Python environment that can already run the model code, then add the
Worker dependencies:

```text
python -m pip install -r ../src/models/requirements.txt
python -m pip install -r worker/requirements.txt
Copy-Item worker/.env.example worker/.env
```

Edit `worker/.env`. The most important values are:

```text
THETA_PROJECT_ROOT=D:/codesoul/projects/THETA
REDIS_ADDR=127.0.0.1:6379
REDIS_PASSWORD=
OBJECT_STORAGE_DRIVER=filesystem
OBJECT_STORAGE_FILESYSTEM_ROOT=D:/theta-object-storage
```

For local filesystem storage, if a `theta_datasets` row contains the storage
key `datasets/dev/train.csv`, put the input file at:

```text
D:/theta-object-storage/datasets/dev/train.csv
```

Start from the `trainning` directory:

```text
python -m worker
```

Set `WORKER_KEEP_JOB_DIR=true` while debugging. In normal operation set it to
`false`; a task directory is removed only after its terminal event has been
published and acknowledged.

## S3 or MinIO

Use the same bucket for the business backend, control-plane download provider,
and Workers:

```text
OBJECT_STORAGE_DRIVER=s3
OBJECT_STORAGE_ENDPOINT=127.0.0.1:9000
OBJECT_STORAGE_BUCKET=training-artifacts
OBJECT_STORAGE_ACCESS_KEY=...
OBJECT_STORAGE_SECRET_KEY=...
OBJECT_STORAGE_USE_SSL=false
```

In a container, `127.0.0.1` refers to the container itself. Use a Compose service
name such as `minio:9000`, or `host.docker.internal:9000` for a service exposed by
the host.

## Images

Build from the THETA repository root, not from `trainning`:

```text
docker build -f trainning/worker/Dockerfile.cpu -t theta-worker-cpu:v1 .
docker build -f trainning/worker/Dockerfile.gpu -t theta-worker-gpu:cu124-v1 .
```

Set `REDIS_PASSWORD=redis_dev_password` when using the Redis service in this
project's `docker-compose.yml`. Leave it blank for the currently running
passwordless Redis container.

The GPU container requires the NVIDIA Container Toolkit and should be started
with GPU access. The runtime image recorded in `model_runtime.image` documents
which Worker image is compatible with the task; Redis does not launch that
image itself. A later Kubernetes/Nomad scheduler can create these Worker pods
dynamically from the same runtime catalog.

## Worker replica count

A Worker process is a stateless queue consumer, not an Agent and not a user
session. One compatible Worker is sufficient: jobs wait in Redis and run one at
a time in that process. Multiple replicas in the same stream/Consumer Group
increase parallel throughput; a task lease ensures that one replica owns an
attempt. Size CPU and GPU streams independently and keep each runtime image
compatible with the registered `model_runtime`.

The three CPU containers used in the local 12-model acceptance test were a
temporary test setting for shorter wall-clock time. Production does not require
three. Start with one replica when serial latency is acceptable, then scale from
measured queue depth, task duration, CPU/RAM/GPU capacity and retry rate. Scaling
replicas does not change the Go Worker API or the Agent API.

## Result contract

The Worker uploads these objects below `output_prefix`:

```text
model.tar.gz              complete result directory
manifest.json             checksums and object-key index
worker.log                preprocessing/training stdout and stderr
files/...                 JSON/CSV/TXT and PNG/JPG/SVG/PDF/HTML outputs
```

`model.tar.gz` is returned as `model_weights_path`, so the existing download API
can serve one stable artifact for every supported model, including models whose
native outputs contain several weight and matrix files. Its compatible layout is
`result/`, plus `workspace/` evidence and `input/data.csv` for verified
source-row interpretation. `output_files.integrity` binds the archive and
normalized input with SHA-256 and byte sizes; old output-file keys remain valid.

The Go control plane stores `model_weights_path`, metrics, and the indexed
`output_files` object in `training_result`; the binary weights and charts stay
in object storage. `GET /api/v1/tasks/{taskID}/model/download` resolves that
stored archive path through the configured download provider. A production
private bucket therefore needs the control plane's signed-URL provider (the
development provider only joins a public base URL).

Delivery is at-least-once. Redis leases reduce concurrent duplicate execution,
and terminal event IDs are deterministic so the control plane's
`processed_event` table makes repeated terminal delivery idempotent.
When a Worker shuts down or loses its lease, it terminates the child process and
leaves the message pending instead of reporting a false training failure; another
Worker can reclaim it after `REDIS_WORKER_CLAIM_IDLE_MS`.
