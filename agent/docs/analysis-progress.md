# Research progress and deeper interpretation

Hosts can explicitly select `new ConversationAgent({ ...ports, mode: 'research' })` for an autonomous research request. The default remains interactive. Research mode uses a compact, evidence-oriented instruction set and requires a progress checkpoint after eight tool calls; it does not add tools, execute approvals, or change model selection. Interleaved repeated computations are checked over the recent eight calls, not just consecutive identical calls. All limits and native authorization checks still apply.

## Executable post-model mining

### Measurement alignment and complete feedback

CSV cleaning now emits `*.lineage.json`, binding the input/cleaned files and each raw/cleaned row by SHA256. Native preprocessing exports selected records as `row_provenance.json`; reports verify original prepared-input identity, raw text hashes, model text hashes and selected indices before joining rows to topic weights. Stale or mismatched records cannot establish alignment. Legacy cleaned runs without lineage remain unverified; identity arrays and equal dimensions do not upgrade them. This is provenance, not a semantic label or model-quality check. CSV cleaning/BOW integration and report/cache tests cover the new path; other input converters do not automatically gain this lineage.

Method critique hashes exclude `requiredFiles`: filenames are delivery metadata, while question, hypotheses and stopping rule remain part of the scientific review identity. Changing actual measurements still invalidates the review; changing file names does not repeat a model vote. Required-file byte checks and external user-required submission validation remain separate.

The mining library includes `audit_sample` and `audit_disagreements` for a reproducible raw-text panel across assigned positive/negative/unknown states. The analyst must supply judgments; no helper invents labels or calls a model. Disagreement reports do not certify accuracy. Preserve definitions, old errors and seed selection, and inspect topic-boundary/near-miss cases separately before scaling lexical rules into population claims.

Approved synthesis takes precedence over autonomous research instructions: it interprets supplied evidence with the read-only interpretation role, returns its body, and lets the host save it. Lack of execution tools during this phase is intentional, not a missing capability. Both interactive and research modes have regression coverage for this boundary.

Method self-review requests its named structured tool explicitly. Extra issue entries are retained (up to the bounded schema limit), never converted into approval. Review instructions prohibit imposing a broader objective or rejecting a permissible narrow comparison merely for optional extensions. This does not guarantee reliable model critique. Workspace inventory excludes the automatically saved script directory and exposes recent script receipts separately, so accumulated scripts cannot hide actual output files.

Exploration no longer requires an invented hypothesis: use `analysis_execute(stage='explore')` to inspect topic artifacts, original/boundary documents and public contracts before `analysis_plan`. Planned tests use `stage='test'` (the default).

When the host supplies `inferenceFactory`, saving a plan invokes a bounded methodological self-review through the same configured provider. It compares the original user objective with the proposed constructs, measurements, comparison and falsification. It receives no evaluator, reference answers, previous submissions or score feedback. A `revise`/`unclear` critique blocks planned testing and delivery until the plan is revised, while exploration stays available. Identical plan/objective hashes reuse the existing critique; changed plans or objectives require another review. This adds metered inference within host/provider budgets. It is **not independent validation**, semantic-label verification, a score, or new authorization. Hosts without a reviewer retain explicit unreviewed operation.

Sandbox stdout/stderr now retain both beginning and end, with byte counts, truncation flags and an explicit omitted-region marker. This prevents lengthy setup/file listings from silently hiding a final calculation or traceback. UTF-8 chunks are decoded incrementally. The full analysis script and output files remain the reproducible source; omitted output must not be inferred. The worker revision/fingerprint changes with this transport behavior.

### Worker-owned sandbox

`WorkerAnalysisRuntime` now delegates process creation, limits, cancellation and confirmed cleanup to a `SandboxWorker`. The concrete implementation lives in `agent/src/workers/sandbox-worker.ts`; `DockerSandboxWorker` is the currently implemented backend. `DockerAnalysisRuntime` remains a backward-compatible local default. This is not an SSH/remote backend implementation or a persistent Python kernel.

The host may choose `worker(session)` per study. For an explicitly selected worker, the corresponding `AnalysisBinding.workerFingerprint` must match `worker.describe().fingerprint` and its image must match the authorized binding. The descriptor covers worker ID/revision, backend, image, CPU/memory/time limits, offline policy and persistence semantics. Changing the worker/profile without updating host authorization fails closed; the Agent cannot transfer approval or select arbitrary endpoints. Receipts are scoped by session, study, binding and worker fingerprint, so a newly authorized worker does not inherit an old worker's delivery status. Sharing or migrating files is an explicit host choice; scripts still require compatible inputs and provenance.

```ts
const worker = new DockerSandboxWorker({ image: approvedImageDigest, cpus: 4, memoryMiB: 6144 });
const runtime = new WorkerAnalysisRuntime({
  stateDirectory, libraryDirectory,
  worker: session => authorizedWorkers.get(session.id + ':' + session.runId)!,
  binding: session => approvedBindings.get(session.id + ':' + session.runId),
});
// The host stores worker.describe().fingerprint in that study's approved binding.
// It must not copy a model-proposed fingerprint into an authorization record.
```

Workspace inspection exposes the actual worker profile to the Agent. Execution receipts include that profile and `cleanupConfirmed`; the Docker worker removes only its invocation's container and confirms it is absent. An unresolved cleanup blocks further execution and delivery. Host-only `runtime.recover(session, executionId)` can recheck/remove that owned sandbox, but cannot rewrite an execution error into success. No model tool exposes this recovery permission. Code runs with a cleared process environment (`env -i`), read-only public/model mounts, no network and only the designated writable workspace. Reports and source data never become commands or authorizations.

The native SDK now exports `DockerAnalysisRuntime`, an optional **host-authorized** downstream execution port. `LocalProductTools({ ..., analysis: runtime })` exposes `analysis_workspace`, `analysis_plan`, `analysis_execute` and `analysis_deliver` after an authorized complete report and interpretation. A training or interpretation approval alone does **not** authorize arbitrary code: the host must separately authorize and supply a bounded binding for the exact current session/study. Never construct that binding from model-supplied directories, image IDs or claims of approval. CLI/Web do not silently enable this runtime; hosts without a configured binding retain their existing behavior.

```ts
const analysis = new DockerAnalysisRuntime({
  stateDirectory: approvedHostAuditDirectory,
  libraryDirectory: installedMiningLibraryDirectory, // agent/workers/mining
  binding(session) {
    // Return undefined unless this exact session/study has a host-approved
    // downstream scope. Resolve only owned, authorized current-job artifacts.
    return approvedBindings.get(session.id + ':' + session.runId);
  },
});
const tools = new LocalProductTools({ runtimeDb, uploadDir, analysis });
const agent = new ConversationAgent({ inference, tools, save, mode: 'research' });
```

A binding supplies `scope`, real absolute `workspace`, read-only `inputs` and `model` directories, a locally available SHA256-pinned `image`, `maxExecutions` and `maxSeconds` per execution. Do not mount an entire project or home. The runtime never pulls images, installs packages or mounts credentials. Docker uses no network, four CPUs, 6 GiB memory, unprivileged execution and a read-only root. Maximum individual execution is 180 seconds; the host conversation signal can shorten it. Cancellation/timeout removes only the uniquely named container. Native training resources remain separate.

Each submitted script is automatically saved under `analysis-scripts/` and separately audited outside the sandbox. Scripts, expected outputs, exit code, missing files, changed output hashes and duration form actual execution receipts. Variables **do not** persist between Python processes; reuse saved scripts with `runpy.run_path` and persist intermediate results. This provides reproducible script-based continuity, not a persistent notebook kernel. Plans and notes are not verified milestones. The host reserves delivery guidance near the budget end and attempts bounded recovery before accepting an unfinished response.

`analysis_deliver` requires real successful execution and every nonempty plan-required file. It archives original bytes outside the sandbox with hashes and host download paths. A later workspace mutation invalidates current delivery state without changing the archived snapshot. Delivery does not establish semantic accuracy, statistical validity or fulfillment of a separate external submission protocol; those require the user's actual validator/evaluator. The host does not generate findings or modify answers.

The mounted `theta_mining` library provides model-axis probing with representative/boundary/contrast originals, exhaustive positive/negative/unknown partition checks, two-arm descriptive contrasts, missing-label bounds, metadata stratification, largest-stratum removal, support concentration and exact quotation verification. It neither assigns semantic labels nor optimizes definitions to obtain stronger effects. Row alignment remains a provenance requirement, not something inferred from equal matrix dimensions. See `agent/workers/mining/README.md` for usage and limits.

This makes the downstream cycle executable: model evidence → explicit concepts and alternatives → persistent computation → falsification and scope checks → verified files. It does not prove better empirical mining performance. Use fresh frozen-source evaluations to measure that separately.

## Verification

`npm test` runs native contract tests. Set `THETA_ANALYSIS_TEST_IMAGE` to a locally installed pinned image ID to include the opt-in real Docker lifecycle test (otherwise that test is skipped). Run the Python library tests with a NumPy-equipped interpreter: `python -m unittest discover -s workers/mining -p 'test_*.py' -v`. Tests cover missing labels, incorrect known counts, a synthetic Simpson reversal, null/missing strata, source quotations, row mapping, authorization absence, unsafe paths, script reuse, timeout cleanup, exact delivery and changed artifacts. Synthetic checks are not benchmark scores.

THETA can preserve a bounded, study-specific analysis notebook with `analysis_checkpoint`. The notebook contains the research objective, candidate explanations, supporting and opposing evidence references, completed calculations, open questions, saved artifact paths and the next concrete action. It is included in subsequent conversation context and survives a session resume.

Notes are authored by the agent, not verified evidence. Saving a path does not create a file, and saving a proposed action never approves it. A selected study receives its own notebook; switching studies does not silently reuse another study's claims. Authorized result synthesis remains a separate, read-only interpretation stage and cannot overwrite the notebook.

In research mode, a successfully saved checkpoint anchors subsequent conversational history for that study. The current request and subsequent complete tool/result pairs remain visible, and the original user objective is carried in host state. `analysis_history` retrieves original receipts by call ID or backward pagination without repeating the action. Historical receipts explicitly identify their study when available; older records with unknown study identity must be verified before reuse. A failed notebook write never becomes a context boundary.

Repeated identical computations are stopped before a third consecutive execution. The agent must record an evidence-based checkpoint and select a different next action. Repeated cached inspections also enter this recovery path rather than disabling every tool. Live job-status polling remains available. These guards prevent waste; they do not establish that an analysis is correct or complete.

Model-facing history preserves the latest complete tool-call/result pair even when a large user receipt occupies the usual context budget. User-facing artifact links remain in the saved answer, while subsequent inference uses the undecorated answer body. The inference provider retains public tool-call narrative, but not hidden reasoning content.

For substantive research, topics are intermediate evidence. Interpret their words alongside original and boundary-case documents, verify row provenance before joining metadata, define observable concepts and uncertain cases, compare competing explanations, and use actual computations to examine effect size, scope, composition, missingness and counterexamples. A finding may be a relationship, a reversal, a composition effect or an unsupported hypothesis. Same-corpus exploration is not independent confirmation, and association is not causation.

When execution tools exist, save reusable analysis and intermediate results instead of relying on interpreter variables. Complete and verify the requested deliverable before ending the turn; a promise to export is not an exported artifact. The host still controls tool availability, resource limits and per-action authorization.
