# Free analysis mode

**English** | [中文](free-analysis.md)

Topic mode focuses on discovering and interpreting text topics. Free analysis starts from the research question and can use statistics, econometrics, prediction, survival analysis, optimization, or text models without requiring topic modeling first.

The Python worker registers 138 executable method IDs. Availability does not imply equivalence to every Stata/SPSS feature, default, or estimator, nor that a method's identifying assumptions hold for a dataset.

## Use the mode

In the conversation workbench, choose free analysis from the analysis-mode selector below the composer, import data, and describe your question. In the CLI, use `/mode free`; return with `/mode topic`. Mode is stored in the session and cannot change while a task or confirmation is pending.

For example, request descriptive statistics followed by OLS with HC3 standard errors, a predictive comparison with an independent test set, survival analysis with censoring, panel estimates with clustered errors, or a constrained optimization problem.

The Agent inspects data and method specifications, saves the question, hypotheses, assumptions, sensitivity checks, and stopping rule, then presents a confirmation card. Only that confirmed batch executes. Each batch allows at most six steps and 120 seconds. Inputs are limited to 20,000 rows, 200 columns, and 50 MiB without silent sampling.

If you only have a dataset, ask the Agent to help formulate a question. It can inspect fields, missingness, duplicates, observation units, and representative text. Text models are suggested for meaningful text, not identifiers or category codes. Synthetic examples are for practice rather than population-level findings.

## Source environment

Desktop installers bundle the compute environment. For a source installation, create a separate Python 3.11–3.13 statistics environment:

```sh
python3 -m venv agent/.local/runtimes/statistics
agent/.local/runtimes/statistics/bin/python -m pip install -r agent/workers/statistics/requirements-lock.txt
cd agent
npm run build
```

The Agent selects this environment by default. `THETA_WORKER_STATISTICS_PYTHON` and `THETA_WORKER_STATISTICS_REVISION` can select an explicit environment. Missing dependencies are reported rather than installed automatically. The virtual environment isolates dependencies; it is not an operating-system sandbox.

Approvals bind the dataset hash, full plan, research ID, interpreter, dependencies, implementation, and environment revision. Credentials are consumed once. Registered statistical tools accept controlled parameters, not arbitrary Python or formulas. Code-based analysis retains its separate execution requirements.

## Tools and plans

| Tool | Purpose |
| --- | --- |
| `statistics_methods` | List methods with family/query filters and pagination |
| `statistics_inspect` | Read exact inputs, parameter ranges, examples, and limitations |
| `statistics_plan` | Validate and save a plan without estimating or approving |
| `statistics_request_approval` | Create a confirmation card |
| `statistics_status` | Inspect progress and validate saved result hashes |
| `statistics_results` | Read delivered metrics, coefficients, and tables |
| `statistics_synthesize` | Revise interpretation of an existing delivered batch without recalculation |
| `analysis_checkpoint`, `analysis_history` | Save notes and inspect original receipts |

Example plan; replace `datasetRef` with an actual import receipt:

```json
{
  "datasetRef": "dataset-<sha256>",
  "plan": {
    "question": "How is x associated with y after adjusting for x2?",
    "hypotheses": ["The adjusted association is positive; this is exploratory"],
    "assumptions": ["Independent observations", "Linear conditional mean"],
    "steps": [
      {"method":"describe","x":["x","x2","y"]},
      {"method":"ols","name":"Model 1","x":["x","x2"],"y":"y","params":{"covariance":"HC3"},"missing":"drop","alpha":0.05,"seed":42}
    ],
    "sensitivity": ["Inspect residuals and influential observations"],
    "stoppingRule": "Report the prespecified estimates and diagnostics regardless of significance"
  }
}
```

Some optimization and prospective power methods accept parameters without a dataset. Derived CSV files are not automatically chained into the next step: inspect row identifiers and explicitly attach the derived data. Joining topic weights to metadata requires verified provenance, not only equal lengths.

## Jobs and recovery

The service saves jobs before returning HTTP 202. Polling and event streams update the workbench; completed steps are saved atomically. Plans, approvals, and notes are stored in the Agent home database; deliverables are under `statistics/<analysisId>/`.

Refreshing or navigating away does not cancel an accepted task while the service continues running. A service interruption preserves records and marks orphaned tasks interrupted. Resume by inspecting `statistics_status`: complete results with valid hashes may be re-registered, but unfinished batches require a new plan and confirmation. Duplicate request IDs with identical content return the original receipt.

## Results and interpretation

Batches deliver HTML/Markdown reports, complete CSV tables, full-precision JSON, esttab-style coefficient tables in CSV/LaTeX/RTF, supported PDF/SVG/PNG figures, the plan, `reproduce.py`, and file hashes. Empty estimates are not zeros. Significance symbols are defined in table notes and do not imply batch-wide multiplicity adjustment.

Run a reproduction script from `agent/` using the statistics interpreter. It verifies the input version and writes a separate `reproduced/` directory with environment provenance. Preserve the source commit and locked dependencies to recreate an environment.

Interpretation separates observations, estimates, and inference, and should explain denominators, exclusions, scales, reference groups, effects, uncertainty, assumptions, and limits. Invalid or nonconverged inference is not presented as a valid significance table.

## Method limits

Cox and AFT methods support right censoring. DID is limited to two groups and two periods. RDD uses a prespecified bandwidth and local linear sharp design. PSM uses nearest-neighbor matching with replacement and does not supply matching-inference standard errors. SEM does not provide all commercial estimators or multilevel features. Repeated-measures ANOVA requires balanced within-subject designs without sphericity correction.

Predictive tools require explicit group/time splits when relevant. Encoding, imputation, scaling, and tuning must use training data only. Parameter grids allow at most 12 combinations and cross-validation up to five folds. The test set must not be used for tuning or selecting a favorable random seed.

Use `statistics_methods` and `statistics_inspect` to obtain the complete executable specifications for the installed version. Source references include statsmodels, scikit-learn, lifelines, linearmodels, SciPy, and semopy; THETA does not execute Stata or SPSS scripts.
