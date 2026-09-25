# LAIM RCA node

A separate SberDS node that takes the laim detector's output (`test_anomalies`),
filters false positives with an LLM, and writes a root-cause analysis for every
confirmed anomaly. It uses the detector's own attribution (`detector_rca`): which
branch fired, which spans (steps) and features deviated, and by how much. It does
not import anything from laim-andes; the only coupling is the JSON contract below.

```
laim-detector (inference) ──test_anomalies──▶ LAIM RCA ──res──────▶ report builder (detector_anomalies)
                                                        └─rca_audit─▶ (monitoring / feedback)
```

## Deploying

The node is built like any SberDS git node: the **contents of this directory**
become the root of the node's repository (`descriptor.json` + `main.py` entry
point `main` + `requirements.txt`, image `py312-simple`). The descriptor keeps
the previous node `id`, so it registers as an upgrade of the existing LAIM RCA
node. `llm/` (GigaChat / AI Gateway clients) is unchanged in behavior. Its env
contract is the same: `AI_GATEWAY_URL` (SDS contour), or
`CREDENTIALS`/`AUTH_URL`/`SCOPE` (sigma contour), plus `TIMEOUT`, `VERIFY_SSL_CERTS`,
`TOP_P`.

Tests: `pip install -r requirements.txt pytest && pytest` (runs offline; the LLM is faked).

## Input: `anom_data`

`{"anomalies": [...]}` as JSON text, bytes, dict or list. The previous input
shapes still work (`root:` prefix, double-encoded JSON, BOM). Each record is
laim's product record (`trace_id`, `starttime`, `endtime`, `anomaly_type`,
`confidence`, `user_query`, `agent_response`, …). Since this upgrade, a flagged
record also carries **`detector_rca`** (schema `laim.detector_rca/1`, produced
by laim-andes `ars/stages/s4__rca.py::export_detector_rca`). The main parts:

| field | meaning |
|---|---|
| `agent_id` | which agent's step sequence this record scores (one record per trace × agent) |
| `scores.p_anomaly` | calibrated probability of an anomaly. The legacy `confidence` is `max(p, 1−p)`, i.e. confidence in the detector's own decision, not in the anomaly |
| `scores.branch_z` | robust z of each branch's error compared with normal training traces: `behavior` (EPI: timings, lengths, counts, step structure) vs `semantic` (SEM: embedding of step texts) |
| `scores.flag_share` | split of the flagging (combined) error between the two branches |
| `scores.logit` | additive decomposition: `p_anomaly = sigmoid(sum(logit))` |
| `spans` | catalog of the referenced spans: name, kind, status, status_message, start_time, duration_s, http/llm/kafka attributes, input/output excerpts |
| `behavior.spans` | the spans with the worst behavioral reconstruction, with `vs_typical` (× the normal error level) and their **step-level driver features** |
| `behavior.features` | the EPI features with the largest error: observed vs expected (the value the model reconstructs as normal) in the feature scale and, where defined, natural units (`*_raw`: ns, chars, counts), plus the span where each deviates most |
| `semantic.spans` | the spans whose text is least typical |
| `feature_info` | provenance per feature: base feature, aggregation, window, log1p, `scope` (`step` = the step's own value, `sequence` = a static aggregate over the agent's whole sequence) |

Records without `detector_rca` (older detector versions, or attribution
switched off with `attribution_top_k = 0`) are processed as before.

## How the detector fields are used

1. **Evidence** (`laim_rca/evidence.py`, deterministic): dominant branch,
   signal strength (from `p_anomaly`), a hypothesis with a category, suspicious
   spans (a span with an error status comes first), deviating features in human
   units (e.g. «длительность вызова инструмента: 12.3 с при ожидаемых ~400 мс
   (выше нормы)»), and caveats: truncated sequence, low `p_anomaly` despite the
   flag, sequence-wide rather than step-specific deviation, values clipped by
   normalization. Feature names come from `laim_rca/glossary.py`. laim's test
   suite fails if a new detector feature is missing there.
2. **Prompt**: each record goes to the LLM with a compact `detector_evidence`
   in place of the raw block. The system prompt explains the evidence and tells
   the model to start from the suspicious spans. It still gives hallucinations
   priority, but asks the model to consider technical causes when the signal is
   behavioral. `use_detector_evidence = false` turns this off (useful for A/B
   testing).
3. **Localization**: the model returns the `span_id` where the cause shows up.
   It is accepted only if the detector knows that span. Otherwise the detector's
   top span is used (`location.source` says which).
4. **Without an LLM**: `detector_only` mode, `llm_fallback` after an LLM failure,
   and records the model never analyzed (`unverified`) get an RCA built from the
   evidence: category, hypothesis, evidence, recommendation.
5. **Audit**: each decision records whether the LLM verdict agrees with the
   detector's signal strength. The disagreements are the most useful labeled
   cases for tuning the detector.

## Parameters

| parameter | default | meaning |
|---|---|---|
| `add_info` | `" "` | operator context appended to the system prompt: domain, rules, taxonomy, the desired `rca` format |
| `model_id` | `minimax-m2.5` | `giga*` → GigaChat client; anything else → AI Gateway `chat/completions` |
| `llm_temp` | `0.001` | temperature |
| `max_tokens` | `8192` | completion budget per batch (the model no longer echoes records, so answers are short) |
| `mode` | `llm` | `llm`: an LLM failure fails the node (as before). `llm_fallback`: if the LLM is unavailable, records get detector-based RCA. `detector_only`: no LLM calls |
| `use_detector_evidence` | `true` | send the detector's explanation to the LLM |
| `keep_uncertain` | `true` | keep records with verdict `uncertain` and records the LLM could not analyze (`unverified`) |

## Outputs

**`res`**: `{"anomalies": [...]}` for the report builder. Records are kept for
verdict `anomaly`, plus `uncertain`/`unverified` when `keep_uncertain` is on (all
records in `detector_only`), in input order. Detector fields are never changed.
The model fills `business_description`/`tech_details` only when the detector
left them blank. The raw `detector_rca` is removed from the output. `rca_results`
is an object:

```json
{
  "verdict": "anomaly | uncertain | unverified",
  "verdict_confidence": 85,
  "severity": "low | medium | high | critical",
  "rca": "<the model's root cause: string, or the object shape add_info asks for>",
  "location": {"agent_id": "…", "span_id": "…", "span_name": "get_rate", "span_kind": "tool", "source": "llm | detector"},
  "detector_evidence": {"p_anomaly": 0.97, "strength": "strong", "signal": "behavior", "category": "Аномальные задержки",
                        "hypothesis": "…", "top_spans": [...], "top_features": [...], "caveats": [...]},
  "analyzed_by": "llm:minimax-m2.5 | detector"
}
```

**`rca_audit`** (schema `laim.rca_audit/1`): mode, model, counts per verdict,
detector agreement totals, LLM statistics (requests, failures, 429 waits,
elapsed time, fallback reason), and one decision per **input** record,
including the ones filtered out. Each decision carries the verdict, confidence,
severity, a short reason, the detector's view and the agreement flag.

## What changed versus the previous version

- **The detector's attribution is used**, as described above.
- **Explicit verdicts, matched by id.** The model returns only
  `{id, verdict, confidence, severity, span_id, rca, …}` per record. Before, it
  echoed every record in full: output tokens were wasted, answers got cut at the
  length limit, and the batch had to be split. Before, a record the model left
  out was treated as "not an anomaly". Now omissions are retried, and verdict
  `normal` is required to drop a record.
- **Records the model could not analyze are no longer silently dropped.** They
  are kept as `unverified` with detector-based RCA (`keep_uncertain = true`).
- **Records of one trace (different agents) go into the same batch** and are
  analyzed together. Before, duplicate `trace_id`s were forced into separate
  batches because matching was by `trace_id`.
- **System/user message separation and a prompt-injection guard.** Record
  contents are declared data, not instructions.
- **429/503 back off** (honoring `Retry-After`) instead of splitting the batch.
- Very long texts are cut to head + tail (24k chars) instead of failing the
  record on context overflow.
- `llm_fallback` / `detector_only` modes; `rca_audit` out-port.
- `requirements.txt` now lists only what the code imports. `python-dotenv` was
  imported by `llm/config.py` but not declared; `tqdm`, `krippendorff`,
  `pandas`, `numpy`, `langchain`, `langchain-community` were unused.
  `VERIFY_SSL_CERTS` now reaches the AI Gateway client (default unchanged: off).
- Descriptor: the `res` port label no longer reads «максимальная длина
  контекста»; `max_tokens` is an integer.
- `tests/test_full_coverage.py` from the previous package was not carried over.
  It targeted a different (id-based) version of `main.py` and failed 28 of 34
  tests against the shipped one. Its intents (completeness, field integrity,
  routing, retry behavior) are covered by `tests/test_main.py`.

**Contract note for the report builder:** `rca_results` used to be whatever the
model returned (a string or an object). It is now always an object, and the
model's own analysis sits under `rca_results.rca`.

## Further upgrade proposals (not implemented)

- **Detector `confidence` semantics** (laim-andes): for flagged records report
  `p_anomaly × 100`, not `max(p, 1−p) × 100`. A flagged trace with
  `p_anomaly = 0.3` is currently shown with confidence 70.
- **Trace-level verdict**: records of one trace are already analyzed together.
  The next step is an explicit per-trace conclusion: which agent is the origin,
  and which agents only propagate the anomaly.
- **Feedback loop**: `rca_audit` decisions (especially `disagrees`) are a ready
  source of weak labels for threshold and calibration tuning, and for the s3
  anomaly-type classifier, which is trained on synthetic types only.
- **Throughput**: 2–4 concurrent batch requests (threads; the platform is not
  spawn-safe), plus caching by `(trace_id, agent_id, content hash)` so reruns do
  not re-query the LLM.
- **Stronger evidence from laim**: counterfactual deltas (the existing
  `attribution.counterfactual` hook: "with a normal duration the error drops by
  X %"), and nearest normal examples for semantic anomalies ("what a normal
  answer looks like").
- **Detector feature design**: static aggregates (`*_max`, `*_q95`, …) are
  constant across a sequence yet dominate EPI attribution on the real corpus
  (see PLAN.md: ~57 `avg_word_length_sem_*` variants). Better feature diversity
  gives better evidence.
- **Structured output** (`response_format` / JSON schema) where the gateway
  supports it, and a lower `max_tokens` now that answers are short.
- **PII masking** of queries, responses and span excerpts before they are sent
  to models outside the contour.
