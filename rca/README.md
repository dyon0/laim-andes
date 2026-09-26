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
| `scores.p_anomaly` | calibrated probability of an anomaly (the record's `confidence` is now `p_anomaly × 100`; detector builds before this fix wrote `max(p, 1−p)` there) |
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

## Optional input: `agent_report` (the agent's development report)

When the port is connected, the agent's development report becomes context for
the analysis. It is placed once in the system prompt, shared by every batch. The
LLM is told to check each record against it: violated correctness criteria,
stop-list phrases, answers outside the agent's competence, and wrong tool
choices count as anomalies, and `rca` names the violated requirement.
Fallbacks and scenarios the report describes are not anomalies by themselves.
With the port empty, the node behaves exactly as without it. If data is
connected but cannot be read, the run continues without the report and
`rca_audit.agent_report.error` says why.

**Decision: feed the document itself, not the g-aiva-doc-browser output.**
doc-browser distills a report for *model validation*: metric and threshold, task
type, dataset/sample description, a summary, generation hyperparameters, and a
summarized ML architecture. What RCA needs is what that distillation drops,
checked against the three example reports:
- the correctness criteria and "за что штрафуем" lists, with stop-list phrases
  quoted verbatim («займите у родственников», «возьмите микрозайм»);
- the competence boundaries («у нашего агента нет компетенций в других темах»);
- the tool table (`extract_credit_products`, `get_product_conditions`, …),
  whose names match span names in traces;
- the class → chain mapping, the fallback stub text («По техническим причинам
  сейчас я не могу ответить…»), the data-source APIs (ФССК, Исп.П, …), and the
  expected response time.

The raw document is also cheaper: no extra LLM pipeline (doc-browser makes
5+ calls and starts a gpt2giga proxy process). Reading it is deterministic, and
wording is kept exactly as written, which matters when the question is "did
the agent say a forbidden phrase". doc-browser output is still accepted as a
fallback, rendered as text, for pipelines that only have that.

Accepted inputs (the format is detected by content, not by file name):

| input | examples |
|---|---|
| **.docx** | a local path (including the platform's extension-less `unstructured_data` blob), raw bytes, or doc-browser's own input dict `{"bin": …, "ext": "docx"}` |
| **HTML** | a Confluence page or export, Word "save as web page" (utf-8 or windows-1251), `.html`/`.htm`, or an HTML string |
| **MHTML** | Confluence "export to Word" `.doc`, Word "single-file web page" `.mht` |
| **plain text / Markdown** | a string |
| **g-aiva-doc-browser output** | `{"all_results": {"bp_card": …}, "extracted_fields": {…}}` or either part alone |

How the platform delivers the port is normalized first, the same way the
kriteria-selector node reads `development_report_artifact`. SberDS has no single Python form
for a DataArtifact, so all of these lead to the document bytes:
a file path (with or without extension), a **port directory** mounted "as files and folders"
(service files `_SUCCESS`, `_committed*`, `*.crc`, dotfiles are ignored; with several files
.docx > HTML > MHTML > pickle > extension-less > text), raw bytes / `bytearray` / `memoryview`,
a dict with the document under any of `bin`, `bytes`, `content`, `data`, `payload`, `value`,
`unstructured_data`, `path`, `local_path`, `file` (extension from `ext`/`extension`/`filename`/`name`;
nested dicts and base64 strings too), a pickle of such a dict, a one-row **parquet container**
or `pandas.DataFrame`/`Series` with bytes or a path inside, a one-item list, a file-like object,
and an **opened python-docx `Document`**, which is how SberDS delivers a .docx port. The Document is
saved back to .docx bytes and parsed by the same parser as a file. If saving fails, paragraph and table text
is taken through the python-docx API.
The run log shows what arrived (`порт agent_report подан: dict {bin: bytes, ext: str}`,
`путь к каталогу … (файлы: …)`) and every unwrapping step. If a report was supplied but
could not be read, the log and the prompt line say **«подан, но НЕ ПРОЧИТАН»** with the
reason. They never say «не подан» in that case. «не подан» means the port really was empty
(`None`, an empty string, `"None"`, `NaN`).

PDF is rejected with a clear message: convert the report to .docx or HTML. A pickle file (doc-browser's
`report_dict` form) is read with a builtins-only unpickler: dicts, strings and
bytes are allowed, any class is refused. Bytes that are not text, .docx or
HTML are refused rather than sent to the model. Either way the reason ends up
in `rca_audit.agent_report.error`.

How the report is used in the prompt: it comes right after the role, as the
**leading context**, before the record fields. The analysis steps start with
it ("step 0: determine from the report what the agent should have done").
The RCA must cite a violated requirement explicitly («По отчёту о разработке
агент должен …, а в ответе …»), and `business_description` is written in terms
of the report's business process. Records analyzed with the report carry
`rca_results.agent_report_used = true`, which the report node shows as the tag
«с учётом отчёта о разработке». If that tag is missing, the report did not
reach the model: check `rca_audit.agent_report`.

Parsing uses only the standard library (zipfile + XML, `html.parser`, `email`
for MHTML), so no new requirements. Paragraphs and tables are kept in document
order; table rows become `| key | value`. The template's unanswered questions
(«- описание формул…» with no answer), «Заполняется на этапе…» lines, empty
sections and contact rows (names and emails) are dropped. When the text exceeds
`report_max_chars` (default 20 000), sections are dropped in an order that keeps
what describes the agent's behavior: artifacts list, SOTA, pilot, training and
control datasets, labeling, and so on first; the appendix of prompts last. The
tail is cut only after that. The largest example report (35K chars) fits into
17K with its criteria, tools table and fallback stubs intact.

## Cross-trace analysis and concise RCA

Feedback on the first upgrade: explanations got long and hard to read, and the
model stopped relating traces to each other. The previous simple version had
done that well (e.g. «агент не нашёл ГБК, хотя в трейсе 6fdbccc1… дал
расшифровку»). What changed since:

- **Related records** (`laim_rca/related.py`, deterministic, over the whole
  input). Each record's query terms are matched against the queries and
  answers of all other records. Terms are codes (`DFA_OPER_FEE_SC_WD`,
  `47109.99`, `П3399`), abbreviations (`ГБК`, matched case-insensitively,
  so «крюл» finds «КРЮЛ») and significant words. A link requires a shared code
  or at least two shared words, weighted by IDF, so words common to the whole
  agent never link. Each record gets up to 3 `related` traces (trace_id, query,
  answer) in the prompt, including ones processed in another batch. Batches are
  ordered so linked records travel together.
- **Prompt:** a dedicated step to compare records ("found here, «not found»
  there", different expansions of one term, the same failure repeated) and to
  cite trace_ids.
- **`rca` format** is back to the proven short form, one string of at most
  600 characters: «суть с фактами и trace_id. Возможные причины: 1) …;
  2) …; 3) …». **No categories.** The model is told not to invent categories,
  classifications, terms or abbreviations, and to write plain Russian (English
  only when quoting data: trace_id, codes, step names). Labels it adds anyway
  are stripped: «Категория: failure_propagation_or_guardrail.», «ГАЛЛЮЦИНАЦИЯ:»,
  `snake_case:` prefixes, and `category` / `anomaly_category` / `type` keys.
  Codes like «ГБК:» are kept. `add_info` can still define another format;
  category fields in it are not shown either.
- **Detector signal in the prompt is brief by default**
  (`evidence_detail = brief`): probability, signal, and the suspicious steps
  with excerpts. It no longer includes feature numbers or a generated
  hypothesis, which the model used to paraphrase instead of explaining.
  `full` restores them.
- `rca_results.related_traces` lists the other traces the RCA cites. The
  report node links those trace_ids to their cards.

## Run log

Every stage prints an explicit line (`print(..., flush=True)`, so it shows in
the SberDS run log immediately), formatted `[RCA HH:MM:SS +N.Ns] stage: message`:
- `старт`: parameters;
- `вход`: record and trace counts;
- `детектор`: how many records carry `detector_rca`, and the signal mix;
- `отчёт`: whether the port is empty, the detected format (path/pickle/.docx/
  HTML/MHTML/text/doc-browser), size in the prompt, template sections found,
  sections dropped by `report_max_chars`, and the start of the loaded text.
  A read error is printed too;
- `модель`: client created or not;
- `связи`: records with related traces, with examples;
- `промпт`: system prompt size, and whether the report, detector signal and
  `add_info` are included;
- `LLM`: every batch (size, bytes, trace_ids), its time and verdicts; failures,
  splits, 429/503 waits, retries, records skipped by the model, fallback;
- `итог`: verdict counts, requests, and in how many output records the report
  was used and other traces were cited.

The report node logs `[REPORT …]` lines: input, types shown or hidden and
why, RCA format, traffic light.

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
| `evidence_detail` | `brief` | detector signal shown to the LLM: `brief` (probability, suspicious steps with excerpts) or `full` (plus feature deviations and the detector hypothesis) |
| `report_max_chars` | `20000` | character budget for the development report in the prompt (used only when `agent_report` is connected) |

## Outputs

**`res`**: `{"anomalies": [...]}` for the report builder. Records are kept for
verdict `anomaly`, plus `uncertain`/`unverified` when `keep_uncertain` is on (all
records in `detector_only`), in input order. Detector fields are never changed.
The model fills `business_description` only when the detector
left it blank; `tech_details` is not filled (technical details are part of the RCA). The raw `detector_rca` is removed from the output. `rca_results`
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

**`rca_audit`** (schema `laim.rca_audit/1`): mode, model, the development report
used (`agent_report`: source format, size, truncation, or a read error), counts per verdict,
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
