# LAIM anomaly report node

A separate SberDS node: it takes the anomaly list (laim's `test_anomalies`, or
the `res` output of the LAIM RCA node) and renders a self-contained HTML
report (no JS) plus a JSON summary with a traffic light. Like `rca/`, the
contents of this directory become the node's own repository root:
`descriptor.json` + `main.py` (entry point `main`) + `requirements.txt` (empty
of packages: standard library only).

```
laim-detector ──test_anomalies──▶ LAIM RCA ──res──▶ LAIM anomaly report ──anomaly_report (html) / all_results
```

## Parameters

| parameter | default | meaning |
|---|---|---|
| `min_confidence` | `75` | only anomalies with `confidence ≥` this (0–100) are listed |
| `anomaly_types` | `auto` | `auto`: show anomaly types only if the detector's type classifier (s3) ran; `show` / `hide`: force |

## Anomaly types and the s3 classifier

Anomaly types come from laim's s3 classifier. With it disabled
(`classifier_enabled = false`), laim leaves `anomaly_type` empty in every
record. In `auto` mode the node detects that (no detector record carries a
type; drift/quality-test records do not count) and removes everything
type-related:
- the «Описание показателей» section (type descriptions and the confidence
  scale guide);
- the «Типов аномалий» / «Преобладающий тип» stats and the type distribution
  in «Сводка»;
- the type badges in «Перечень аномалий» (cards then use a neutral colour).

`all_results.anomaly_types_shown` records the decision.

## Changes versus the previous version

- The eyebrow header «Автономный мониторинг ИИ-агентов · Детектор аномалий»
  is removed, in every case.
- No "markup needed" messages: the traffic-light title no longer ends with
  «— требуется разметка Владельцем», and the confidence-scale band says
  «стоит перепроверить» instead of «требует проверки Владельцем агента».
- `rca_results` from the upgraded LAIM RCA node is an object (verdict,
  severity, `rca`, location, detector evidence). It used to be printed as a
  raw JSON blob. It is now rendered as readable text: the root cause (with
  `add_info`-shaped fields as `key: value` lines), «Где: шаг … агент …», and
  the detector's hypothesis, plus tags for severity and for records the LLM
  was unsure about or did not verify. Plain-string `rca_results` render as
  before.

## Tests

`pytest` (standard library + pytest only).
