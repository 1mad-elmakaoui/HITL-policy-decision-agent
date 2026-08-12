# Policy Review Assistant

A policy-review assistant that sits inside a business process. Someone asks whether
an action is allowed under company policy. The system finds the governing policy,
works out how risky the action is, reasons over what it found, and for high-risk
actions like termination, demotion or disciplinary action it stops and waits for a
human to sign off before releasing any answer.

Two separate systems share one interface:

| Layer | Framework | Owns | Runs |
|---|---|---|---|
| Knowledge | ZenML | parse, verify, chunk, embed, index, evaluate | offline, on demand |
| Orchestration | LangGraph | retrieve, assess, classify risk, interrupt, resume, answer | online, per request |

The vector store is the only thing they share. Ingestion writes it. The runtime
reads it and never writes.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ingestion,dev]"

# 1. Build the policy index (offline, ZenML)
python -m cli.ingest

# 2. Ask a low-risk question -> answered automatically
python -m cli.review submit "How many days of annual leave can be carried over?"

# 3. Ask a high-risk question -> pauses for human sign-off
python -m cli.review submit "Can we terminate an employee immediately for expense fraud?" \
    --request-id case-4711 --requested-by manager@example.com

# 4. Resume it, from a different shell, a different day, a different machine
python -m cli.review resume case-4711 \
    --decision changes_requested \
    --reviewer hr.partner@example.com \
    --feedback "Section 6 requires a completed investigation and Legal sign-off first."
```

Everything above runs with no API keys.

## Running the ingestion pipeline

Ingestion is a ZenML pipeline. It runs on demand and whenever policy documents
change, never as part of a user request.

```
Load -> Parse -> Verify -> Chunk -> Embed -> Index -> Evaluate
```

```bash
python -m cli.ingest                      # run the ZenML pipeline
python -m cli.ingest --local              # same steps, no orchestrator (CI, fresh clone)
python -m cli.ingest --report             # show current index statistics
python -m cli.ingest --no-strict-verification   # record findings as warnings, don't fail

zenml pipeline runs list                  # inspect versioned runs
```

Output:

```
Indexed 34 chunks from 4 documents into 'policy_chunks' (chroma).

Retrieval evaluation (component level):
  recall_at_5      0.933
  precision_at_5   0.187
  mrr              0.900
  ndcg_at_5        0.909

Evidence-threshold calibration:
  weak / strong    0.55 / 1.00
  on-corpus  top   0.63 .. 3.43
  off-corpus max   0.33 (0 reaching strong)

  gate: PASSED
```

## Running the assistant

```bash
python -m cli.review submit "<question>" [--request-id ID] [--requested-by WHO] [--force-review]
python -m cli.review pending <thread_id>     # what a parked request is waiting for
python -m cli.review resume  <thread_id> --decision {approved|rejected|changes_requested} \
                                          --reviewer <id> [--feedback "..."]
python -m cli.review show    <thread_id>     # checkpointed state
python -m cli.review trace   <thread_id>     # observability events
```

Or from Python:

```python
from agent.runtime import PolicyReviewService

service = PolicyReviewService()
result = service.submit("Can we demote an employee to a lower job level?",
                        request_id="case-4712")

if result.awaiting_human_review:
    print(result.interrupt_payload["preliminary_assessment"]["draft_answer"])
    # ... process may exit here; the request is safely checkpointed ...
    final = service.resume("case-4712",
                           {"decision": "approved", "reviewer_id": "hr@example.com"})
    print(final.final_answer)
```

### Handling an interrupted high-risk request

A high-risk request comes back with `status = pending_human_review` and no answer.
The reviewer gets the original request, the risk level and the reason it was
assigned, the retrieved policy passages with their document versions, and the
assistant's preliminary assessment, which says plainly that it is not an
authorisation.

Resume with the same `thread_id`. It is the request id, it never changes, and it is
all you need to pick the request back up:

```bash
python -m cli.review pending case-4711     # inspect
python -m cli.review resume  case-4711 --decision approved --reviewer hr@example.com
```

A reviewer decision has to identify the reviewer, and anything other than an
approval has to carry a reason. Malformed input never counts as consent: the
request stays parked and interrupts again, carrying the validation error.

The submitting process does not need to still exist. State lives in the SQLite
checkpoint database, not in memory:

```bash
python -m cli.review submit "Can we terminate an employee for cause?" --request-id case-99
# shell exits, machine reboots, a week passes
python -m cli.review resume case-99 --decision approved --reviewer hr@example.com
```

The resumed run continues from the interrupt. It does not re-retrieve, re-draft or
re-classify. `route_history` on the result shows each earlier node exactly once,
and `tests/persistence/` proves it from a genuinely separate OS process.

## Live demo interface

A web UI over the same service. Nothing extra happens in it. The routing, the risk
gate, the interrupt and the checkpointing are what the CLI runs.

```bash
pip install -e ".[web]"
python -m webapp                 # http://127.0.0.1:8000
```

It shows the workflow advancing node by node, the risk classification with its
reason and matched signals, the retrieved passages with scores, and for high-risk
questions a sign-off panel with Approve / Request changes / Reject. No answer
appears until a reviewer decides.

### With a real model

```powershell
pip install -e ".[live]"
$env:ANTHROPIC_API_KEY = "sk-ant-..."      # bash: export ANTHROPIC_API_KEY=sk-ant-...
python -m webapp --live
```

`--live` is what switches the mode. Setting the API key on its own does nothing.
The run mode defaults to mock, so the key sits unused and the app keeps working
with deterministic answers. If that happens the header turns amber and says so, and
the server prints the same warning at startup.

Live mode changes one thing: the drafting step calls a real model instead of the
deterministic one. Retrieval, the risk classifier, the human gate and checkpointing
are identical. The classifier stays deterministic on purpose, so a model can raise
a risk level but never lower one.

The header shows which mode is running, so there is no ambiguity on a projector.

## Run modes

Set `POLICY_REVIEW_RUN_MODE`:

- `mock` (default): deterministic providers. A hashed-BM25 embedder and a mock
  model that builds its answer out of the retrieved passages, so groundedness holds
  and you can assert on it without a network call.
- `live`: real providers. Set `llm_provider: anthropic` in
  `config/policy_review.yaml`, export `ANTHROPIC_API_KEY`, and install with
  `pip install -e ".[live]"`.

The graph, the interrupt, the checkpointing and the governance guarantees are
identical in both modes. Only the providers change.

## Configuration

`config/policy_review.yaml`, overridable by `POLICY_REVIEW_*` environment variables
(`POLICY_REVIEW_RETRIEVAL_TOP_K`, `POLICY_REVIEW_VECTOR_STORE_DIR`, ...). Run mode
is environment-only, so a checked-in file can never silently switch a deployment
between mock and live.

The evidence thresholds sit on the retrieval score's own scale, and every ingestion
run re-calibrates them against the labelled query set. Changing the embedding
provider changes the scale and means re-calibrating. The pipeline will tell you if
they no longer hold.
