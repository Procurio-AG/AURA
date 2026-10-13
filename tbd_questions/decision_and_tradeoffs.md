# Architectural Decision Record: Agent Execution Mode (Option B with Fallback to Option A)

**Date**: 2026-10-13  
**Status**: Active Decision  
**Context**: Tool-Governed Early Warning System (EWS) evaluating Gemma 4 26B (`gemma-4-26b-a4b-it`) and Gemma 4 31B (`gemma-4-31b-it`) on Google AI Studio API.

---

## 1. The Decision

1. **Primary Execution Mode**: **Option B (Multi-Turn ReAct Agent Loop)**.
   - The LangGraph / LangChain agent executes multi-turn conversational tool-calling (`sql_query`, `predict_risk`, `predict_next_score`, `write_intervention_report`).
2. **Resilience & Fallback Policy**: **Automatic Fallback to Option A (Structured Pre-Fetch + Single-Turn Synthesis)**.
   - If Option B encounters unrecoverable errors (e.g., HTTP 500 on OpenAI compatibility layer, tool schema parsing errors on open-weights Gemma, or 429 quota exhaustion exceeding retry budgets), the agent automatically falls back to Option A to guarantee completion without data loss.
3. **Transport Protocol**: Native Google Gen AI REST endpoint (`/v1beta/models/{model}:generateContent`) with structured schema definitions, bypassing the unstable OpenAI compatibility layer.
4. **Key Management**: Dedicated Key 1 for Gemma 26B, Dedicated Key 2 for Gemma 31B, and Shared Key 3 as dynamic overflow/failover. (All 3 keys belong to separate Google AI Studio accounts/projects with independent 10 RPM quotas).

---

## 2. Supporting Tradeoff Analysis (Audited via Coder-Pro)

### Detailed Comparison Matrix

| Metric / Dimension | Option A: Structured Pre-Fetch + Single-Turn Synthesis | Option B: Multi-Turn ReAct Agent Loop (Tool-Calling) |
| :--- | :---: | :---: |
| **API Calls per Student** | **1 call** | **3 to 5 calls** (`sql_query` $\to$ `predict_risk` $\to$ `report`) |
| **Throughput (10 RPM Limit)** | **Up to 10 students/min** per key | **Only 2 to 3 students/min** per key |
| **Cohort Throughput (2 Keys)**| **~18–20 students/min** | **~4–6 students/min** |
| **Latency per Report** | **15–25 seconds** | **60–120 seconds** (sequential network turns) |
| **Rate-Exhaustion Risk (429)** | **Very Low** (linear, 1:1 consumption) | **Severe** (1 student consumes 40–50% of 60s quota) |
| **Endpoint Compatibility** | **100% Native REST compatible** | Requires custom tool-call token parsing |
| **Failure / Drift Risk** | **< 1%** (direct Pydantic JSON validation) | **15–30%** (multi-turn conversation state drift) |

---

## 3. Detailed Architectural Breakdown

### Why Option B is Evaluated First
Option B represents the standard autonomous ReAct paradigm where the LLM dynamically explores the student's database records and requests model risk scores on demand. Testing Option B directly measures whether open-weights Gemma 4 models can reliably manage multi-turn tool calling within rate-limited environments.

### Why Fallback to Option A is Mandatory
- **The 10 RPM Constraint**: Because each key is capped at 10 requests per minute, an agent taking 4 turns per student will exhaust quota after processing just 2 students concurrently.
- **Open-Weights Tool Parsing**: If Google AI Studio's proxy fails to serialize function calls into Gemma's prompt template (the cause of the observed HTTP 500 error), Option A immediately rescues execution by pre-fetching the verified SQL rows and calibrated ML predictions in Python (<5ms) and prompting Gemma to synthesize the grounded report in a single turn.

---

## 4. Operational Invariants

1. **Thread Safety**: All API dispatches and key state updates are protected by `threading.Lock()` to prevent micro-bursting race conditions between Gemma 26B and 31B.
2. **Quota Tracking**: Each key maintains an independent sliding-window timestamp buffer enforcing $\le 8$ requests per 60 seconds (20% safety margin under the 10 RPM ceiling).
3. **Transparent Auditing**: Every fallback event from Option B to Option A is logged with the triggering error code and student ID for empirical reporting.
