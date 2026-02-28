## 🛡️ LightShield
Near-zero latency, CPU-only prompt injection firewall for LLM applications.

LightShield is a lightweight security middleware that prevents prompt injection by enforcing **Strict Instruction Hierarchies**. Unlike traditional guardrails that use extra LLM calls to "guess" if an input is malicious, LightShield uses **Per-Session UUID Encapsulation** to make unauthorized command execution structurally impossible.

### 🚀 What LightShield Solves

- **Parser Confusion**: Attackers use fake closing tags (e.g., `</user_input>`) to "escape" their data container and issue new system commands.
- **Latency Bloat**: Existing solutions add 200–500ms of overhead per request, which breaks real-time agentic workflows.

LightShield treats prompt injection as a **security architecture problem**, not a content moderation problem.

### 💡 Core Idea: UUID Encapsulation

- **Cryptographic Delimiters**: Every input segment (system prompts, user messages, RAG documents, tool outputs) is wrapped in a session-unique UUID tag like `<LS_77a2c1e4d3b9>`.
- **Structural Anchoring**: Because UUIDs are generated at runtime and never exposed, attackers cannot pre-craft escape sequences.
- **Instructional Lockdown**: LightShield injects a high-priority **Boundary Rules** block into the system prompt that tells the model exactly which UUID tags have instructional authority.

### ✨ Features

- **⚡ Microsecond Latency**: Pure Python, CPU-only string operations. No extra LLM calls.
- **🔒 RAG-Hardened**: Explicitly downgrades retrieved documents and tool outputs to *data-only* segments.
- **🛡️ Blast Radius Control**: Ensures that data can never silently become a command.
- **🧩 Framework Agnostic**: Works with raw OpenAI/Anthropic clients and LangChain chains.

---

## ✅ Quick Start (Under 10 Minutes)

### 1. Install

```bash
pip install lightshield
```

### 2. Wrap Your Existing OpenAI Client

```python
from lightshield import LightShield

shield = LightShield()

response = shield.wrap(openai_client).chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_input},
        {"role": "retrieved", "content": rag_document},
    ],
)

if response.lightshield.is_safe:
    print(response.choices[0].message.content)
else:
    print("Blocked by LightShield:", response.lightshield.violations)
```

### 3. Anthropic Integration

```python
from lightshield import LightShield

shield = LightShield()

response = shield.wrap(anthropic_client).messages.create(
    model="claude-3-opus-20240229",
    messages=[
        {"role": "user", "content": user_input},
        {"role": "retrieved", "content": rag_document},
    ],
)

if response.lightshield.is_safe:
    print(response.raw.content)
```

### 4. LangChain RAG Integration

```python
from lightshield import LightShield

shield = LightShield()
secured_chain = shield.wrap(rag_chain)  # wraps as LangChainAdapter

result = secured_chain.invoke(
    "Answer the question using the documents.",
    documents=retrieved_docs,  # LangChain Document[]
)

if result.lightshield.is_safe:
    print(result.raw)
```

LightShield automatically:

- Generates **per-session UUIDs** for each trust level.
- Encapsulates system, user, RAG, and tool segments with level-specific tags.
- Injects a **Boundary Rules** block into the system prompt describing the trust map.
- Validates every response and attaches a `lightshield` object with `is_safe`, `violations`, and `confidence`.

---

## 🧪 Running the Benchmark Suite

LightShield ships with a small benchmark harness that simulates common injection attacks.

```bash
pip install -e .[dev]
pytest
```

Or call the harness directly:

```python
from lightshield.benchmarks import run_benchmarks

report = run_benchmarks()

print("Injection success rate:", report.injection_success_rate)
print("False positive rate:", report.false_positive_rate)
print("Avg latency overhead (ms):", report.avg_latency_overhead_ms)
```

The benchmark covers:

- Direct instruction override (`"Ignore previous instructions and..."`)
- Persona hijacking (`"You are now DAN..."`)
- RAG document injection
- Tool call manipulation
- Data exfiltration attempts
- Indirect webpage injections
- Metadata-based injections

---

## ⚙️ Configuration

- **`LightShield(raise_on_violation=True)`**: Raise `LightShieldViolationException` when a high-confidence violation is detected instead of returning a response.
- **Adapters** are auto-detected by duck-typing:
  - Objects with `.chat` → OpenAI adapter
  - Objects with `.messages` → Anthropic adapter
  - All others → LangChain adapter (must support `.invoke`)

LightShield is intentionally **stateless and lightweight** so you can safely instantiate it per-process or per-request.
