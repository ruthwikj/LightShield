# LightShield
## 🛡️ LightShield
Near-zero latency, CPU-only structural firewall for LLM applications.

LightShield is a lightweight security middleware that prevents prompt injection by enforcing Strict Instruction Hierarchies. Unlike traditional guardrails that use expensive LLM calls to "guess" if an input is malicious, LightShield uses Per-Session UUID Encapsulation to make unauthorized command execution structurally impossible.

## 🚀 The Problem
In 2026, Large Language Models (LLMs) still struggle to distinguish between Developer Instructions and Untrusted Data.

The "Parser Confusion" Attack: Attackers use fake closing tags (e.g., </user_input>) to "escape" their data container and issue new system commands.

The Latency Trap: Current solutions (like NeMo or Llama Guard) add 200ms–500ms of latency per request, making them unusable for real-time agentic workflows.

## 💡 The Core Innovation: UUID Encapsulation
LightShield treats prompt injection as a Security Architecture problem, not a content moderation problem.

Cryptographic Delimiters: Every input (user messages, RAG documents, tool outputs) is wrapped in a session-unique UUID tag (e.g., <LS_77a2>).

Structural Anchoring: Because the UUID is generated at runtime and unknown to the attacker, they cannot pre-craft an "escape sequence."

Instructional Lockdown: LightShield injects a high-priority "Boundary Rule" into the system prompt that anchors the LLM's attention to these specific UUID tokens.

## ✨ Features
⚡ Microsecond Latency: Runs on the CPU using optimized string-wrapping and regex. No GPU or extra LLM calls required.

🔒 RAG-Hardened: Specifically designed for "Indirect Injection" where malicious instructions are hidden in retrieved PDFs or websites.

🛡️ Blast Radius Control: Prevents "Agentic Hijacking" by ensuring data can never be promoted to a command.

🧩 Framework Agnostic: One-line integration for LangChain, AutoGen, CrewAI, or raw OpenAI/Anthropic SDKs.
