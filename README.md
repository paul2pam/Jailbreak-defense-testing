# Jailbreak Defense Testing

Two tools for testing and defending local LLMs: a jailbreak defense evaluator and a structured LLM test suite.

Both tools connect to a local [llama.cpp](https://github.com/ggerganov/llama.cpp) server via the OpenAI-compatible API.

## Prerequisites

```
pip install openai
```

Start a llama.cpp server before running either tool:

```
llama-server -m your-model.gguf --port 8080
```

Override the default URL with `LLAMA_BASE_URL=http://host:port/v1`.

---

## llm-jailbreak-defense

Tests a set of adversarial prompts against a defended LLM and reports which attacks were blocked vs. succeeded.

```
cd llm-jailbreak-defense
python jailbreak_defense.py
```

Interactive — press ENTER between the three test phases:

1. **Without defense** — sends an attack with no system prompt
2. **With defense** — same attack with the defense system prompt active
3. **All attacks** — runs every prompt in `jailbreak_prompts.txt` and prints BLOCKED/SUCCEEDED per attack

**To improve the defense:** edit `defense_system.txt` and re-run.

**To add attacks:** append to `jailbreak_prompts.txt`, separated by `---`.

---

## llm-test-suite

Tests four LLM-backed text operation functions across three testing paradigms (20 tests total).

```
cd llm-test-suite
python test_runner.py
```

Exit code `0` = all passed, `1` = failures. If llama.cpp isn't reachable, falls back to the OpenAI API (`gpt-4o-mini`) using `OPENAI_API_KEY`.

**Functions under test:**

| Function | Output |
|---|---|
| `sentiment(text)` | `{"sentiment", "confidence", "explanation"}` |
| `classify(text, categories)` | one label from the provided list |
| `summarize(text, length)` | summary string |
| `extract(text, fields)` | dict of field → value |

**Testing paradigms:**

| Paradigm | Count | What it checks |
|---|---|---|
| Black-box | 7 | Known inputs → expected outputs |
| Metamorphic | 6 | Relations between inputs (negation, caps, synonyms, rephrasing) |
| Property-based | 7 | Structural invariants across diverse inputs |

System prompts for each function live in `prompts/`. See `analysis.md` for full results and findings.
