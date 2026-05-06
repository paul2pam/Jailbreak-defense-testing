# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

CS/NPRE 398 coursework: experiments with local and cloud LLMs. The two polished deliverables are:

- **`llm-test-suite/`** — a custom test framework (no pytest) for testing LLM-backed text operations across three paradigms: black-box, metamorphic, and property-based.
- **`llm-jailbreak-defense/`** — an interactive script that runs adversarial prompts against a defended local LLM and reports block/success rates.

Weekly labs (`week01/`–`week08/`) and project submissions (`P01_paulcp2/`, `P02_paulcp2/`, etc.) are earlier iterations that informed the two above.

## Running things

### Prerequisites — llama.cpp server

Almost everything talks to a local llama.cpp server via the OpenAI-compatible API:

```
llama-server -m LFM2-8B-A1B-Q4_K_M.gguf --port 8080
```

Override the default URL with `LLAMA_BASE_URL=http://host:port/v1`.

The RAG scripts (`week06/rag_paulcp2.py`, `P06_paulcp2/rag_paulcp2.py`) need **two** servers — one for embeddings (nomic-embed-text, port 8080) and one for chat (LFM2, port 8081). Override with `EMBED_BASE_URL` / `CHAT_BASE_URL`.

### Run the test suite

```
cd llm-test-suite
python test_runner.py
```

Exit code 0 = all passed, 1 = failures. `test_suite.py` auto-detects llama.cpp; if it isn't reachable it falls back to the OpenAI API (`gpt-4o-mini`) using `OPENAI_API_KEY`.

### Run the jailbreak defense demo

```
cd llm-jailbreak-defense
python jailbreak_defense.py
```

Interactive — press ENTER between the three test phases.

### Install dependencies

```
pip install openai
```

Some week labs also use `tiktoken`, `transformers`, `sentence-transformers`, or `numpy`.

## Architecture of `llm-test-suite/`

```
llm-test-suite/
  test_suite.py      — four LLM-backed functions + all 20 test functions + ALL_TESTS registry
  test_runner.py     — iterates ALL_TESTS, prints PASS/FAIL per paradigm, exits 1 on failure
  prompts/           — system prompt templates loaded at import time
    _sentiment.txt   — returns JSON {sentiment, confidence, explanation}
    _classify.txt    — returns one label from a provided list
    _summarize.txt   — returns a summary string
    _extract.txt     — returns JSON dict of field → value
```

`test_suite.py` is both a library (imported by `test_runner.py`) and self-contained. The four functions (`sentiment`, `classify`, `summarize`, `extract`) use `temperature=0` throughout so tests are deterministic.

`_parse_json()` has a three-stage fallback: direct parse → markdown code fence → bare `{...}` regex, so tests are resilient to models that wrap JSON in prose. The `sentiment()` function uses `.replace("{text}", ...)` instead of `.format()` because the prompt file contains literal `{key}` JSON examples that would confuse `str.format()`.

## Architecture of `llm-jailbreak-defense/`

```
llm-jailbreak-defense/
  jailbreak_defense.py   — loads defense_system.txt + jailbreak_prompts.txt, runs three tests
  defense_system.txt     — 8-rule system prompt; edit this to improve the defense
  jailbreak_prompts.txt  — one attack per block, separated by ---
```

To add attacks: append to `jailbreak_prompts.txt` separated by `---`. To improve the defense: edit `defense_system.txt`. The block-detection heuristic in `jailbreak_defense.py` checks for ~15 refusal phrases in the response.

## .gitignore essentials

These should never be committed:

- `*.gguf` — model weights (multi-GB)
- `llama-b7898-bin-win-cpu-x64/` — compiled binaries
- `llama.cpp/build/` — CMake build artifacts
- `week03/.env` — API keys
- `**/__pycache__/`
- `*.zip`
