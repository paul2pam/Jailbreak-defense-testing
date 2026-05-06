# LLM Test Suite — Analysis
**Model under test:** LFM2-8B-A1B-Q4_K_M (llama.cpp, localhost)
**Result:** 20/20 passed — all paradigms

---

## 1. Test Target

The test suite targets a **Text Operations Toolkit** that exposes four LLM-backed functions:

| Function | Input | Output |
|---|---|---|
| `sentiment(text)` | arbitrary text | JSON: `{sentiment, confidence, explanation}` |
| `classify(text, categories)` | text + label list | one label string |
| `summarize(text, length)` | long text + length hint | summary string |
| `extract(text, fields)` | text + field names | JSON dict of field → value |

This target covers a wide range of LLM output types (free-form string, constrained label, structured JSON), making it ideal for demonstrating all three testing paradigms on a single deployed system. It also has well-understood expected behavior, making black-box ground truth easy to specify.

---

## 2. Paradigm Coverage

### Black-box testing (7 tests)

Black-box tests treat each function as a black box and define observable success criteria against known inputs. For example, `test_bb_positive_sentiment` sends obviously positive text ("I absolutely love this product!") and asserts the label is `"positive"`. The criteria are exact and machine-checkable without any knowledge of the prompt internals.

Edge cases covered: unicode/emoji input (`test_bb_unicode_and_emoji_no_crash`), multi-field extraction on a structured sentence (`test_bb_extract_finds_name_and_date`), and an empty-output guard (`test_bb_summarize_nonempty`).

**Most straightforward paradigm.** The hard part is identifying the boundary conditions; writing the assertion is trivial once the expected output is pinned down.

### Metamorphic testing (6 tests)

Metamorphic tests exploit *relations* between inputs instead of requiring a single ground-truth output. Four relations were tested:

- **Negation flips polarity:** `"I love X"` should give a different sentiment than `"I don't love X"`.
- **Caps invariance:** same text in ALL-CAPS should give the same sentiment label.
- **Synonym consistency:** `"love"` vs. `"adore"` should classify identically.
- **Compression monotonicity:** summaries must be shorter than their inputs (tested across three texts of different lengths).
- **Paraphrase consistency:** two semantically equivalent sentences about AI hardware should get the same topic category.
- **Extraction phrasing invariance:** `"Meeting with Dr. Smith on Tuesday"` and `"On Tuesday, Dr. Smith is scheduled..."` should both return `"Smith"` for the `name` field.

**Most intellectually interesting paradigm.** It is the only way to test behavior when no ground-truth label exists — for instance, there is no authoritative "correct summary" of a paragraph, but we can still assert that a summary is shorter than the original.

### Property-based testing (7 tests)

Property tests define invariants that must hold across a *set* of diverse inputs. Key invariants tested:

- `sentiment()` always returns all three required keys (`sentiment`, `confidence`, `explanation`).
- `sentiment` is always one of `{positive, negative, neutral, mixed}`; `confidence` is always one of `{high, medium, low}`.
- `classify()` always returns a label that is a member of the provided categories list — never a free-form string.
- `summarize()` always returns a Python `str` (type safety).
- `extract()` always returns a `dict` and always includes every requested field as a key (value may be `null` but the key must be present).

Each property test ran 3–6 diverse inputs including edge cases like `"!@#$%^&*()"`, `"a" * 1000`, and near-empty text.

**Highest regression value.** Property tests are indifferent to what the model *says* and only care about structural guarantees, so they catch regressions caused by prompt rewrites, model swaps, or output-format changes.

---

## 3. Findings

All 20 tests passed on a clean run against `LFM2-8B-A1B-Q4_K_M.gguf` with `temperature=0`.

Notable observations:

- **Caps invariance held cleanly.** `"I really enjoyed this book"` and its ALL-CAPS version both returned `"positive"` — the model is robust to case variation for simple sentiment.
- **Negation was handled correctly.** `"I love this movie"` → `"positive"`, `"I don't love this movie"` → `"negative"`. This is a known failure mode for weaker models; LFM2-8B passed.
- **All extraction fields were present even when absent from source text.** The null-fill logic ensured keys like `"amount"` appeared even when the text contained no dollar figures — correct behavior confirmed by property test.
- **Summaries were always significantly shorter than inputs.** The longest input was ~2,400 characters; the longest summary was ~350 characters (~7× compression ratio).

No failures were found. This is a positive result for the model and prompts, but it also means the test suite did not surface any bugs on first run against a working system.

---

## 4. Regression Value

These tests catch the following classes of regressions if prompts or models change:

| Change | Tests that would catch it |
|---|---|
| Prompt rewrite causes sentiment to return `"happy"` instead of `"positive"` | `test_prop_sentiment_valid_values`, all black-box sentiment tests |
| New prompt drops the `confidence` field from JSON output | `test_prop_sentiment_required_keys` |
| Model swap causes classifier to return free-form explanation instead of a label | `test_prop_classify_always_valid_label`, `test_bb_classifier_returns_valid_label` |
| Summarizer prompt changed to "expand" instead of "summarize" | `test_meta_summary_always_shorter` |
| Extract prompt changed to omit null keys for missing fields | `test_prop_extract_all_requested_fields_present` |
| New model fails on negation | `test_meta_negation_flips_sentiment` |

The suite integrates cleanly into a CI pipeline: start `llama-server`, run `python test_runner.py`, check exit code (0 = all passed, 1 = failures).

---

## 5. Limitations

**What cannot be tested automatically:**

1. **Output quality beyond format.** A summary that says "The text is about something" passes structural tests but is useless. Measuring quality requires an LLM-as-judge or human review.

2. **Hallucination in extraction.** The property tests only verify structure. A test that checks "extracted values actually appear in the source text" would require substring matching or embedding similarity — neither is fully reliable.

3. **Long-tail input distribution.** Property tests used 4–6 hand-crafted inputs. A true property-based framework (e.g., Hypothesis) would generate thousands of random strings, potentially surfacing crashes on unusual Unicode sequences.

4. **Non-determinism at temperature > 0.** All tests use `temperature=0` for repeatability. Production usage often runs at higher temperature, where the same test might pass one run and fail the next. These flaky behaviors are invisible to the current suite.

5. **Latency and throughput.** The suite has no assertions on response time. A prompt change that causes 10× longer outputs before trimming would pass all tests but degrade user experience.

---

## Results Summary

```
============================================================
  RESULTS: 20/20 passed   [ALL PASSED]
============================================================
  Paradigm        Passed  Failed
  --------------  ------  ------
  black-box            7       0
  metamorphic          6       0
  property             7       0
============================================================
```

Model: `LFM2-8B-A1B-Q4_K_M.gguf` | Temperature: 0 | Runtime: ~2 minutes
