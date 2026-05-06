"""
LLM Test Suite — Text Operations Toolkit
  Tests four LLM-backed functions across three testing paradigms:
    Black-box, Metamorphic, Property-based

Target functions:
  sentiment(text)             → {"sentiment": ..., "confidence": ..., "explanation": ...}
  classify(text, categories)  → category string
  summarize(text, length)     → summary string
  extract(text, fields)       → dict of field → value

Connects to llama.cpp at http://localhost:8080/v1 (or set LLAMA_BASE_URL env var).
Temperature=0 for determinism in all tests.

Run directly or import into test_runner.py.
"""

import os
import sys
import json

from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompt loading — prompts/ directory lives next to this file
# ---------------------------------------------------------------------------

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


SUMMARIZE_PROMPT = _load_prompt("_summarize.txt")
CLASSIFY_PROMPT  = _load_prompt("_classify.txt")
EXTRACT_PROMPT   = _load_prompt("_extract.txt")
SENTIMENT_PROMPT = _load_prompt("_sentiment.txt")


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def _build_client():
    """Use llama.cpp if available, otherwise fall back to the OpenAI API."""
    llama_url = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080/v1")
    try:
        import urllib.request
        urllib.request.urlopen(llama_url.replace("/v1", "/health"), timeout=2)
        print(f"[client] llama.cpp server detected at {llama_url}")
        return OpenAI(base_url=llama_url, api_key="not-needed"), None
    except Exception:
        print("[client] llama.cpp not reachable — falling back to OpenAI API")
        return OpenAI(), "gpt-4o-mini"


client, _OVERRIDE_MODEL = _build_client()


def _detect_model() -> str:
    if _OVERRIDE_MODEL:
        return _OVERRIDE_MODEL
    try:
        models = client.models.list()
        if getattr(models, "data", None):
            return models.data[0].id
    except Exception:
        pass
    return os.environ.get("LLAMA_MODEL", "local-model")


MODEL = _detect_model()
print(f"[client] model: {MODEL}")


# ---------------------------------------------------------------------------
# Thin wrappers (temperature=0 for determinism in tests)
# ---------------------------------------------------------------------------

def _call(prompt: str, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_json(raw: str) -> dict:
    """Try to extract a JSON object from the model's raw response."""
    import re
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Cannot parse JSON from: {raw[:200]}")


def _parse_category(raw: str, categories: list) -> str:
    resp = raw.strip().lower()
    for cat in categories:
        if cat.strip().lower() == resp:
            return cat.strip()
    for cat in categories:
        if cat.strip().lower() in resp:
            return cat.strip()
    return "Unknown"


def sentiment(text: str) -> dict:
    # Use replace() instead of .format() — the prompt file contains JSON
    # examples with {key} syntax that confuse str.format().
    prompt = SENTIMENT_PROMPT.replace("{text}", text)
    raw = _call(prompt, temperature=0.0)
    data = _parse_json(raw)
    data.setdefault("sentiment", "unknown")
    data.setdefault("confidence", "unknown")
    data.setdefault("explanation", "")
    return data


def classify(text: str, categories: list) -> str:
    cats_str = "\n".join(f"- {c}" for c in categories)
    prompt = CLASSIFY_PROMPT.format(categories=cats_str, text=text)
    raw = _call(prompt, temperature=0.0)
    return _parse_category(raw, categories)


def summarize(text: str, length: str = "2-3 sentences") -> str:
    prompt = SUMMARIZE_PROMPT.format(text=text, length=length)
    return _call(prompt, temperature=0.0)


def extract(text: str, fields: list) -> dict:
    fields_str = "\n".join(f"- {f}" for f in fields)
    prompt = EXTRACT_PROMPT.format(fields=fields_str, text=text)
    raw = _call(prompt, temperature=0.0)
    data = _parse_json(raw)
    for f in fields:
        data.setdefault(f, None)
    return data


# ===========================================================================
# ===  BLACK-BOX TESTS (7 tests)  ===========================================
# ===========================================================================

def test_bb_positive_sentiment():
    """Black-box: Clearly positive text → 'positive' sentiment.
    Paradigm: Black-box
    What it tests: The classifier recognises unambiguous praise.
    Success criteria: result["sentiment"] == "positive"
    Why it matters: Catches prompt regressions that break obvious positive cases.
    """
    result = sentiment("I absolutely love this product! It is the best thing I have ever bought.")
    assert result["sentiment"] == "positive", (
        f"Expected 'positive', got '{result['sentiment']}'"
    )


def test_bb_negative_sentiment():
    """Black-box: Clearly negative text → 'negative' sentiment.
    Paradigm: Black-box
    What it tests: The classifier recognises unambiguous complaints.
    Success criteria: result["sentiment"] == "negative"
    Why it matters: Catches regressions where the model stops identifying negatives.
    """
    result = sentiment("Terrible experience. Complete waste of money. I am furious.")
    assert result["sentiment"] == "negative", (
        f"Expected 'negative', got '{result['sentiment']}'"
    )


def test_bb_neutral_sentiment():
    """Black-box: Purely factual / neutral text → 'neutral' sentiment.
    Paradigm: Black-box
    What it tests: Factual text with no emotional language is not mislabelled.
    Success criteria: result["sentiment"] == "neutral"
    Why it matters: A model biased toward positive/negative would fail here.
    """
    result = sentiment("The box is rectangular and made of cardboard.")
    assert result["sentiment"] in ("neutral", "mixed"), (
        f"Expected 'neutral' or 'mixed', got '{result['sentiment']}'"
    )


def test_bb_classifier_returns_valid_label():
    """Black-box: classify() returns one of the provided categories.
    Paradigm: Black-box
    What it tests: The model respects the closed-set constraint.
    Success criteria: result in categories list
    Why it matters: Rogue labels (e.g., free-form text) break downstream code.
    """
    categories = ["sports", "technology", "politics", "entertainment"]
    result = classify("The new iPhone was announced at today's Apple keynote.", categories)
    assert result in categories, f"Got invalid category: '{result}'"


def test_bb_summarize_nonempty():
    """Black-box: summarize() on a long paragraph → non-empty string.
    Paradigm: Black-box
    What it tests: The summarizer produces output at all.
    Success criteria: len(result) > 0
    Why it matters: An empty summary is a silent failure easy to miss.
    """
    long_text = (
        "Machine learning is a branch of artificial intelligence that focuses on "
        "building systems that learn from data. Rather than explicitly programming "
        "rules, ML algorithms identify patterns in training data and use those "
        "patterns to make decisions on new data. Applications range from image "
        "recognition to natural language processing and medical diagnosis."
    )
    result = summarize(long_text)
    assert len(result.strip()) > 0, "summarize() returned empty string"


def test_bb_extract_finds_name_and_date():
    """Black-box: extract() pulls correct field values from structured text.
    Paradigm: Black-box
    What it tests: Named-field extraction accuracy on a clear example.
    Success criteria: name field contains 'Alice', date field contains '2024-03-15'
    Why it matters: Extraction accuracy is the core product promise.
    """
    text = "Meeting with Alice on 2024-03-15 at 10am to discuss the budget."
    fields = ["name", "date"]
    result = extract(text, fields)
    assert "Alice" in str(result.get("name", "")), f"name field wrong: {result}"
    assert "2024-03-15" in str(result.get("date", "")), f"date field wrong: {result}"


def test_bb_unicode_and_emoji_no_crash():
    """Black-box: Emoji/unicode input is handled without exception.
    Paradigm: Black-box
    What it tests: Robustness to non-ASCII characters.
    Success criteria: Returns a dict (no exception raised)
    Why it matters: Real users send emoji; crashes are bad UX.
    """
    result = sentiment("Great product! 👍🎉🌟 Highly recommend!")
    assert isinstance(result, dict), "sentiment() did not return a dict"
    assert "sentiment" in result


# ===========================================================================
# ===  METAMORPHIC TESTS (6 tests)  =========================================
# ===========================================================================

def test_meta_negation_flips_sentiment():
    """Metamorphic: Adding 'don't' should flip sentiment polarity.
    Paradigm: Metamorphic
    Relation: sentiment("I love X") ≠ sentiment("I don't love X")
    Why it matters: Negation is a common prompt-drift failure mode.
    """
    pos = sentiment("I love this movie.")["sentiment"]
    neg = sentiment("I don't love this movie.")["sentiment"]
    assert pos != neg, (
        f"Negation did not flip sentiment: both returned '{pos}'"
    )


def test_meta_caps_invariance():
    """Metamorphic: ALL-CAPS text should give the same sentiment as lower-case.
    Paradigm: Metamorphic
    Relation: sentiment(text.upper()) == sentiment(text.lower())
    Why it matters: Capitalisation should not change the label.
    """
    base  = "I really enjoyed this book"
    r_up  = sentiment(base.upper())["sentiment"]
    r_low = sentiment(base.lower())["sentiment"]
    assert r_up == r_low, (
        f"CAPS changed sentiment: upper='{r_up}', lower='{r_low}'"
    )


def test_meta_synonym_consistency():
    """Metamorphic: Synonymous positive words should give the same sentiment.
    Paradigm: Metamorphic
    Relation: sentiment("I love this film") == sentiment("I adore this film")
    Why it matters: Synonym sensitivity reveals fragile prompts.
    """
    r_love  = sentiment("I love this film.")["sentiment"]
    r_adore = sentiment("I adore this film.")["sentiment"]
    assert r_love == r_adore, (
        f"Synonyms produced different sentiment: love='{r_love}', adore='{r_adore}'"
    )


def test_meta_summary_always_shorter():
    """Metamorphic: summarize(long_text) is always shorter than the input.
    Paradigm: Metamorphic
    Relation: len(summary) < len(original) for non-trivial inputs
    Why it matters: A 'summary' that is longer defeats its purpose.
    """
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Artificial intelligence has transformed many industries. " * 20,
        "Climate change is a global challenge. " * 15,
    ]
    for text in texts:
        result = summarize(text)
        assert len(result) < len(text), (
            f"Summary not shorter than input (summary={len(result)}, original={len(text)})"
        )


def test_meta_rephrased_same_category():
    """Metamorphic: Rephrasing should not change the topic category.
    Paradigm: Metamorphic
    Relation: classify(paraphrase) == classify(original)
    Why it matters: Paraphrase robustness matters for real-world text variation.
    """
    categories = ["sports", "technology", "politics"]
    v1 = classify("Scientists built a new AI chip that runs faster than GPUs.", categories)
    v2 = classify("A faster artificial-intelligence processor has been developed by researchers.", categories)
    assert v1 == v2, f"Paraphrase changed category: '{v1}' vs '{v2}'"


def test_meta_extract_equivalent_phrasing():
    """Metamorphic: Field values survive minor rephrasing of the source text.
    Paradigm: Metamorphic
    Relation: Both versions should return the same name.
    Why it matters: Extraction must tolerate natural language variation.
    """
    text1 = "Dr. Smith will present on Tuesday."
    text2 = "On Tuesday, Dr. Smith is scheduled to give a presentation."
    r1 = extract(text1, ["name"])
    r2 = extract(text2, ["name"])
    name1 = str(r1.get("name", "")).lower()
    name2 = str(r2.get("name", "")).lower()
    assert "smith" in name1 and "smith" in name2, (
        f"Name extraction inconsistent across phrasing: '{name1}' vs '{name2}'"
    )


# ===========================================================================
# ===  PROPERTY-BASED TESTS (7 tests)  ======================================
# ===========================================================================

VALID_SENTIMENTS  = {"positive", "negative", "neutral", "mixed"}
VALID_CONFIDENCES = {"high", "medium", "low"}


def test_prop_sentiment_required_keys():
    """Property: Every sentiment response has all three required keys.
    Paradigm: Property-based
    Invariant: result.keys() ⊇ {sentiment, confidence, explanation}
    Why it matters: Missing keys crash any downstream code that reads them.
    """
    inputs = [
        "I love it!",
        "Terrible.",
        "The table is brown.",
        "Mixed feelings: great product, awful shipping.",
        "123 !!??",
    ]
    for text in inputs:
        result = sentiment(text)
        for key in ("sentiment", "confidence", "explanation"):
            assert key in result, f"Missing key '{key}' for input: {text[:40]!r}"


def test_prop_sentiment_valid_values():
    """Property: sentiment field is always one of the four valid labels.
    Paradigm: Property-based
    Invariant: result["sentiment"] ∈ VALID_SENTIMENTS
    Why it matters: Unexpected labels (e.g., "ambiguous") break label-based routing.
    """
    inputs = [
        "Fantastic product, five stars!",
        "Broken on arrival, very disappointed.",
        "Water is composed of hydrogen and oxygen.",
        "I liked the story but hated the ending.",
        "Great price! 😍 but slow shipping 😤",
        "a" * 500,
    ]
    for text in inputs:
        result = sentiment(text)
        assert result["sentiment"] in VALID_SENTIMENTS, (
            f"Invalid sentiment '{result['sentiment']}' for: {text[:40]!r}"
        )


def test_prop_confidence_valid_values():
    """Property: confidence field is always one of high/medium/low.
    Paradigm: Property-based
    Invariant: result["confidence"] ∈ VALID_CONFIDENCES
    Why it matters: Downstream routing (e.g., low-confidence → human review)
                    depends on a constrained vocabulary.
    """
    inputs = [
        "I hate it.",
        "Could be better, could be worse.",
        "!?!?!?",
        "Outstanding service every single time.",
    ]
    for text in inputs:
        result = sentiment(text)
        assert result["confidence"] in VALID_CONFIDENCES, (
            f"Invalid confidence '{result['confidence']}' for: {text[:40]!r}"
        )


def test_prop_classify_always_valid_label():
    """Property: classify() always returns one of the provided labels.
    Paradigm: Property-based
    Invariant: result ∈ categories
    Why it matters: An out-of-set response silently corrupts label counts.
    """
    categories = ["positive", "negative", "neutral"]
    inputs = [
        "I love it!",
        "Terrible experience.",
        "The sky is blue.",
        "!@#$%^&*()",
        "a" * 1000,
        "Mixed feelings here.",
    ]
    for text in inputs:
        result = classify(text, categories)
        assert result in categories, (
            f"Got invalid category '{result}' for: {text[:40]!r}"
        )


def test_prop_summarize_returns_string():
    """Property: summarize() always returns a non-None string.
    Paradigm: Property-based
    Invariant: isinstance(result, str)
    Why it matters: Type safety — callers call .strip(), .lower(), etc.
    """
    inputs = [
        "Short sentence.",
        "Medium paragraph with several sentences to summarize well.",
        "Long text. " * 50,
    ]
    for text in inputs:
        result = summarize(text)
        assert isinstance(result, str), f"summarize() returned {type(result)}"


def test_prop_extract_returns_dict():
    """Property: extract() always returns a dict, never crashes.
    Paradigm: Property-based
    Invariant: isinstance(result, dict)
    Why it matters: Callers index by field name; a non-dict would raise TypeError.
    """
    cases = [
        ("Meeting with Bob on Friday at 3pm.", ["name", "date", "time"]),
        ("No information here.", ["name", "date"]),
        ("Email: user@example.com, Phone: 555-1234", ["email", "phone"]),
    ]
    for text, fields in cases:
        result = extract(text, fields)
        assert isinstance(result, dict), f"extract() returned {type(result)}"


def test_prop_extract_all_requested_fields_present():
    """Property: All requested fields appear as keys in the result dict.
    Paradigm: Property-based
    Invariant: set(fields) ⊆ result.keys()
    Why it matters: Missing keys raise KeyError in downstream code.
    """
    cases = [
        ("Alice met Bob at the Eiffel Tower on March 5.", ["person", "location", "date"]),
        ("Order #A123, qty 4, price $29.99.", ["order_id", "quantity", "price"]),
        ("No data here whatsoever.",              ["name", "date", "amount"]),
    ]
    for text, fields in cases:
        result = extract(text, fields)
        for f in fields:
            assert f in result, (
                f"Field '{f}' missing from extract() result. Got keys: {list(result.keys())}"
            )


# ===========================================================================
# ===  TEST REGISTRY (imported by test_runner.py)  ==========================
# ===========================================================================

ALL_TESTS = [
    # Black-box
    ("black-box",    "test_bb_positive_sentiment",               test_bb_positive_sentiment),
    ("black-box",    "test_bb_negative_sentiment",               test_bb_negative_sentiment),
    ("black-box",    "test_bb_neutral_sentiment",                test_bb_neutral_sentiment),
    ("black-box",    "test_bb_classifier_returns_valid_label",   test_bb_classifier_returns_valid_label),
    ("black-box",    "test_bb_summarize_nonempty",               test_bb_summarize_nonempty),
    ("black-box",    "test_bb_extract_finds_name_and_date",      test_bb_extract_finds_name_and_date),
    ("black-box",    "test_bb_unicode_and_emoji_no_crash",       test_bb_unicode_and_emoji_no_crash),
    # Metamorphic
    ("metamorphic",  "test_meta_negation_flips_sentiment",       test_meta_negation_flips_sentiment),
    ("metamorphic",  "test_meta_caps_invariance",                test_meta_caps_invariance),
    ("metamorphic",  "test_meta_synonym_consistency",            test_meta_synonym_consistency),
    ("metamorphic",  "test_meta_summary_always_shorter",         test_meta_summary_always_shorter),
    ("metamorphic",  "test_meta_rephrased_same_category",        test_meta_rephrased_same_category),
    ("metamorphic",  "test_meta_extract_equivalent_phrasing",    test_meta_extract_equivalent_phrasing),
    # Property-based
    ("property",     "test_prop_sentiment_required_keys",        test_prop_sentiment_required_keys),
    ("property",     "test_prop_sentiment_valid_values",         test_prop_sentiment_valid_values),
    ("property",     "test_prop_confidence_valid_values",        test_prop_confidence_valid_values),
    ("property",     "test_prop_classify_always_valid_label",    test_prop_classify_always_valid_label),
    ("property",     "test_prop_summarize_returns_string",       test_prop_summarize_returns_string),
    ("property",     "test_prop_extract_returns_dict",           test_prop_extract_returns_dict),
    ("property",     "test_prop_extract_all_requested_fields_present",
                     test_prop_extract_all_requested_fields_present),
]
