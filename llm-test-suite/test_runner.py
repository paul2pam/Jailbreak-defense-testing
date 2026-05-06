"""
LLM Test Runner — runs all tests from test_suite.py and reports results.

Usage:
    python test_runner.py

Prerequisites:
    llama.cpp server running at http://localhost:8080/v1
    (or set LLAMA_BASE_URL env var)
"""

import sys
import time
import traceback

# Force UTF-8 output on Windows so emoji in tracebacks don't crash the runner
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from test_suite import ALL_TESTS


def run_all_tests():
    results = {
        "passed":     0,
        "failed":     0,
        "by_paradigm": {},
    }

    for paradigm, name, fn in ALL_TESTS:
        if paradigm not in results["by_paradigm"]:
            results["by_paradigm"][paradigm] = {
                "passed": 0,
                "failed": 0,
                "failures": [],
            }

    print("=" * 60)
    print("  LLM TEST SUITE — Text Operations Toolkit")
    print(f"  Total tests: {len(ALL_TESTS)}")
    print("=" * 60)

    for paradigm, name, fn in ALL_TESTS:
        bucket = results["by_paradigm"][paradigm]
        label  = f"[{paradigm.upper():<12}] {name}"
        sys.stdout.write(f"  {label:<65} ... ")
        sys.stdout.flush()

        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            print(f"PASS  ({elapsed:.1f}s)")
            results["passed"] += 1
            bucket["passed"] += 1
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"FAIL  ({elapsed:.1f}s)")
            tb = traceback.format_exc()
            bucket["failed"] += 1
            bucket["failures"].append((name, str(exc), tb))
            results["failed"] += 1

    total   = results["passed"] + results["failed"]
    overall = "ALL PASSED" if results["failed"] == 0 else f"{results['failed']} FAILED"

    print()
    print("=" * 60)
    print(f"  RESULTS: {results['passed']}/{total} passed   [{overall}]")
    print("=" * 60)
    print(f"  {'Paradigm':<14}  {'Passed':>6}  {'Failed':>6}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*6}")
    for paradigm, bucket in results["by_paradigm"].items():
        print(f"  {paradigm:<14}  {bucket['passed']:>6}  {bucket['failed']:>6}")
    print()

    if results["failed"] > 0:
        print("FAILED TESTS:")
        for paradigm, bucket in results["by_paradigm"].items():
            for name, msg, tb in bucket["failures"]:
                print(f"\n  [{paradigm.upper()}] {name}")
                print(f"  Error: {msg}")
                for line in tb.strip().splitlines()[-6:]:
                    print(f"    {line}")
        print()

    return results["failed"] == 0


if __name__ == "__main__":
    success = run_all_tests()
    print("Tests", "PASSED" if success else "FAILED")
    sys.exit(0 if success else 1)
