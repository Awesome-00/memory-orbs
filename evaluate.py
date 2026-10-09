"""
Evaluate Memory Orbs ingestion and semantic retrieval on an isolated SQLite database.

Prerequisite: store.py must use DB_PATH for all SQLite connections:
    import os
    DB_PATH = os.environ.get("MEMORY_DB_PATH", "memory.db")
and each sqlite3.connect(...) call should use sqlite3.connect(DB_PATH).

Run from the project root:
    python evaluate.py
Optional:
    MEMORY_EVAL_TOP_K=5 python evaluate.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# Set the database path BEFORE importing project modules.
_PROJECT_DIR = Path(__file__).resolve().parent
_CASES_PATH = _PROJECT_DIR / "eval_cases.json"
_TOP_K = int(os.environ.get("MEMORY_EVAL_TOP_K", "5"))

_temp_dir = tempfile.TemporaryDirectory(prefix="memory_orbs_eval_")
os.environ["MEMORY_DB_PATH"] = str(Path(_temp_dir.name) / "evaluation.db")

# Import only after setting MEMORY_DB_PATH so store.py uses the isolated DB.
try:
    from store import init_db
    from ingest import ingest_message
    from retrieve import retrieve_memories
except Exception as exc:
    print(f"Could not import Memory Orbs modules: {exc}", file=sys.stderr)
    _temp_dir.cleanup()
    raise

def normalize(value):
    return " ".join(str(value).lower().split())

def result_text(result):
    # Retrieval currently returns dicts with a "text" key.
    if isinstance(result, dict):
        return str(result.get("text", ""))
    return str(result)

def contains_expected(results, expected_any):
    """True if at least one expected phrase appears in any returned memory."""
    if not expected_any:
        return False
    haystack = normalize(" ".join(result_text(r) for r in results))
    return any(normalize(phrase) in haystack for phrase in expected_any)

def main():
    if not _CASES_PATH.exists():
        raise FileNotFoundError(f"Missing evaluation cases: {_CASES_PATH}")

    payload = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if not cases:
        raise ValueError("eval_cases.json contains no cases")

    init_db()
    print(f"Using isolated evaluation database: {os.environ['MEMORY_DB_PATH']}")
    print(f"Ingesting {len(cases)} messages...")

    ingestion_ok = 0
    ingestion_failures = []
    for case in cases:
        # Unknown cases are queries only. Do not ingest their message, or the
        # test setup itself could teach the system the answer to the unknown query.
        if case.get("kind") == "unknown":
            continue
        try:
            ids = ingest_message(case["message"])
            if ids:
                ingestion_ok += 1
            else:
                ingestion_failures.append((case["id"], "no memory IDs returned"))
        except Exception as exc:
            ingestion_failures.append((case["id"], str(exc)))

    print(f"Recall messages producing at least one saved memory: {ingestion_ok}/{len(recall_cases) if "recall_cases" in locals() else sum(c.get("kind") == "recall" for c in cases)}")
    if ingestion_failures:
        print("\nIngestion failures:")
        for case_id, reason in ingestion_failures:
            print(f"  {case_id}: {reason}")

    recall_cases = [c for c in cases if c.get("kind") == "recall"]
    unknown_cases = [c for c in cases if c.get("kind") == "unknown"]
    recall_hits = 0
    unknown_correct = 0
    false_positives = []
    misses = []

    print("\nRetrieval results:")
    for case in cases:
        try:
            results = retrieve_memories(case["query"], top_k=_TOP_K)
        except Exception as exc:
            print(f"  {case['id']} ERROR: {exc}")
            if case.get("kind") == "recall":
                misses.append((case["id"], f"retrieval error: {exc}"))
            else:
                false_positives.append((case["id"], f"retrieval error: {exc}"))
            continue

        expected = case.get("expected_any", [])
        if case.get("kind") == "recall":
            hit = contains_expected(results, expected)
            if hit:
                recall_hits += 1
            else:
                misses.append((case["id"], case["query"]))
            print(f"  {case['id']} {'HIT' if hit else 'MISS'} | {case['query']}")
            for result in results:
                print(
                    f"      score={result.get('score', 0):.3f} "
                    f"text={result.get('text', '')}"
                )
        else:
            # Unknown-query success means no memories were returned above the configured threshold.
            correct = len(results) == 0
            if correct:
                unknown_correct += 1
            else:
                false_positives.append((case["id"], case["query"]))
            print(f"  {case['id']} {'CORRECT EMPTY' if correct else 'FALSE POSITIVE'} | {len(results)} result(s) | {case['query']}")

    recall_rate = recall_hits / len(recall_cases) if recall_cases else 0.0
    unknown_rate = unknown_correct / len(unknown_cases) if unknown_cases else 0.0
    total = len(recall_cases) + len(unknown_cases)
    overall = (recall_hits + unknown_correct) / total if total else 0.0

    print("\n=== SUMMARY ===")
    print(f"Recall hit rate:       {recall_hits}/{len(recall_cases)} = {recall_rate:.1%}")
    print(f"Unknown rejection:     {unknown_correct}/{len(unknown_cases)} = {unknown_rate:.1%}")
    print(f"Overall case accuracy: {recall_hits + unknown_correct}/{total} = {overall:.1%}")
    print(f"Top-k:                 {_TOP_K}")
    print(f"Embedding/retrieval threshold: configured in retrieve.py")

    if misses:
        print("\nRecall misses to inspect:")
        for case_id, query in misses:
            print(f"  {case_id}: {query}")
    if false_positives:
        print("\nUnknown-query false positives to inspect:")
        for case_id, query in false_positives:
            print(f"  {case_id}: {query}")

    print("\nNote: These are small local-model benchmark results, not a statistically robust general benchmark.")
    print("The evaluation database is temporary and will be removed when this process exits.")

if __name__ == "__main__":
    try:
        main()
        from store import get_all_memories

        print("\n=== STORED MEMORIES ===")
        for m in get_all_memories():
            print(
                f"id={m[0]} status={m[5]} superseded_by={m[6]} "
                f"type={m[2]} | {m[1]}"
                )

    finally:
        _temp_dir.cleanup()
