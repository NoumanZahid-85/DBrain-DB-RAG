"""RAGAS evaluation script.

Evaluate retrieved context and generated answers against the golden dataset.
The evaluator LLM is standardised on Groq ``openai/gpt-oss-120b`` (same model
as the RAG runtime), and evaluator embeddings use the local MiniLM provider —
no Gemini dependency, matching the runtime stack in ``backend``.

CI gate: ``.github/workflows/eval.yml`` runs this and fails when faithfulness
< 0.75 or answer_relevancy < 0.70.
"""

from __future__ import annotations

import json
import os
import sys
import time

import sys as _sys

# Add repo root so `backend.*` package imports resolve.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from backend import config
from backend.guardrails import filter_relevant, normalize_relevance
from backend.rag_chain import query_rag

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "last_results.json")

THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
}


def query_rag_with_retry(question: str, retries: int = 3, base_delay: int = 3) -> dict:
    """Wraps query_rag with a generic backoff for transient provider errors."""
    import re

    for i in range(retries):
        try:
            return query_rag(question)
        except Exception as e:  # noqa: BLE001
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate limit" in err_str.lower():
                match = re.search(r"retry in ([\d\.]+)s", err_str)
                wait_time = float(match.group(1)) + 1.5 if match else base_delay * (i + 1)
                print(f"    [429 Rate Limit] Retrying query in {wait_time:.1f}s... (Attempt {i+1}/{retries})")
                sys.stdout.flush()
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError(f"Failed to execute RAG query after {retries} retries due to rate limits.")


def _build_evaluators():
    """RAGAS LLM + embeddings on the same stack as the runtime.

    LLM: Groq gpt-oss-120b (controlled failover to any ChatGroq-compatible model).
    Embeddings: local MiniLM via ChromaDB's bundled ONNX engine (no API key).
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from backend.vectorstore import get_embed_model, get_embedding_model

    from langchain_groq import ChatGroq

    eval_llm_obj = ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
        max_tokens=config.MAX_OUTPUT_TOKENS,
    )
    print(f"Evaluator LLM: Groq ({config.GROQ_MODEL})")

    eval_embeddings_obj = get_embed_model()
    print(f"Evaluator Embeddings: {get_embedding_model()}")

    return (
        LangchainLLMWrapper(eval_llm_obj),
        LangchainEmbeddingsWrapper(eval_embeddings_obj),
    )


def run_evaluation(db_filter_eval: str | None = None) -> int:
    """Run RAGAS evaluation on the golden dataset. Returns process exit code."""
    if not os.path.exists(GOLDEN_PATH):
        print(f"Golden dataset not found at {GOLDEN_PATH}")
        sys.stdout.flush()
        return 1

    if not config.GROQ_API_KEY:
        print("GROQ_API_KEY not set — evaluation requires an LLM provider.")
        sys.stdout.flush()
        return 1

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    if db_filter_eval:
        golden = [g for g in golden if g.get("db") == db_filter_eval]

    limit = int(os.getenv("EVAL_LIMIT", "0"))
    if limit > 0:
        golden = golden[:limit]

    print(
        f"Evaluating {len(golden)} questions (limit={limit if limit > 0 else 'none'})"
        + (f" for: {db_filter_eval}" if db_filter_eval else " across all databases")
    )
    sys.stdout.flush()

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden:
        try:
            result = query_rag_with_retry(item["question"])
            questions.append(item["question"])
            answers.append(result["answer"])
            contexts.append([c["text"] for c in result.get("context_used", [])])
            ground_truths.append(item["ground_truth"])
            print(f"  OK [{item.get('db', '?'):<12}] {item['question'][:55]}...")
            sys.stdout.flush()
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL Failed for query: '{item['question']}'. Error: {e}")
            sys.stdout.flush()
        time.sleep(0.1)

    if not questions:
        print("No evaluation questions were processed successfully.")
        sys.stdout.flush()
        return 1

    from datasets import Dataset

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nRunning RAGAS scoring...")
    sys.stdout.flush()
    evaluator_llm, evaluator_embeddings = _build_evaluators()

    from ragas import RunConfig, evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    run_config = RunConfig(timeout=300)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    import math

    avg_scores: dict[str, float] = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        val = None
        try:
            val = scores[metric]
        except Exception:  # noqa: BLE001
            pass
        if val is None:
            val = getattr(scores, metric, None)

        if val is None:
            avg_scores[metric] = 0.0
        elif isinstance(val, (int, float)):
            avg_scores[metric] = 0.0 if math.isnan(val) else val
        elif hasattr(val, "__iter__"):
            valid = []
            for s in val:
                if s is not None:
                    try:
                        f_val = float(s)
                        if not math.isnan(f_val):
                            valid.append(f_val)
                    except (TypeError, ValueError):
                        pass
            avg_scores[metric] = sum(valid) / len(valid) if valid else 0.0
        else:
            avg_scores[metric] = 0.0

    print("\n" + "=" * 55)
    print(f"EVALUATION RESULTS {f'[{db_filter_eval}]' if db_filter_eval else '[all databases]'}")
    print("=" * 55)
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        suffix = f" (threshold >= {THRESHOLDS[metric]})" if metric in THRESHOLDS else ""
        print(f"  {metric:<18} {avg_scores[metric]:.3f}{suffix}")
    sys.stdout.flush()

    failed = False
    for metric, threshold in THRESHOLDS.items():
        if avg_scores[metric] < threshold:
            print(f"\n  [FAIL] {metric} = {avg_scores[metric]:.3f} < {threshold}")
            failed = True
    if not failed:
        print("\n  [PASS] All metrics above thresholds")

    output = {
        "faithfulness": avg_scores["faithfulness"],
        "answer_relevancy": avg_scores["answer_relevancy"],
        "context_precision": avg_scores["context_precision"],
        "db_filter": db_filter_eval,
        "n_questions": len(golden),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")
    sys.stdout.flush()

    return 1 if failed else 0


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run_evaluation(db_filter_eval=db))