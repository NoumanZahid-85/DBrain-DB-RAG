"""
RAGAS evaluation script.
Evaluate retrieved context and generated answers against the golden dataset.
"""

import json
import sys
import os
import time

# Insert backend directory to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from rag_chain import query_rag

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "last_results.json")

# Quality thresholds — CI fails if below these
THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
}


def query_rag_with_retry(question: str, retries: int = 5, base_delay: int = 5) -> dict:
    """Wraps query_rag with automatic retry and parsing of Gemini 429 quota exception retry delay."""
    import re
    for i in range(retries):
        try:
            return query_rag(question)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                match = re.search(r"retry in ([\d\.]+)s", err_str)
                wait_time = float(match.group(1)) + 1.5 if match else base_delay * (i + 1)
                print(f"    [429 Rate Limit] Retrying query in {wait_time:.1f}s... (Attempt {i+1}/{retries})")
                sys.stdout.flush()
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError(f"Failed to execute RAG query after {retries} retries due to rate limits.")


def run_evaluation(db_filter_eval: str | None = None):
    """
    Run RAGAS evaluation on golden dataset.
    db_filter_eval: restrict to one database subset for targeted testing.
    """
    if not os.path.exists(GOLDEN_PATH):
        print(f"Golden dataset not found at {GOLDEN_PATH}")
        sys.stdout.flush()
        return 1

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    if db_filter_eval:
        golden = [g for g in golden if g.get("db") == db_filter_eval]

    limit = int(os.getenv("EVAL_LIMIT", "0"))
    if limit > 0:
        golden = golden[:limit]

    if db_filter_eval:
        print(f"Evaluating {len(golden)} questions (limit={limit if limit > 0 else 'none'}) for: {db_filter_eval}")
    else:
        print(f"Evaluating all {len(golden)} questions (limit={limit if limit > 0 else 'none'}) across all databases")
    sys.stdout.flush()

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden:
        try:
            result = query_rag_with_retry(item["question"])
            questions.append(item["question"])
            answers.append(result["answer"])
            contexts.append([c["text"] for c in result.get("context_used", [])])
            ground_truths.append(item["ground_truth"])
            print(f"  OK [{item.get('db','?'):<12}] {item['question'][:55]}...")
            sys.stdout.flush()
        except Exception as e:
            print(f"  FAIL Failed for query: '{item['question']}'. Error: {e}")
            sys.stdout.flush()
        time.sleep(0.1)  # Brief pause between queries

    if not questions:
        print("No evaluation questions were processed successfully.")
        sys.stdout.flush()
        return 1

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nRunning RAGAS scoring...")
    sys.stdout.flush()
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from vectorstore import get_embedding_model

    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true" or not os.getenv("GOOGLE_API_KEY")
    if use_ollama:
        try:
            from langchain_ollama import ChatOllama
            eval_llm_obj = ChatOllama(
                model="gemma4:12b-it-qat",
                temperature=0.0
            )
            print("Evaluator LLM: Using local Ollama (gemma4:12b-it-qat)")
        except ImportError:
            from langchain_google_genai import ChatGoogleGenerativeAI
            eval_llm_obj = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.0
            )
            print("Evaluator LLM: Fallback to Gemini (gemini-1.5-flash) - langchain-ollama not installed")
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        eval_llm_obj = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.0
        )
        print("Evaluator LLM: Using Google Gemini (gemini-1.5-flash)")

    evaluator_llm = LangchainLLMWrapper(eval_llm_obj)

    if not os.getenv("GOOGLE_API_KEY"):
        # Local fallback only if GOOGLE_API_KEY is not set
        from langchain_community.embeddings import HuggingFaceEmbeddings
        eval_embeddings_obj = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Evaluator Embeddings: Using local HuggingFace (all-MiniLM-L6-v2)")
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        eval_embeddings_obj = GoogleGenerativeAIEmbeddings(
            model=get_embedding_model(),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print(f"Evaluator Embeddings: Using Gemini ({get_embedding_model()})")

    evaluator_embeddings = LangchainEmbeddingsWrapper(eval_embeddings_obj)

    from ragas import RunConfig
    run_config = RunConfig(timeout=300)

    # Evaluate dataset using RAGAS metrics
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    # Calculate average scores robustly across Ragas versions and handle NaNs/None
    import math
    avg_scores = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        val = None
        # Try dict-like access
        try:
            val = scores[metric]
        except Exception:
            pass
        # Try attribute access
        if val is None:
            val = getattr(scores, metric, None)

        if val is None:
            avg_scores[metric] = 0.0
        elif isinstance(val, (int, float)):
            avg_scores[metric] = 0.0 if math.isnan(val) else val
        elif hasattr(val, "__iter__"):
            # If it is a list or Series, compute the average of valid non-NaN scores
            valid_scores = []
            for s in val:
                if s is not None:
                    try:
                        f_val = float(s)
                        if not math.isnan(f_val):
                            valid_scores.append(f_val)
                    except ValueError:
                        pass
            avg_scores[metric] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        else:
            avg_scores[metric] = 0.0

    print("\n" + "=" * 55)
    print(f"EVALUATION RESULTS {f'[{db_filter_eval}]' if db_filter_eval else '[all databases]'}")
    print("=" * 55)
    print(f"  Faithfulness      : {avg_scores['faithfulness']:.3f}  (threshold >= {THRESHOLDS['faithfulness']})")
    print(f"  Answer Relevancy  : {avg_scores['answer_relevancy']:.3f}  (threshold >= {THRESHOLDS['answer_relevancy']})")
    print(f"  Context Precision : {avg_scores['context_precision']:.3f}")
    sys.stdout.flush()

    failed = False
    for metric, threshold in THRESHOLDS.items():
        if avg_scores[metric] < threshold:
            print(f"\n  [FAIL] {metric} = {avg_scores[metric]:.3f} < {threshold}")
            sys.stdout.flush()
            failed = True

    if not failed:
        print(f"\n  [PASS] All metrics above thresholds")
        sys.stdout.flush()

    # Save results
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
    exit_code = run_evaluation(db_filter_eval=db)
    sys.exit(exit_code)
