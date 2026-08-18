from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


import math


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        eval_llm = None
        eval_embeddings = None
        if OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                from langchain_community.embeddings import HuggingFaceEmbeddings
                eval_llm = ChatOpenAI(
                    model=LLM_MODEL,
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
                    max_tokens=1024,
                    temperature=0.0,
                    request_timeout=30.0,
                )
                eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            except Exception as e:
                print(f"  ⚠️  Could not init custom LLM/embeddings for RAGAS: {e}")

        eval_kwargs = {
            "dataset": dataset,
            "metrics": [faithfulness, answer_relevancy, context_precision, context_recall],
            "raise_exceptions": False,
        }
        if eval_llm is not None:
            eval_kwargs["llm"] = eval_llm
        if eval_embeddings is not None:
            eval_kwargs["embeddings"] = eval_embeddings

        result = evaluate(**eval_kwargs)
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            ctx = row.get("contexts", [])
            if isinstance(ctx, (list, tuple)):
                ctx_list = [str(c) for c in ctx]
            else:
                ctx_list = [str(ctx)]
            per_question.append(
                EvalResult(
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    contexts=ctx_list,
                    ground_truth=str(row.get("ground_truth", "")),
                    faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                    answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                    context_precision=_safe_float(row.get("context_precision", 0.0)),
                    context_recall=_safe_float(row.get("context_recall", 0.0)),
                )
            )
        return {
            "faithfulness": _safe_float(result.get("faithfulness", 0.0)),
            "answer_relevancy": _safe_float(result.get("answer_relevancy", 0.0)),
            "context_precision": _safe_float(result.get("context_precision", 0.0)),
            "context_recall": _safe_float(result.get("context_recall", 0.0)),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        per_question = [
            EvalResult(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=gt,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
            )
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    analyzed = []
    for res in eval_results:
        metrics_dict = {
            "faithfulness": _safe_float(res.faithfulness),
            "answer_relevancy": _safe_float(res.answer_relevancy),
            "context_precision": _safe_float(res.context_precision),
            "context_recall": _safe_float(res.context_recall),
        }
        avg_score = sum(metrics_dict.values()) / 4.0
        worst_metric = min(metrics_dict, key=metrics_dict.get)
        worst_score = metrics_dict[worst_metric]
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric, ("Unknown failure", "Review pipeline configuration")
        )
        analyzed.append({
            "question": res.question,
            "worst_metric": worst_metric,
            "score": float(worst_score),
            "avg_score": float(avg_score),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to the reports directory."""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
