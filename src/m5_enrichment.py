from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


_openai_client = None
_openai_disabled = False


def _get_openai_client():
    global _openai_client, _openai_disabled
    if _openai_disabled or not OPENAI_API_KEY:
        return None
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
                max_retries=0,
                timeout=30.0,
            )
        except Exception:
            _openai_disabled = True
            return None
    return _openai_client


def _handle_openai_error(e: Exception, task_name: str):
    global _openai_disabled
    err_str = str(e).lower()
    if (
        "credit_balance_exhausted" in err_str
        or "insufficient_quota" in err_str
        or "insufficient credits" in err_str
        or "error code: 402" in err_str
        or "429" in err_str
    ):
        if not _openai_disabled:
            print(f"  ⚠️  OpenAI quota không khả dụng ({err_str[:80]}...). Chuyển sang fallback mode.")
            _openai_disabled = True
    else:
        print(f"  ⚠️  OpenAI {task_name} failed: {e}")


def _extractive_summary(text: str) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


def _extractive_questions(text: str, n_questions: int = 3) -> list[str]:
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


def _extractive_context(text: str, document_title: str = "") -> str:
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


def _extractive_metadata(text: str) -> dict:
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.0,
            )
            summary = resp.choices[0].message.content.strip()
            if summary:
                return summary
        except Exception as e:
            _handle_openai_error(e, "summarize")

    return _extractive_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi bằng tiếng Việt mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            raw_lines = resp.choices[0].message.content.strip().split("\n")
            questions = [re.sub(r'^\s*(\d+[\.\)]\s*|-\s*)', '', q).strip() for q in raw_lines if q.strip()]
            if questions:
                return questions[:n_questions]
        except Exception as e:
            _handle_openai_error(e, "HyQA")

    return _extractive_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về đúng 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
                temperature=0.0,
            )
            context = resp.choices[0].message.content.strip()
            if context:
                return f"{context}\n\n{text}"
        except Exception as e:
            _handle_openai_error(e, "contextual")

    return _extractive_context(text, document_title)


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.0,
            )
            data = json.loads(resp.choices[0].message.content)
            if isinstance(data, dict):
                return data
        except Exception as e:
            _handle_openai_error(e, "metadata")

    return _extractive_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu ngắn gọn",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
                temperature=0.0,
            )
            data = json.loads(resp.choices[0].message.content)
            if isinstance(data, dict):
                return data
        except Exception as e:
            _handle_openai_error(e, "combined enrichment")

    return {
        "summary": _extractive_summary(text),
        "questions": _extractive_questions(text),
        "context": f"Trích từ {source}" if source else "",
        "metadata": _extractive_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []

    def build_combined(chunk: dict) -> EnrichedChunk:
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        result = _enrich_single_call(text, source)
        summary = result.get("summary", "")
        questions = result.get("questions", [])
        context_line = result.get("context", "")
        enriched_text = f"{context_line}\n\n{text}" if context_line else text
        auto_meta = result.get("metadata", {})
        return EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        )

    if use_combined:
        # Keep one request per chunk while overlapping network latency.
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i, item in enumerate(executor.map(build_combined, chunks)):
                enriched.append(item)
                if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                    print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)
        return enriched

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        summary = summarize_chunk(text) if "summary" in methods else ""
        questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
        enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
        auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
