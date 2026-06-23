"""
Retrieval eval harness (known-item / "needle" retrieval).

For each sampled chunk we generate a question that the chunk answers, then check
whether retrieval surfaces that exact chunk in the top-k. The eval set is cached
to eval_set.json so every strategy is scored on the *same* questions (fair A/B).

Run:
    .venv/bin/python -m rag.eval                 # score the registered strategies
    .venv/bin/python -m rag.eval --rebuild       # regenerate the eval set
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Callable, Dict, List

from dotenv import load_dotenv

load_dotenv()

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")
SAMPLE_PER_COURSE = 8
MIN_CHUNK_CHARS = 250


def _supabase():
    from supabase import create_client

    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


# ---------------------------------------------------------------------------
# Eval set (cached)
# ---------------------------------------------------------------------------
def build_eval_set(rebuild: bool = False) -> List[Dict]:
    if os.path.exists(EVAL_SET_PATH) and not rebuild:
        return json.load(open(EVAL_SET_PATH))

    from providers import structured_call

    sb = _supabase()
    courses = [c["course_id"] for c in sb.table("courses").select("course_id").execute().data]
    items: List[Dict] = []
    for cid in courses:
        rows = (sb.table("embeddings")
                .select("doc_name, chunk_id, content")
                .eq("course_id", cid).limit(500).execute().data) or []
        good = [r for r in rows
                if len(r.get("content", "")) > MIN_CHUNK_CHARS and "[IMAGE" not in r["content"]]
        if not good:
            continue
        step = max(1, len(good) // SAMPLE_PER_COURSE)
        sampled = good[::step][:SAMPLE_PER_COURSE]
        for r in sampled:
            try:
                q = structured_call(
                    [{"role": "user", "content": (
                        "Write ONE specific exam-style question that THIS passage directly answers. "
                        "Do not reference 'the passage'. Return JSON {\"question\": \"...\"}.\n\n"
                        f"PASSAGE:\n{r['content'][:1200]}"
                    )}],
                    schema={"type": "object", "properties": {"question": {"type": "string"}},
                            "required": ["question"]},  # noqa: E127
                    tool_name="q", max_tokens=200,
                )
                items.append({
                    "course_id": cid,
                    "doc_name": r["doc_name"],
                    "chunk_id": r["chunk_id"],
                    "question": q["question"].strip(),
                })
            except Exception as e:  # noqa: BLE001
                print(f"  skip (gen failed): {e}")
        print(f"  {cid}: {len([i for i in items if i['course_id'] == cid])} questions")

    json.dump(items, open(EVAL_SET_PATH, "w"), indent=2)
    print(f"Saved {len(items)} eval items -> {EVAL_SET_PATH}")
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def run_eval(retrieve: Callable[[str, str, int], List[Dict]],
             label: str, k_values=(1, 5, 10), top_k: int = 10) -> Dict:
    items = build_eval_set()
    recalls = {k: [] for k in k_values}
    rrs: List[float] = []
    for it in items:
        results = retrieve(it["question"], it["course_id"], top_k) or []
        rank = None
        for i, r in enumerate(results):
            if r.get("doc_name") == it["doc_name"] and r.get("chunk_id") == it["chunk_id"]:
                rank = i + 1
                break
        for k in k_values:
            recalls[k].append(1.0 if (rank and rank <= k) else 0.0)
        rrs.append(1.0 / rank if rank else 0.0)

    out = {f"recall@{k}": round(statistics.mean(recalls[k]), 3) for k in k_values}
    out["mrr"] = round(statistics.mean(rrs), 3)
    out["n"] = len(items)
    print(f"[{label:14}] " + "  ".join(f"{k}={v}" for k, v in out.items()))
    return out


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def baseline_retrieve(question: str, course_id: str, top_k: int) -> List[Dict]:
    """Current behaviour: plain local embedding + pgvector cosine search."""
    from providers import make_client
    from vector_store import VectorStore

    qvec = make_client().embeddings.create(model="local", input=[question]).data[0].embedding
    return VectorStore().query(course_id, qvec, top_k=top_k) or []


def prefixed_retrieve(question: str, course_id: str, top_k: int) -> List[Dict]:
    """Vector search with the BGE query-instruction prefix (asymmetric)."""
    from providers import embed_query
    from vector_store import VectorStore

    return VectorStore().query(course_id, embed_query(question), top_k=top_k) or []


def hybrid_retrieve(question: str, course_id: str, top_k: int) -> List[Dict]:
    """Dense (BGE-prefixed) + sparse (FTS) fused with RRF."""
    from rag.retrieval import hybrid_retrieve as _hybrid

    return _hybrid(question, course_id, top_k=top_k)


def reranked_retrieve(question: str, course_id: str, top_k: int) -> List[Dict]:
    """Hybrid candidates reordered by a cross-encoder reranker."""
    from rag.retrieval import hybrid_retrieve as _hybrid
    from rag.rerank import rerank

    candidates = _hybrid(question, course_id, top_k=20, candidates=24)
    return rerank(question, candidates, top_k=top_k)


STRATEGIES: Dict[str, Callable[[str, str, int], List[Dict]]] = {
    "baseline": baseline_retrieve,
    "prefixed": prefixed_retrieve,
    "hybrid": hybrid_retrieve,
    "reranked": reranked_retrieve,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--only", help="comma-separated strategy names")
    args = ap.parse_args()

    if args.rebuild:
        build_eval_set(rebuild=True)

    names = args.only.split(",") if args.only else list(STRATEGIES)
    for name in names:
        run_eval(STRATEGIES[name], label=name)
