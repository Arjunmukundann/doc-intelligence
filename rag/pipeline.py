from groq import Groq
import os
from rag.query_rewriter import rewrite_query
from rag.prompts import MEMORY_QA_PROMPT
from rag.hallucination_guard import check_grounding
from rag.rerank import rerank
from vector_store.manager import search_by_doc_id, get_all_chunks
from dotenv import load_dotenv
from rag.query_router import classify_query
from rag.summarizer import map_reduce_summarize
from rag.memory import add_message, format_history_for_prompt

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def run_rag_pipeline(question: str, doc_id: str, session_id: str = "default") -> dict:

    # ── ROUTE: summarize or normal QA ────────────────────────────
    query_type = classify_query(question)

    if query_type == "summarize":
        all_chunks = get_all_chunks(doc_id)
        summary = map_reduce_summarize(all_chunks)
        add_message(session_id, "user", question)
        add_message(session_id, "assistant", summary)
        return {
            "type": "summary",
            "question": question,
            "answer": summary,
            "grounding": {"is_grounded": True, "confidence": "high"},
            "sources": []
        }

    # ── STEP 1: Rewrite query ─────────────────────────────────────
    search_query = rewrite_query(question)

    # ── STEP 2: Retrieve from FAISS ───────────────────────────────
    # raw_results shape: [{"text":"..","page":1,"source":"..","score":0.87}, ...]
    raw_results = search_by_doc_id(doc_id, search_query, k=10)

    raw_results=[r for r in raw_results if r["score"]>=0.30]

    if not raw_results:
        return {
            "type": "qa",
            "question": question,
            "search_query": search_query,
            "answer": "The document does not clearly provide this information.",
            "grounding": {"is_grounded": False, "confidence": "low"},
            "sources": []
        }

    # ── STEP 3: Wrap for reranker ─────────────────────────────────
    # reranker expects: [{"chunk": {...}, "score": float, "rank": int}]
    # raw_results is flat so we wrap each dict under "chunk" key
    wrapped_for_rerank = [
        {
            "chunk": chunk,
            "score": chunk.get("score", 0.0),
            "rank":  i + 1
        }
        for i, chunk in enumerate(raw_results)
    ]

    # ── STEP 4: Rerank ────────────────────────────────────────────
    # reranked shape: [{"chunk":{..}, "score":float, "rank":int, "rerank_score":float}]
    # ── STEP 4: Rerank ────────────────────────────────────────────
    reranked = rerank(question, wrapped_for_rerank, top_k=5)

    # Use rerank_score not score — reranker assigns this directly
    # If reranker failed (all fallback to 0), use original FAISS score
    best_rerank = max((r.get("rerank_score", 0) for r in reranked), default=0)
    best_faiss   = max((r.get("score", 0)       for r in reranked), default=0)
    top_score    = best_rerank if best_rerank > 0 else best_faiss

    print(f"[pipeline] top_score={top_score:.3f} (rerank={best_rerank:.3f} faiss={best_faiss:.3f})")

    if top_score < 0.20:   # very low threshold — only blocks truly irrelevant
        return {
            "type": "qa",
            "question": question,
            "search_query": search_query,
            "answer": "The document does not clearly provide this information.",
            "grounding": {"is_grounded": False, "confidence": "low"},
            "sources": []
        }
    # ── STEP 5: Build context ─────────────────────────────────────
    # Bug fixed: was chunk["text"], now r["chunk"]["text"]
    context = "\n\n".join(
        f"[Page {r['chunk']['page']}] {r['chunk']['text']}"
        for r in reranked
    )

    # ── STEP 6: Build prompt with memory ─────────────────────────
    history = format_history_for_prompt(session_id)
    if history :
        history_block = f"Previous conversation:\n{history}\n" 
    else: 
        history_block='' 
        

    prompt = MEMORY_QA_PROMPT.format(
        history_block=history_block,
        context=context,
        question=question
    )
    print("\n--- PROMPT BEING SENT ---")
    print(prompt[:800])
    print("--- END PROMPT ---\n")
    # ── STEP 7: Call LLM ─────────────────────────────────────────
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    answer = response.choices[0].message.content.strip()

    # ── STEP 8: Hallucination guard ───────────────────────────────
    # Bug fixed: pass reranked (has "chunk" key) — matches what guard expects
    grounding = check_grounding(answer, reranked)
    
    if not grounding["is_grounded"]:
        answer = "The document does not clearly provide this information."
    

    # ── STEP 9: Save to memory ────────────────────────────────────
    add_message(session_id, "user", question)
    add_message(session_id, "assistant", answer)

    # ── STEP 10: Format sources ───────────────────────────────────
    # Bug fixed: was chunk.get("page") on wrong shape
    # now correctly accesses r["chunk"]["page"]
    sources = [
        {
            "rank":         r["rank"],
            "page":         r["chunk"].get("page"),
            "source":       r["chunk"].get("source"),
            "rerank_score": r.get("rerank_score"),
            "preview":      r["chunk"]["text"][:200]
        }
        for r in reranked
    ]

    return {
        "type":         "qa",
        "question":     question,
        "search_query": search_query,
        "answer":       answer,
        "grounding":    grounding,
        "sources":      sources
    }