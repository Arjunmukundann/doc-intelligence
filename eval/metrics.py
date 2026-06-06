def retrieval_precision(sources:list[dict],relevant_pages:list[int]) -> float:
    if not relevant_pages:
        return 1.0
    retrieved_pages=[s["page"] for s in sources]
    hits=sum(1 for p in relevant_pages if p in retrieved_pages)
    return hits/len(relevant_pages)

def grounding_score(answer:str,context:str)-> float:
    if "does not clearly" in answer.lower():
        return 1.0
    answer_words=[w.lower().strip(".,!?") for w in answer.split() if len(w)>5]
    if not answer_words:
        return 0.0
    found=sum(1 for w in answer_words if w in context.lower())
    return found/len(answer_words)
