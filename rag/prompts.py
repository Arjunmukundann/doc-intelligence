MEMORY_QA_PROMPT = """You are a document question-answering assistant.

STRICT RULES:
1. Answer ONLY using the CONTEXT PASSAGES below
2. Do NOT use information from the conversation history to answer
3. Do NOT invent or assume anything not in the context
4. If the answer is not in the context, say exactly: "The document does not clearly provide this information."
5. Keep answers concise and factual
{history_block}
=== CONTEXT PASSAGES ===
{context}

Question: {question}

Answer:"""
def build_prompt_with_memory(question: str, context: str, session_id: str) -> str:
    from rag.memory import format_history_for_prompt
    history = format_history_for_prompt(session_id)
    history_block = f"Previous conversation:\n{history}\n" if history else ""
    return MEMORY_QA_PROMPT.format(
        history_block=history_block,
        context=context,
        question=question
    )