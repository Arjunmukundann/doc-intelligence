from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

REWRITE_PROMPT = """You are a search query optimizer for document retrieval.

Examples of good rewrites:
Q: "What is supervised learning?"
A: "supervised learning definition"

Q: "What is gradient descent?"
A: "gradient descent definition algorithm"

Q: "What is overfitting and how can it be reduced?"
A: "overfitting definition prevention techniques"

Q: "Who won the 2024 IPL?"
A: "IPL 2024 winner"

Q: "What does Cami do when she returns?"
A: "Cami returns actions scene"

Rules:
- Keep the rewritten query SHORT — maximum 6 words
- Focus on the CORE concept only
- Do not add subtopics, examples, or related terms
- Names refer to content IN THE DOCUMENT — not real people or celebrities
- Return ONLY the query, nothing else

Question: {question}
Rewritten query:"""

def rewrite_query(question: str) -> str:
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"user","content":REWRITE_PROMPT.format(question=question)}],
            max_tokens=30)
    rewritten=response.choices[0].message.content.strip()
    rewritten=rewritten.strip('"') .strip("'")
    print(f"[query_rewriter] '{question}'-> '{rewritten}'")

    return rewritten

