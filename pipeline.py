from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

def build_prompt(context, query):

    prompt = f"""
You are an AI assistant.

Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt


def generate_response(query, retrieved_chunks):

    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

    prompt = build_prompt(context, query)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content