import json
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

client=Groq(api_key=os.getenv("GROQ_API_KEY"))
RERANK_PROMPT="""Rate how relevant this passage is for answering the question.
Question:{question}
Passage:{passage}

Reply with ONLY a JSON object:{{"score":0.0_to_1.0,"reason":"one sentence"}}
Score meaning:1.0=directly answers,0.5=partially relevant,0.0=irrelevant
"""

def rerank(question:str,retrieved:list[dict],top_k:int =5)-> list[dict]:
    scored=[]
    for item in retrieved:
        try:
            response=client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role":"user","content":RERANK_PROMPT.format(question=question,passage=item["text"][:400])}
                ],
                max_tokens=80
            )
            raw=response.choices[0].message.content.strip()
            result=json.loads(raw)
            item['rerank_score']=result.get("score",0.0)
        except Exception:
            item["rerank_score"]=item.get("score",0.0)

        scored.append(item)

    scored.sort(key=lambda x:x["rerank_score"],reverse=True)

    for i, item in enumerate(scored[:top_k],start=1):
        item["rank"]=i
    return scored[:top_k]