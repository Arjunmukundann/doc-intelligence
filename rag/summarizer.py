from groq import Groq
import os 
from dotenv import load_dotenv
load_dotenv()

client=Groq(api_key=os.getenv("GROQ_API_KEY"))
MAP_PROMPT=""" Summarize the key information in this passage in 2-3 sentences.
Be factual and concide.Only include what is explicitly stated.

Passage:{passage}

Summary:"""

REDUCE_PROMPT=""" You have been given summaries of different secttions of a document.
Combine them into one coherent final summary.

Structure your summary as:

- main topic :(1 sentence)
- key points :(3-5 bullet points)
- conclusion :(1 sentence)

Section summaries:{summaries}

final summary

 """

def map_reduce_summarize(all_chunks :list[dict])->str:
    print(f"[summarizer] mapping {len(all_chunks)} chunks...")

    chunk_summaries=[]
    BATCH_SIZE=5
    grouped_chunks=[]

    for i in range(0,len(all_chunks),BATCH_SIZE):
        batch=all_chunks[i:i+BATCH_SIZE]
        combined_text="\n\n".join(chunk['text'][:500]
                                  for chunk in batch)
        
        grouped_chunks.append(combined_text)

        print(f"[summarizer] created {len(grouped_chunks)} grouped batches..." )
    

    for group in grouped_chunks:

        response=client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":MAP_PROMPT.format(passage=group)}],
            max_tokens=150,
        
        )
        summary=response.choices[0].message.content.strip()
        chunk_summaries.append(summary)

    
    return response.choices[0].message.content.strip()    

