def check_grounding(answer:str,retrieved_chunks:list[dict]) ->dict:
    if "does not clearly" in answer.lower():
        return{"is_grounded":True,"confidence":"high"}
    context_text=" ".join(chunk['chunk']["text"] for chunk in retrieved_chunks).lower()
    answer_words=[w.lower().strip(".,!?") for w in answer.split() if len(w)>5]

    if not answer_words:
        return{"is_grounded":False,"confidence":"low"}
    
    found=sum(1 for word in answer_words if word in context_text)
    ratio=found/len(answer_words)


    if ratio >0.25:
        return{"is_grounded":True,"confidence":"high"}
    elif ratio >0.10:
        return{"is_grounded":True,"confidence":"medium"}
    else:
        return{"is_grounded":False,"confidence":"low"}