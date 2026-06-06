SUMMARY_KEYWORDS=[
 "summarize", "summary", "overview", "what is this about",
    "what is the document about", "briefly", "main points",
    "key points", "give me an idea"

]


def classify_query(question:str)->str:
    q=question.lower()
    if any(kw in q for kw in SUMMARY_KEYWORDS):
        return "summarize"
    return "qa"