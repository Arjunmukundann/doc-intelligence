from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from config import EMBEDDING_MODEL

model=SentenceTransformer(EMBEDDING_MODEL)

def create_embeddings(chunks):
    texts = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        elif isinstance(chunk, str):
            text = chunk
        else:
            text = getattr(chunk, "text", "")
        text = text.replace('\x00', '').strip()
        # Clean — ensure it's a non-empty string
        if text and len(text) >= 5:
            texts.append(text)
        else:
            texts.append("empty placeholder") # placeholder so index stays aligned

    embeddings = model.encode(
        texts,
        convert_to_tensor=False,
        normalize_embeddings=True,
        batch_size=32
    )
    return np.array(embeddings).astype("float32")



def create_faiss_index(embeddings):
    dimension=len(embeddings[0])
    index=faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def search_similar_chunks(index, chunks, query, k=3):

    if isinstance(query, str):
        query = [query]

    query_embedding_model = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores,indices=index.search(query_embedding_model,k)

    retrieved_chunks = [
        chunks[i] for i in indices[0]
    ]

    return retrieved_chunks, scores
