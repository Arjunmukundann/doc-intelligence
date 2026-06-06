# vector_store.py — add these to your existing file

import os
import json
from vector_store.vector_store import create_embeddings, create_faiss_index, search_similar_chunks
import faiss
INDEX_DIR = "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# Store indexes and chunks in memory
_indexes = {}
_chunks = {}


def save_index(doc_id, index, chunks):
    """Save FAISS index + chunks to disk"""
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, f"{INDEX_DIR}/{doc_id}.faiss")
    with open(f"{INDEX_DIR}/{doc_id}_chunks.json", "w") as f:
        json.dump(chunks, f)
    _indexes[doc_id] = index
    _chunks[doc_id] = chunks


def build_index(doc_id: str, chunks) -> int:

    chunk_dicts = []
    for c in chunks:
        if isinstance(c, dict):
            chunk_dicts.append(c)
        else:
            
            chunk_dicts.append({
                "text":        c.text,
                "page":        c.page,
                "source":      c.source,
                "chunk_index": c.chunk_index,
            })

    if not chunk_dicts:
        raise ValueError(f"No chunks to index for doc: {doc_id}")

    
    embeddings = create_embeddings(chunk_dicts)
    index      = create_faiss_index(embeddings)

    
    save_index(doc_id, index, chunk_dicts)

    print(f"[manager] ✓ Built index for '{doc_id}': {len(chunk_dicts)} chunks")
    return len(chunk_dicts)

def load_index(doc_id):
    """Load a saved index from disk into memory"""
    index_path = f"{INDEX_DIR}/{doc_id}.faiss"
    chunks_path = f"{INDEX_DIR}/{doc_id}_chunks.json"
    if not os.path.exists(index_path):
        return False
    _indexes[doc_id] = faiss.read_index(index_path)
    with open(chunks_path) as f:
        _chunks[doc_id] = json.load(f)
    return True

def delete_index(doc_id):
    """Remove index from memory and disk"""
    _indexes.pop(doc_id, None)
    _chunks.pop(doc_id, None)
    for path in [f"{INDEX_DIR}/{doc_id}.faiss", f"{INDEX_DIR}/{doc_id}_chunks.json"]:
        if os.path.exists(path):
            os.remove(path)

def is_ready(doc_id):
    return doc_id in _indexes

def search_by_doc_id(doc_id: str, query: str, k: int = 5) -> list[dict]:
    if doc_id not in _indexes:
        if not load_index(doc_id):
            raise ValueError(f"No index found for doc: {doc_id}")

    retrieved, scores = search_similar_chunks(
        _indexes[doc_id],
        _chunks[doc_id],
        query,
        k
    )

    results = []
    for chunk, dist in zip(retrieved, scores[0]):
        score = float(1 / (1 + dist))

        chunk_with_score = dict(chunk)      
        chunk_with_score["score"] = float(score)
        results.append(chunk_with_score)

    return results

def get_all_chunks(doc_id):

    if doc_id not in _chunks:

        if not load_index(doc_id):
            raise ValueError(
                f"No chunks found for doc: {doc_id}"
            )

    return _chunks[doc_id]
def search_all_documents(query: str,k: int=10)-> list[dict]:
    all_results=[]
    for doc_id in _indexes.keys():
        results=search_by_doc_id(doc_id, query, k=k)
        for r in results:
            r["doc_id"]=doc_id
        all_results.extend(results)
    all_results.sort(key=lambda x: x["score"],reverse=True)

    return all_results[:k]


