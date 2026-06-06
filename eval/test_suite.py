# eval/test_suite.py
from rag.pipeline import run_rag_pipeline
from eval.metrics import retrieval_precision
from vector_store.manager import load_index, build_index
from ingestion import ingest_document
import os

TEST_CASES = [
    {
        "question": "What is supervised learning?",
        "relevant_pages": [27, 28],
        "should_answer": True
    },
    {
        "question": "What is the difference between classification and regression?",
        "relevant_pages": [28, 29],
        "should_answer": True
    },
    {
        "question": "What is gradient descent?",
        "relevant_pages": [113, 114],
        "should_answer": True
    },
    {
        "question": "What is overfitting and how can it be reduced?",
        "relevant_pages": [146, 147],
        "should_answer": True
    },
    {
        "question": "What is a confusion matrix?",
        "relevant_pages": [106, 107],
        "should_answer": True
    },
    {
        "question": "What is cross validation?",
        "relevant_pages": [75, 76],
        "should_answer": True
    },
    {
        "question": "What is a decision tree?",
        "relevant_pages": [171, 172],
        "should_answer": True
    },
    {
        "question": "What is the purpose of a training set and test set?",
        "relevant_pages": [36, 37],
        "should_answer": True
    },
    {
        "question": "Who won the 2024 IPL tournament?",
        "relevant_pages": [],
        "should_answer": False
    },
    {
        "question": "What is the capital of Brazil?",
        "relevant_pages": [],
        "should_answer": False
    },
]


# eval/test_suite.py
from rag.memory import clear_session   # add this import

def run_eval(doc_id: str):
    results = []
    total = len(TEST_CASES)
    print(f"\nRunning {total} test cases...\n")

    for i, test in enumerate(TEST_CASES, start=1):
        print(f"[{i}/{total}] {test['question'][:60]}")

        # Clear memory before each test so questions don't affect each other
        clear_session("eval-session")

        result = run_rag_pipeline(
            question=test["question"],
            doc_id=doc_id,
            session_id="eval-session"
        )

        answered = "does not clearly" not in result["answer"].lower()
        correct_behavior = answered == test["should_answer"]

        if test["relevant_pages"]:
            precision = retrieval_precision(
                result["sources"],
                test["relevant_pages"]
            )
        else:
            precision = 1.0 if not answered else 0.0

        results.append({
            "question":         test["question"],
            "correct_behavior": correct_behavior,
            "precision":        precision,
            "grounding":        result["grounding"]["confidence"],
            "answer":           result["answer"][:100],
        })

    correct = sum(1 for r in results if r["correct_behavior"])
    avg_precision = sum(r["precision"] for r in results) / total

    print(f"\n{'='*55}")
    print(f"EVAL RESULTS")
    print(f"{'='*55}")
    print(f"Correct behavior : {correct}/{total}")
    print(f"Avg precision    : {avg_precision:.2f}")
    print(f"\nPer question breakdown:")
    for r in results:
        status = "✓" if r["correct_behavior"] else "✗"
        print(f"  {status} [{r['grounding']:6}] p={r['precision']:.2f} | {r['question'][:45]}")
        print(f"         Answer: {r['answer'][:80]}")

    # ── Print summary ─────────────────────────────────────────
    correct = sum(1 for r in results if r["correct_behavior"])
    avg_precision = sum(r["precision"] for r in results) / total

    print(f"\n{'='*55}")
    print(f"EVAL RESULTS")
    print(f"{'='*55}")
    print(f"Correct behavior : {correct}/{total}")
    print(f"Avg precision    : {avg_precision:.2f}")
    print(f"\nPer question breakdown:")
    for r in results:
        status = "✓" if r["correct_behavior"] else "✗"
        print(f"  {status} [{r['grounding']:6}] p={r['precision']:.2f} | {r['question'][:45]}")
        print(f"         Answer: {r['answer'][:80]}")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    PDF    = r"C:\Users\Arjun\OneDrive\Desktop\mediretriver\data\Hands On Machine Learning with Scikit Learn and TensorFlow.pdf"
    DOC_ID = "eval-doc-002"

    # Load existing index if already built, otherwise build fresh
    print(f"Loading index for '{DOC_ID}'...")
    loaded = load_index(DOC_ID)

    if not loaded:
        print("No index found — building fresh index...")
        chunks = ingest_document(PDF)
        print(f"Chunks returned: {len(chunks)}")
        if len(chunks) > 0:
            print(f"First chunk type: {type(chunks[0])}")
            print(f"First chunk: {chunks[0]}")
        else:
            print("NO CHUNKS — check PDF filename and parser")

        build_index(DOC_ID, chunks)
        print(f"Index built: {len(chunks)} chunks")
    else:
        print("Index loaded from disk")

    run_eval(DOC_ID)