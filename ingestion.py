from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE,CHUNK_OVERLAP

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import re

def clean_text(text: str) -> str:
    text = text.replace('\x00', '')
    # Remove other non-printable control characters (keep newlines and tabs)
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

SKIP_PAGES_AFTER=520
def ingest_document(pdf_path: str) -> list:
    reader = PdfReader(pdf_path)
    total_pages=len(reader.pages)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        if page_num>SKIP_PAGES_AFTER:
            continue
        if page_num<25:
            continue    
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            continue

        # Clean null bytes before splitting
        page_text = clean_text(page_text)

        split_chunks = text_splitter.split_text(page_text)

        for raw_chunk in split_chunks:
            if not raw_chunk:
                continue
            cleaned = clean_text(raw_chunk)  # clean again after splitting
            if len(cleaned) < 20:
                continue
            # Final check — must be encodable as UTF-8
            try:
                cleaned.encode('utf-8')
            except UnicodeEncodeError:
                continue

            all_chunks.append({
                "text":        cleaned,
                "page":        page_num,
                "source":      pdf_path,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    print(f"[parser] Extracted {len(all_chunks)} chunks from '{pdf_path}'")
    return all_chunks

