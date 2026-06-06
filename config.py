import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHUNK_SIZE=1000
CHUNK_OVERLAP=100

EMBEDDING_MODEL="all-mpnet-base-v2"
GROQ_MODEL="llama-3.3-70b-versatile"